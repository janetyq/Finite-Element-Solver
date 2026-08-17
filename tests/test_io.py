"""Round-trip tests for fem.io persistence.

Meshes go to JSON, solutions to a pickle-free npz archive. The point of these
tests is that a save/load cycle preserves everything a caller depends on --
value arrays, mesh geometry, mesh class and n_components -- and that the load path never
falls back to pickle.
"""
import meshio
import numpy as np
import pytest

from fem.integrators import ThetaMethod
from fem.io import _mesh_from_meshio, load_mesh, save_mesh, save_solution
from fem.mesh.mesh import Mesh
from fem.mesh.structured import create_box_mesh
from fem.numerics import bump_function
from fem.problem import heat
from fem.solution import (
    BucklingSolution, ElasticSolution, FieldSolution, Solution, TransientSolution,
)


def test_mesh_json_round_trip(make_unit_square, tmp_path):
    """Geometry survives a JSON save/load unchanged."""
    mesh = make_unit_square(6)
    path = tmp_path / "mesh.json"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert np.allclose(loaded.vertices, mesh.vertices)
    assert np.array_equal(loaded.elements, mesh.elements)
    assert np.array_equal(loaded.boundary, mesh.boundary)


def test_mesh_load_rebuilds_the_requested_class(make_unit_square, tmp_path):
    """`cls` controls the reconstructed type."""
    mesh = make_unit_square(6)
    path = tmp_path / "mesh.json"
    mesh.save(path)

    assert type(load_mesh(path)) is Mesh
    assert type(Mesh.load(path)) is Mesh


# --- standard formats via meshio --------------------------------------------

def test_mesh_vtu_round_trip(make_unit_square, tmp_path):
    """A 2D mesh survives a VTK .vtu save/load: the 3D-padded points come back
    reduced to 2D, and the boundary is re-derived to the same facets."""
    mesh = make_unit_square(6)
    path = tmp_path / "mesh.vtu"

    mesh.save(path)
    loaded = Mesh.load(path)

    assert loaded.spatial_dim == 2
    assert np.allclose(loaded.vertices, mesh.vertices)
    assert np.array_equal(loaded.elements, mesh.elements)
    assert np.array_equal(np.sort(loaded.boundary, axis=0), np.sort(mesh.boundary, axis=0))


def test_mesh_msh_round_trip_3d(tmp_path):
    """A 3D tet mesh survives a Gmsh .msh save/load unchanged."""
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(3, 3, 3))
    path = tmp_path / "mesh.msh"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert loaded.spatial_dim == 3
    assert np.allclose(loaded.vertices, mesh.vertices)
    assert np.array_equal(loaded.elements, mesh.elements)


def test_from_meshio_reduces_padded_3d_points():
    """Triangles with an all-zero z column import as a 2D mesh."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])

    mesh = _mesh_from_meshio(m)

    assert mesh.spatial_dim == 2
    assert np.allclose(mesh.vertices, points[:, :2])


def test_from_meshio_rejects_embedded_surface():
    """A surface mesh with nonzero out-of-plane coordinates is out of scope."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    m = meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])

    with pytest.raises(NotImplementedError):
        _mesh_from_meshio(m)


def test_from_meshio_prunes_unreferenced_vertices():
    """An isolated point no element references is dropped and indices renumber."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    m = meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])

    mesh = _mesh_from_meshio(m)

    assert len(mesh.vertices) == 3
    assert np.allclose(mesh.vertices, points[:3])


def test_from_meshio_prefers_volume_cells():
    """With both boundary triangles and volume tets present, the tets win."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    m = meshio.Mesh(
        points,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
        ],
    )

    mesh = _mesh_from_meshio(m)

    assert mesh.elements.shape[1] == 4
    assert mesh.spatial_dim == 3


def test_from_meshio_higher_order_uses_corners():
    """A P2 (triangle6) cell imports at its three corner nodes; midside nodes,
    now unreferenced, are pruned."""
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    )
    m = meshio.Mesh(points, [("triangle6", np.array([[0, 1, 2, 3, 4, 5]]))])

    mesh = _mesh_from_meshio(m)

    assert mesh.elements.shape == (1, 3)
    assert len(mesh.vertices) == 3


def test_from_meshio_rejects_unsupported_cell_type():
    """A non-simplex cell (a quad) has no linear-simplex mapping."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    m = meshio.Mesh(points, [("quad", np.array([[0, 1, 2, 3]]))])

    with pytest.raises(NotImplementedError):
        _mesh_from_meshio(m)


# --- physical-group tags -----------------------------------------------------

def _tagged_square_meshio():
    """A unit square (two triangles) with physical groups: the whole area "domain"
    (id 1), the left edge "inlet" (id 3), the right edge "outlet" (id 4)."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    lines = np.array([[3, 0], [1, 2]])  # left edge (x=0), right edge (x=1)
    return meshio.Mesh(
        points,
        [("triangle", triangles), ("line", lines)],
        cell_data={"gmsh:physical": [np.array([1, 1]), np.array([3, 4])]},
        field_data={
            "domain": np.array([1, 2]),
            "inlet": np.array([3, 1]),
            "outlet": np.array([4, 1]),
        },
    )


def test_from_meshio_reads_cell_and_facet_tags():
    """Physical groups import as cell/facet tags, with facet tags matched onto the
    re-derived boundary by vertex set and untagged facets left at 0."""
    mesh = _mesh_from_meshio(_tagged_square_meshio())

    assert np.array_equal(mesh.cell_tags, [1, 1])
    assert mesh.tag_names == {1: "domain", 3: "inlet", 4: "outlet"}
    # The left and right edges carry their tags; the top and bottom edges get 0.
    inlet = mesh.vertices[mesh.boundary[mesh.facets_with_tag("inlet")]]
    outlet = mesh.vertices[mesh.boundary[mesh.facets_with_tag("outlet")]]
    assert np.allclose(inlet[..., 0], 0.0)
    assert np.allclose(outlet[..., 0], 1.0)
    assert (mesh.facet_tags == 0).sum() == 2


def test_msh_round_trip_preserves_tags(tmp_path):
    """A Gmsh .msh save/load keeps the cell tags, facet tags, and their names."""
    mesh = _mesh_from_meshio(_tagged_square_meshio())
    path = tmp_path / "tagged.msh"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert np.array_equal(loaded.cell_tags, mesh.cell_tags)
    assert np.array_equal(loaded.facet_tags, mesh.facet_tags)
    assert loaded.tag_names == mesh.tag_names


def test_untagged_mesh_writes_only_volume_cells(make_unit_square, tmp_path):
    """A mesh with no tags round-trips through .msh with tags still absent."""
    mesh = make_unit_square(5)
    path = tmp_path / "plain.msh"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert loaded.cell_tags is None
    assert loaded.facet_tags is None
    assert loaded.tag_names == {}


def test_elastic_solution_round_trip_preserves_fields_mesh_and_dim(make_unit_square, tmp_path):
    """An ElasticSolution comes back the same class, with identical fields."""
    mesh = make_unit_square(6)
    n_el = len(mesh.elements)
    rng = np.random.default_rng(0)
    # Symmetric tensor fields, the shape a solve actually produces -- a 3-D array
    # per field, so this also pins that npz round-trips rank-3 values.
    stress = rng.random((n_el, 3, 3))
    stress = stress + np.swapaxes(stress, -2, -1)
    strain = 0.1 * stress
    solution = ElasticSolution(
        mesh, 2,
        np.arange(len(mesh.vertices) * 2, dtype=float),
        strain,
        stress,
        np.linspace(2, 3, n_el),
    )
    path = tmp_path / "solution.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert type(loaded) is ElasticSolution
    assert loaded.n_components == 2
    assert np.allclose(loaded.u, solution.u)
    assert np.allclose(loaded.compliance, solution.compliance)
    assert np.allclose(loaded.stress, solution.stress)
    assert loaded.stress.shape == (n_el, 3, 3)
    # Derived scalars are properties, so they are recomputed rather than stored --
    # and must agree with what the original reported.
    assert np.allclose(loaded.von_mises, solution.von_mises)
    assert type(loaded.mesh) is Mesh
    assert np.allclose(loaded.mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded.mesh.elements, mesh.elements)


def test_loading_a_pre_tensor_elastic_solution_fails_loudly(make_unit_square):
    """Solutions saved when stress was a scalar per element cannot be read as
    tensors. The archive has no format version, so the shape is the only signal --
    and a silently-wrong axis is worse than a refusal."""
    mesh = make_unit_square(4)
    n_el = len(mesh.elements)
    with pytest.raises(ValueError, match='tensor field'):
        ElasticSolution(
            mesh, 2,
            np.zeros(len(mesh.vertices) * 2),
            np.zeros(n_el),   # the old scalar shape
            np.zeros(n_el),
            np.zeros(n_el),
        )


def test_buckling_solution_round_trip(make_unit_square, tmp_path):
    """Load factors and mode vectors (a 2-D array) survive the npz round-trip, and
    mode_mesh still deforms the geometry afterwards."""
    mesh = make_unit_square(6)
    n_dofs = len(mesh.vertices) * 2
    rng = np.random.default_rng(1)
    solution = BucklingSolution(
        mesh, 2,
        np.array([1.5, 4.0, 9.0]),          # load factors
        rng.random((3, n_dofs)),            # mode shapes
    )
    path = tmp_path / "buckling.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert type(loaded) is BucklingSolution
    assert np.allclose(loaded.load_factors, solution.load_factors)
    assert np.allclose(loaded.modes, solution.modes)
    assert loaded.critical_load_factor == pytest.approx(1.5)
    assert loaded.mode_mesh(0).vertices.shape == mesh.vertices.shape


def test_transient_solution_round_trip_after_solve(make_unit_square, tmp_path):
    """A time series (times + a list of per-step fields) reloads as a TransientSolution."""
    mesh = make_unit_square(8)
    u0 = bump_function(mesh.vertices, mesh.vertices.max(axis=0), mag=50, size=0.3) + 300
    solution = ThetaMethod(dt=0.01, steps=3).run(heat(mesh), u0.copy())
    path = tmp_path / "heat.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert type(loaded) is TransientSolution
    assert np.allclose(loaded.t, solution.t)
    assert np.allclose(loaded.u, solution.u)
    # Geometry round-trips; a solve rebuilds element data into its own space.
    assert np.allclose(loaded.mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded.mesh.elements, mesh.elements)


def test_solution_load_does_not_unpickle(make_unit_square, tmp_path):
    """The archive must be readable with allow_pickle=False -- that is the whole
    point of moving off pickle, so pin it rather than trusting the default."""
    mesh = make_unit_square(6)
    solution = FieldSolution(mesh, 1, np.zeros(len(mesh.vertices)))
    path = tmp_path / "solution.npz"
    solution.save(path)

    with np.load(path, allow_pickle=False) as data:
        assert "value.u" in data.files


def test_saving_a_ragged_field_fails_loudly(make_unit_square, tmp_path):
    """A ragged field can only be stored as an object array (i.e. pickle), so it
    must raise at save time rather than silently degrade. Typed solves never
    produce one; a hand-built series with unequal-length steps does."""
    mesh = make_unit_square(6)
    ragged = TransientSolution(mesh, 1, np.array([0.0, 1.0]), [np.zeros(3), np.zeros(5)])

    with pytest.raises(ValueError):
        save_solution(ragged, tmp_path / "ragged.npz")
