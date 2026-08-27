"""Round-trip tests for fem.io: a save/load cycle preserves value arrays, mesh geometry,
mesh class, and n_components, and the load path never falls back to pickle.
"""
import numpy as np
import pytest

from fem.integrators import ThetaMethod
from fem.io import load_mesh, save_mesh, save_solution
from fem.mesh.mesh import Mesh
from fem.space import FunctionSpace
from fem.numerics import bump_function
from fem.equations import Poisson
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
        FunctionSpace(mesh, n_components=2),
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
    tensors. The archive has no format version, so the shape is the only signal."""
    mesh = make_unit_square(4)
    n_el = len(mesh.elements)
    with pytest.raises(ValueError, match='tensor field'):
        ElasticSolution(
            FunctionSpace(mesh, n_components=2),
            np.zeros(len(mesh.vertices) * 2),
            np.zeros(n_el),   # a scalar per element
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
        FunctionSpace(mesh, n_components=2),
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
    heat = Poisson().problem(mesh)
    solution = ThetaMethod(dt=0.01, steps=3).solve(heat, u0.copy())
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
    """The archive must be readable with allow_pickle=False."""
    mesh = make_unit_square(6)
    solution = FieldSolution(FunctionSpace(mesh), np.zeros(len(mesh.vertices)))
    path = tmp_path / "solution.npz"
    solution.save(path)

    with np.load(path, allow_pickle=False) as data:
        assert "value.u" in data.files


def test_saving_a_ragged_field_fails_loudly(make_unit_square, tmp_path):
    """A ragged field can only be stored as an object array (i.e. pickle), so it
    must raise at save time."""
    mesh = make_unit_square(6)
    ragged = TransientSolution(FunctionSpace(mesh), np.array([0.0, 1.0]), [np.zeros(3), np.zeros(5)])

    with pytest.raises(ValueError):
        save_solution(ragged, tmp_path / "ragged.npz")
