"""The assembled global operators, pinned entry for entry.

`test_forms.py` pins the element matrices and `test_convergence*.py` the solution;
this pins the scatter that joins them. Small meshes get exact golden matrices; the 3D
meshes get invariants (a mass matrix sums to the domain measure, a Laplacian
annihilates constants, an elastic stiffness annihilates rigid translations) plus
scalar fingerprints (nnz, trace, Frobenius norm, row sums). Of the fingerprints, `nnz`
follows from the connectivity and `trace` and `sum` from the element formulas; `fro` is
a regression pin recorded from the implementation, there to catch drift.
"""
import numpy as np
import pytest
from scipy.sparse import csr_array

from fem.mesh.structured import box_mesh
from fem.physics.forms import DiffusionForm, LinearElasticForm
from fem.physics.materials import LinearElasticMaterial
from fem.space import FunctionSpace, dof_indices


def fingerprint(A):
    """Scalar reductions of a sparse operator, for meshes too big to write out."""
    A = A.toarray()
    return {
        'shape': A.shape,
        # Counted against a relative threshold rather than `!= 0`: an entry that
        # cancels to rounding is structurally a zero, and whether it lands at
        # exactly 0.0 or at 1e-16 depends on the arithmetic used to get there.
        # Pinning the exact-zero count would pin that incidental detail.
        'nnz': int((np.abs(A) > 1e-12 * np.abs(A).max()).sum()),
        'trace': float(np.trace(A)),
        'fro': float(np.linalg.norm(A)),
        'sum': float(A.sum()),
    }


def approx(expected):
    """Compare fingerprints relatively, with an absolute floor for the ~0 entries."""
    return pytest.approx(expected, rel=1e-10, abs=1e-8)


# --------------------------------------------------------------------------
# 2D, exact
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def unit_square():
    # The two-triangle unit square: vertices (0,0) (1,0) (0,1) (1,1), split
    # along the 0--3 diagonal. Small enough to write every operator out in full.
    return FunctionSpace(box_mesh(corners=[[0, 0], [1, 1]], resolution=(2, 2)))


def test_laplacian_global_matrix(unit_square):
    expected = np.array([
        [1.0, -0.5, -0.5, 0.0],
        [-0.5, 1.0, 0.0, -0.5],
        [-0.5, 0.0, 1.0, -0.5],
        [0.0, -0.5, -0.5, 1.0],
    ])
    K = unit_square.assemble(DiffusionForm()).toarray()
    # atol, not exact: the shared diagonal's off-diagonal entry is a cancellation
    # of two equal-and-opposite element contributions, so it lands at rounding.
    np.testing.assert_allclose(K, expected, atol=1e-12)


def test_mass_global_matrix(unit_square):
    expected = np.array([
        [1 / 6, 1 / 24, 1 / 24, 1 / 12],
        [1 / 24, 1 / 12, 0.0, 1 / 24],
        [1 / 24, 0.0, 1 / 12, 1 / 24],
        [1 / 12, 1 / 24, 1 / 24, 1 / 6],
    ])
    np.testing.assert_allclose(unit_square.mass_matrix.toarray(), expected, atol=1e-12)


def test_boundary_mass_global_matrix(unit_square):
    # Four unit edges; each contributes the 1D consistent mass [[1/3, 1/6], [1/6, 1/3]].
    expected = np.array([
        [2 / 3, 1 / 6, 1 / 6, 0.0],
        [1 / 6, 2 / 3, 0.0, 1 / 6],
        [1 / 6, 0.0, 2 / 3, 1 / 6],
        [0.0, 1 / 6, 1 / 6, 2 / 3],
    ])
    np.testing.assert_allclose(
        unit_square.boundary_mass_matrix.toarray(), expected, atol=1e-12
    )


# --------------------------------------------------------------------------
# 3D, fingerprinted
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def unit_cube():
    return box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(3, 3, 3))


def test_scalar_operators_on_cube(unit_cube):
    V = FunctionSpace(unit_cube)
    assert fingerprint(V.mass_matrix) == approx({
        'shape': (27, 27), 'nnz': 207, 'trace': 0.4,
        'fro': 0.12140668570278, 'sum': 1.0,
    })
    assert fingerprint(V.assemble(DiffusionForm())) == approx({
        'shape': (27, 27), 'nnz': 207, 'trace': 20.0,
        'fro': 4.987484335815, 'sum': 0.0,
    })


def test_vector_operators_on_cube(unit_cube):
    V = FunctionSpace(unit_cube, n_components=3)
    assert fingerprint(V.mass_matrix) == approx({
        'shape': (81, 81), 'nnz': 621, 'trace': 1.2,
        'fro': 0.21028254801576, 'sum': 3.0,
    })
    assert fingerprint(V.boundary_mass_matrix) == approx({
        'shape': (81, 81), 'nnz': 510, 'trace': 9.0,
        'fro': 1.1456439237390, 'sum': 18.0,
    })
    K = V.assemble(LinearElasticForm(LinearElasticMaterial(200.0, 0.3)))
    assert fingerprint(K) == approx({
        'shape': (81, 81), 'nnz': 1317, 'trace': 8461.538461538,
        'fro': 1347.5496426654, 'sum': 0.0,
    })


# --------------------------------------------------------------------------
# Invariants -- these hold at any resolution, in any dimension
# --------------------------------------------------------------------------

@pytest.fixture(scope='module', params=['2d', '3d'])
def mesh(request):
    if request.param == '2d':
        return box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    return box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(4, 4, 4))


@pytest.mark.parametrize('n_components', [1, 2])
def test_mass_matrix_sums_to_the_measure(mesh, n_components):
    V = FunctionSpace(mesh, n_components=n_components)
    assert V.mass_matrix.toarray().sum() == pytest.approx(n_components * 1.0)
    surface = 4.0 if mesh.spatial_dim == 2 else 6.0
    assert V.boundary_mass_matrix.toarray().sum() == pytest.approx(n_components * surface)


def test_laplacian_is_symmetric_and_annihilates_constants(mesh):
    K = FunctionSpace(mesh).assemble(DiffusionForm()).toarray()
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(K @ np.ones(K.shape[0]), 0, atol=1e-10)


def test_elastic_stiffness_annihilates_rigid_translations(mesh):
    d = mesh.spatial_dim
    V = FunctionSpace(mesh, n_components=d)
    K = V.assemble(LinearElasticForm(LinearElasticMaterial(200.0, 0.3))).toarray()
    np.testing.assert_allclose(K, K.T, atol=1e-9)
    for component in range(d):
        translation = np.zeros(V.n_dofs)
        translation[component::d] = 1.0
        np.testing.assert_allclose(K @ translation, 0, atol=1e-9)


def test_per_element_modulus_reaches_the_global_matrix(mesh):
    """A per-element E (SIMP's density-scaled modulus) must survive the scatter."""
    d = mesh.spatial_dim
    V = FunctionSpace(mesh, n_components=d)
    uniform = V.assemble(LinearElasticForm(LinearElasticMaterial(200.0, 0.3))).toarray()

    E = np.full(len(mesh.elements), 200.0)
    np.testing.assert_allclose(
        V.assemble(LinearElasticForm(LinearElasticMaterial(E, 0.3))).toarray(),
        uniform, atol=1e-9,
    )
    # Halving every element's modulus halves the assembled operator.
    np.testing.assert_allclose(
        V.assemble(LinearElasticForm(LinearElasticMaterial(0.5 * E, 0.3))).toarray(),
        0.5 * uniform, atol=1e-9,
    )


def test_scatter_matches_a_direct_coo_sum(mesh):
    """The cached scatter plan agrees entry for entry with a raw COO sum."""
    d = mesh.spatial_dim
    V = FunctionSpace(mesh, n_components=d)
    form = LinearElasticForm(LinearElasticMaterial(np.linspace(50.0, 200.0, len(mesh.elements)), 0.3))
    element_matrices = form.element_matrices(V.geometry)

    dofs = dof_indices(mesh.elements, d)
    k = dofs.shape[1]
    reference = csr_array(
        (element_matrices.ravel(),
         (np.repeat(dofs, k, axis=1).ravel(), np.tile(dofs, (1, k)).ravel())),
        shape=(V.n_dofs, V.n_dofs),
    )

    assembled = V.assemble(form)
    np.testing.assert_array_equal(assembled.indptr, reference.indptr)
    np.testing.assert_array_equal(assembled.indices, reference.indices)
    # `atol` covers the entries that cancel to zero: the scatter's bincount and the COO
    # sum add the same terms in a different order, so a structural zero lands at ~1e-14
    # in one and ~0 in the other, which no relative tolerance can bridge.
    np.testing.assert_allclose(assembled.data, reference.data, rtol=1e-12, atol=1e-10)


def test_scatter_plan_is_not_bound_to_the_first_form_assembled(mesh):
    """The plan caches destinations, which are connectivity; it must not cache
    values. Assembling a second operator over a space that has already assembled
    one has to give what a fresh space would."""
    d = mesh.spatial_dim
    V = FunctionSpace(mesh, n_components=d)
    elastic = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))

    V.assemble(elastic)                      # populates the cached plan
    reassembled = V.assemble(elastic).toarray()
    mass_after = V.mass_matrix.toarray()     # a different form over the same plan

    fresh = FunctionSpace(mesh, n_components=d)
    np.testing.assert_allclose(reassembled, fresh.assemble(elastic).toarray(), atol=1e-9)
    np.testing.assert_allclose(mass_after, fresh.mass_matrix.toarray(), atol=1e-12)
