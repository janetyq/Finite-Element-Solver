"""The assembled global operators, pinned entry-for-entry.

`test_forms.py` pins the *element* matrices and `test_convergence*.py` pins the
*solution*; neither pins the scatter that joins them. A scatter bug that
misplaces a contribution can leave the element matrices correct and still show up
only as a degraded convergence rate several layers away.

So this file characterizes the global operators directly. Small meshes get exact
golden matrices; the 3D meshes get invariants plus scalar fingerprints (nnz,
trace, Frobenius norm, row sums), which are compact enough to read and strong
enough that a misdirected entry moves at least one of them.

The invariants are the mathematically meaningful half and are worth stating:

- A mass matrix sums to the measure of its domain -- `sum_ij int phi_i phi_j` is
  `int 1` over the domain, since the P1 basis is a partition of unity. For a
  k-component space that is k times the measure, once per component.
- A Laplacian annihilates constants, so its rows sum to zero.
- An elastic stiffness annihilates rigid translations, for the same reason.
"""
import numpy as np
import pytest

from fem.forms import LaplacianForm, LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.generation import create_box_mesh, create_rect_mesh
from fem.space import FunctionSpace


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
    """Compare fingerprints relatively, with an absolute floor for the ~0 entries.

    The reductions span twelve orders of magnitude (a stiffness trace in the
    thousands next to a row sum that is exactly zero up to rounding), so neither
    tolerance alone fits: `rel` handles the large entries, `abs` the vanishing ones.
    """
    return pytest.approx(expected, rel=1e-10, abs=1e-8)


# --------------------------------------------------------------------------
# 2D, exact
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def unit_square():
    # The two-triangle unit square: vertices (0,0) (1,0) (0,1) (1,1), split
    # along the 0--3 diagonal. Small enough to write every operator out in full.
    return FunctionSpace(create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(2, 2)))


def test_laplacian_global_matrix(unit_square):
    expected = np.array([
        [1.0, -0.5, -0.5, 0.0],
        [-0.5, 1.0, 0.0, -0.5],
        [-0.5, 0.0, 1.0, -0.5],
        [0.0, -0.5, -0.5, 1.0],
    ])
    K = unit_square.assemble(LaplacianForm()).toarray()
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
    return create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(3, 3, 3))


def test_scalar_operators_on_cube(unit_cube):
    V = FunctionSpace(unit_cube)
    assert fingerprint(V.mass_matrix) == approx({
        'shape': (27, 27), 'nnz': 223, 'trace': 0.4,
        'fro': 0.10520317644135, 'sum': 1.0,
    })
    assert fingerprint(V.assemble(LaplacianForm())) == approx({
        'shape': (27, 27), 'nnz': 135, 'trace': 24.0,
        'fro': 6.1101009266078, 'sum': 0.0,
    })


def test_vector_operators_on_cube(unit_cube):
    V = FunctionSpace(unit_cube, n_components=3)
    assert fingerprint(V.mass_matrix) == approx({
        'shape': (81, 81), 'nnz': 669, 'trace': 1.2,
        'fro': 0.18221724671416, 'sum': 3.0,
    })
    assert fingerprint(V.boundary_mass_matrix) == approx({
        'shape': (81, 81), 'nnz': 510, 'trace': 9.0,
        'fro': 1.1180339887499, 'sum': 18.0,
    })
    K = V.assemble(LinearElasticForm(LinearElasticMaterial(200.0, 0.3)))
    assert fingerprint(K) == approx({
        'shape': (81, 81), 'nnz': 1659, 'trace': 10153.846153846,
        'fro': 1660.0583982503, 'sum': 0.0,
    })


# --------------------------------------------------------------------------
# Invariants -- these hold at any resolution, in any dimension
# --------------------------------------------------------------------------

@pytest.fixture(scope='module', params=['2d', '3d'])
def mesh(request):
    if request.param == '2d':
        return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    return create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(4, 4, 4))


@pytest.mark.parametrize('n_components', [1, 2])
def test_mass_matrix_sums_to_the_measure(mesh, n_components):
    V = FunctionSpace(mesh, n_components=n_components)
    assert V.mass_matrix.toarray().sum() == pytest.approx(n_components * 1.0)
    surface = 4.0 if mesh.spatial_dim == 2 else 6.0
    assert V.boundary_mass_matrix.toarray().sum() == pytest.approx(n_components * surface)


def test_laplacian_is_symmetric_and_annihilates_constants(mesh):
    K = FunctionSpace(mesh).assemble(LaplacianForm()).toarray()
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
    """TopologyOptimizer's per-element E must survive the scatter, not just the form."""
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
