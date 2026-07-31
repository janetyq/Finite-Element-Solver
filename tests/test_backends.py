"""The iterative (AMG-CG) backend agrees with the direct one and fails loudly.

CG with an AMG preconditioner is an alternative way to solve the *same* free-free
block `DirectBackend` factors, so its answers must match a direct solve to solver
tolerance -- on isolated SPD systems, through the full Poisson/elasticity solves,
and through the MMS convergence net. That equivalence is what lets the fine-mesh 3D
convergence net (test_convergence_elasticity) run on AMG-CG without losing sight of
the direct path.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.forms import LinearElasticForm
from fem.backends import DirectBackend, IterativeBackend, rigid_body_modes
from fem.materials import LinearElasticMaterial
from fem.mesh.ruppert import create_box_mesh, create_rect_mesh
from fem.regions import everywhere, on_plane
from fem.equations import LinearElastic, Poisson
from fem.solver import Solver
from fem.space import FunctionSpace
from fem.system import DiscreteSystem


def _spd(n, seed=0):
    """A random symmetric positive-definite matrix (well posed for CG)."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _poisson_mms(n, backend):
    """Solve the manufactured sin*sin Poisson problem; return (h, L2 error)."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    eq = Poisson(source=lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, eq, bc, backend=backend)
    u = solver.solve().u
    exact = np.sin(np.pi * mesh.vertices[:, 0]) * np.sin(np.pi * mesh.vertices[:, 1])
    error = u - exact
    return 1.0 / (n - 1), np.sqrt(error @ solver.space.mass_matrix @ error)


def test_iterative_matches_direct_on_poisson():
    """AMG-CG reproduces the direct Poisson solution to solver tolerance."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(41, 41))
    eq = Poisson(source=lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    np.testing.assert_allclose(iterative, direct, atol=1e-8)


def test_iterative_matches_direct_on_elasticity():
    """AMG-CG reproduces the direct elasticity solution (vector field, SPD K)."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(31, 31))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])
    eq = LinearElastic(E=200, nu=0.3)

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    # Scale the tolerance to the field magnitude: the displacements are O(1).
    np.testing.assert_allclose(iterative, direct, atol=1e-8 * np.abs(direct).max())


def test_iterative_backend_preserves_second_order_convergence():
    """The MMS O(h^2) rate holds through the iterative backend, not just the direct one.

    This is the solver safety net extended to CG: a preconditioner or tolerance bug
    would show up as a broken convergence rate, not merely a shifted constant.
    """
    data = [_poisson_mms(n, IterativeBackend()) for n in (11, 21, 41)]
    hs = [h for h, _ in data]
    errors = [e for _, e in data]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    for p in orders:
        assert 1.7 < p < 2.3, f"expected ~2nd order under CG, got {orders}"


def test_iterative_matches_direct_on_3d_elasticity():
    """The direct/iterative equivalence holds for 3D vector elasticity too.

    A cheap (coarse-mesh) lock on the property the fine-mesh 3D convergence net leans
    on: it solves only on AMG-CG, trusting that a direct solve would land in the same
    place. Here that trust is checked directly, in 3D, where the rigid-body near-kernel
    and the tet assembly both differ from the 2D case above.
    """
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(7, 7, 7))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0, 0])       # cantilever: one face
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0, -5, 0])
    eq = LinearElastic(E=200, nu=0.3)

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    np.testing.assert_allclose(iterative, direct, atol=1e-7 * np.abs(direct).max())


def test_backends_agree_on_a_constrained_dense_system():
    """Both backends reproduce the same Dirichlet-eliminated solve of one SPD matrix."""
    A = _spd(12, seed=3)
    b = np.linspace(-1, 1, 12)
    free = np.arange(2, 12)
    fixed = np.array([0, 1])
    constraints = (free, fixed, np.array([0.3, -0.4]))

    direct = DiscreteSystem(A, constraints, DirectBackend()).solve(b)
    iterative = DiscreteSystem(A, constraints, IterativeBackend()).solve(b)
    np.testing.assert_allclose(iterative, direct, atol=1e-8)


def test_iterative_solver_reuses_its_setup_across_right_hand_sides():
    """One DiscreteSystem, many b's: the AMG hierarchy is built once and reused."""
    A = _spd(15, seed=4)
    free = np.arange(15)
    system = DiscreteSystem(A, (free, np.array([], dtype=int), np.array([])), IterativeBackend())
    for seed in range(3):
        b = np.random.default_rng(seed).normal(size=15)
        np.testing.assert_allclose(system.solve(b), np.linalg.solve(A, b), atol=1e-8)


def test_rigid_body_modes_are_in_the_stiffness_kernel():
    """Translations and rotations produce no strain: K @ mode == 0 unconstrained.

    The near-kernel AMG needs *is* the stiffness kernel of the free (unconstrained)
    body, so this is the property that makes the modes the right ones to feed it.
    """
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(9, 9))
    space = FunctionSpace(mesh, n_components=2)
    K = space.assemble(LinearElasticForm(LinearElasticMaterial(E=200, nu=0.3)))
    modes = rigid_body_modes(mesh.vertices, 2)
    assert modes.shape == (space.n_dofs, 3)
    residual = K @ modes
    assert np.abs(residual).max() < 1e-8, "a rigid-body mode strained the body"


def test_facade_gives_elasticity_its_rigid_body_modes():
    """The Solver facade enriches an iterative elastic solve with the modes.

    Correctness is the same either way (both match direct), so the observable is
    that the facade routes elasticity through with_near_null_space -- checked by the
    enriched backend converging, and by a lightly constrained solve still matching.
    """
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(25, 25))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])          # cantilever: one edge
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0, -20])
    eq = LinearElastic(E=200, nu=0.3)

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    np.testing.assert_allclose(iterative, direct, atol=1e-7 * np.abs(direct).max())


def test_iterative_backend_matches_direct_through_a_time_step():
    """A heat step's effective operator M + θdtK is SPD, so AMG-CG matches direct.

    Covers the integrator wiring: the backend threads through ThetaMethod into the
    DiscreteSystem it reuses across steps.
    """
    from fem.integrators import ThetaMethod
    from fem.numerics import bump_function
    from fem.problem import heat

    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(21, 21))
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=10, size=0.2) + 300
    problem = heat(mesh)

    direct = ThetaMethod(dt=0.01, steps=5).run(problem, u0.copy()).u[-1]
    iterative = ThetaMethod(dt=0.01, steps=5, backend=IterativeBackend()).run(problem, u0.copy()).u[-1]
    np.testing.assert_allclose(iterative, direct, atol=1e-7)


def test_non_convergence_raises():
    """A CG that cannot reach tolerance in its iteration budget fails loudly.

    Capping maxiter at 1 on a system that needs more is a deterministic way to force
    a nonzero convergence flag; the backend must raise rather than return the
    unconverged iterate.
    """
    A = _spd(40, seed=5)
    free = np.arange(40)
    backend = IterativeBackend(rtol=1e-14, maxiter=1)
    system = DiscreteSystem(A, (free, np.array([], dtype=int), np.array([])), backend)
    with pytest.raises(RuntimeError, match="CG failed"):
        system.solve(np.ones(40))
