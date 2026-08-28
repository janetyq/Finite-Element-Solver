"""The iterative backends agree with the direct one and fail loudly.

AMG-CG and MINRES solve the same free-free block `DirectBackend` factors, so their
answers must match a direct solve to solver tolerance: on isolated systems, through the
full Poisson/elasticity solves, and through the MMS convergence net.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.forms import LinearElasticForm
from fem.backends import DirectBackend, IterativeBackend, MinresBackend, rigid_body_modes
from fem.materials import LinearElasticMaterial
from fem.mesh.structured import create_box_mesh, create_rect_mesh
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


def _symmetric_indefinite(n, seed=0):
    """A symmetric, nonsingular, indefinite matrix with a known spectrum."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(-2.0, 3.0, n)
    eigenvalues[np.abs(eigenvalues) < 0.4] = 0.7  # keep clear of zero
    return (Q * eigenvalues) @ Q.T


def _poisson_mms(n, backend):
    """Solve the manufactured sin*sin Poisson problem; return (h, L2 error)."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    eq = Poisson(source=lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])])
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    solver = Solver(mesh, eq, bc, backend=backend)
    u = solver.solve().u
    exact = np.sin(np.pi * mesh.vertices[:, 0]) * np.sin(np.pi * mesh.vertices[:, 1])
    error = u - exact
    return 1.0 / (n - 1), np.sqrt(error @ solver.space.mass_matrix @ error)


def test_iterative_matches_direct_on_poisson():
    """AMG-CG reproduces the direct Poisson solution to solver tolerance."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(41, 41))
    eq = Poisson(source=lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])])
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    np.testing.assert_allclose(iterative, direct, atol=1e-8)


def test_iterative_matches_direct_on_elasticity():
    """AMG-CG reproduces the direct elasticity solution (vector field, SPD K)."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(31, 31))
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )
    eq = LinearElastic(E=200, nu=0.3)

    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    # Scale the tolerance to the field magnitude: the displacements are O(1).
    np.testing.assert_allclose(iterative, direct, atol=1e-8 * np.abs(direct).max())


def test_iterative_backend_preserves_second_order_convergence():
    """The MMS O(h^2) rate holds through the iterative backend, not just the direct one."""
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
    """The direct/iterative equivalence holds for 3D vector elasticity, where the rigid-body
    near-kernel and the tet assembly both differ from 2D."""
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(7, 7, 7))
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0, 0]),
        Neumann(on_plane(0, 1.0), [0, -5, 0]),
    )
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
    """Translations and rotations produce no strain: K @ mode == 0 unconstrained."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(9, 9))
    space = FunctionSpace(mesh, n_components=2)
    K = space.assemble(LinearElasticForm(LinearElasticMaterial(E=200, nu=0.3)))
    modes = rigid_body_modes(mesh.vertices, 2)
    assert modes.shape == (space.n_dofs, 3)
    residual = K @ modes
    assert np.abs(residual).max() < 1e-8, "a rigid-body mode strained the body"


def _cantilever():
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(25, 25))
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [0, -20]),
    )
    return mesh, bc


def test_linear_solve_gives_elasticity_its_rigid_body_modes():
    """An elastic problem composed by hand hands its rigid-body modes, restricted to the
    free DOFs, to an iterative backend: the same near-kernel the facade path gets."""
    from fem.solve import backend_for

    mesh, bc = _cantilever()
    elastic = LinearElastic(E=200, nu=0.3)
    problem = elastic.problem(mesh, bc)
    backend = backend_for(problem, IterativeBackend())
    assert isinstance(backend, IterativeBackend)
    free = problem.constraints[0]
    assert backend.near_null_space is not None
    assert backend.near_null_space.shape == (len(free), 3)
    np.testing.assert_array_equal(backend.near_null_space, problem.near_null_space()[free])

    # A near-kernel the caller set is kept; a scalar problem and a direct backend get none.
    preset = IterativeBackend(near_null_space=np.ones((len(free), 1)))
    assert backend_for(problem, preset) is preset
    scalar_bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    scalar = backend_for(Poisson(1.0).problem(mesh, scalar_bc), IterativeBackend())
    assert isinstance(scalar, IterativeBackend) and scalar.near_null_space is None
    direct = DirectBackend()
    assert backend_for(problem, direct) is direct


def test_iterative_elastic_solve_matches_direct_through_facade_and_composition():
    from fem.solve import LinearSolve

    mesh, bc = _cantilever()
    eq = LinearElastic(E=200, nu=0.3)
    direct = Solver(mesh, eq, bc, backend=DirectBackend()).solve().u
    tol = 1e-7 * np.abs(direct).max()

    iterative = Solver(mesh, eq, bc, backend=IterativeBackend()).solve().u
    np.testing.assert_allclose(iterative, direct, atol=tol)
    problem = eq.problem(mesh, bc)
    composed = LinearSolve(IterativeBackend()).solve(problem)
    np.testing.assert_allclose(composed, direct, atol=tol)


def test_iterative_backend_matches_direct_through_a_time_step():
    """A heat step's effective operator M + θdtK is SPD, so AMG-CG matches direct."""
    from fem.integrators import ThetaMethod
    from fem.numerics import bump_function

    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(21, 21))
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=10, size=0.2) + 300
    problem = Poisson().problem(mesh)

    direct = ThetaMethod(dt=0.01, steps=5).solve(problem, u0.copy()).u[-1]
    iterative = ThetaMethod(dt=0.01, steps=5, backend=IterativeBackend()).solve(problem, u0.copy()).u[-1]
    np.testing.assert_allclose(iterative, direct, atol=1e-7)


def test_non_convergence_raises():
    """A CG that cannot reach tolerance in its iteration budget raises rather than returning
    the unconverged iterate."""
    A = _spd(40, seed=5)
    free = np.arange(40)
    backend = IterativeBackend(rtol=1e-14, maxiter=1)
    system = DiscreteSystem(A, (free, np.array([], dtype=int), np.array([])), backend)
    with pytest.raises(RuntimeError, match="CG failed"):
        system.solve(np.ones(40))


# -- MINRES: the iterative path for symmetric indefinite systems ---------------


def test_minres_matches_direct_on_an_indefinite_system():
    """MINRES solves a symmetric indefinite block that CG cannot, matching a direct solve."""
    A = _symmetric_indefinite(30, seed=1)
    b = np.linspace(-1, 1, 30)
    free = np.arange(30)
    constraints = (free, np.array([], dtype=int), np.array([]))

    minres = DiscreteSystem(A, constraints, MinresBackend()).solve(b)
    direct = DiscreteSystem(A, constraints, DirectBackend()).solve(b)
    np.testing.assert_allclose(minres, direct, atol=1e-8)


def test_minres_matches_direct_through_dirichlet_elimination():
    """MINRES solves the free-free block of a constrained indefinite system, matching direct."""
    A = _symmetric_indefinite(24, seed=2)
    b = np.ones(24)
    free = np.arange(2, 24)
    fixed = np.array([0, 1])
    constraints = (free, fixed, np.array([0.2, -0.3]))

    minres = DiscreteSystem(A, constraints, MinresBackend()).solve(b)
    direct = DiscreteSystem(A, constraints, DirectBackend()).solve(b)
    np.testing.assert_allclose(minres, direct, atol=1e-8)


def test_minres_matches_direct_on_an_spd_system():
    """MINRES also solves SPD systems (a superset of CG's domain), matching direct."""
    A = _spd(20, seed=6)
    b = np.linspace(2, -2, 20)
    free = np.arange(20)
    constraints = (free, np.array([], dtype=int), np.array([]))

    minres = DiscreteSystem(A, constraints, MinresBackend()).solve(b)
    np.testing.assert_allclose(minres, np.linalg.solve(A, b), atol=1e-8)


def test_minres_non_convergence_raises():
    """MINRES that cannot reach tolerance in its iteration budget fails loudly, like CG."""
    A = _symmetric_indefinite(40, seed=7)
    free = np.arange(40)
    backend = MinresBackend(rtol=1e-14, maxiter=1)
    system = DiscreteSystem(A, (free, np.array([], dtype=int), np.array([])), backend)
    with pytest.raises(RuntimeError, match="MINRES failed"):
        system.solve(np.ones(40))
