"""The adaptive refinement loop's machinery, driven with a hand-written estimator: it
re-solves on each new mesh rather than reusing a stale estimate, and the conditions it
cannot carry across a remesh fail loudly.
"""
import numpy as np
import pytest

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.physics.energies import StVenantKirchhoff
from fem.analysis.estimators import RecoveryEstimator
from fem.physics.forms import EnergyForm
from fem.problem import Problem
from fem.regions import everywhere, at_indices, on_plane
from fem.physics.equations import LinearElastic, Projection, Poisson, FiniteStrainElastic
from fem.loads import Source
from fem.space import FunctionSpace


def refine_near_centre(problem, solution):
    """Estimator stand-in: 'error' is largest at the centre of the domain."""
    mesh = problem.space.mesh
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    return 1.0 / (0.05 + np.linalg.norm(centroids - 0.5, axis=1))


def _for(equation, bc=None):
    return lambda mesh: equation.problem(mesh, bc)


def test_adaptive_refinement_grows_mesh_and_resolves(make_unit_square):
    """The loop must refine repeatedly and leave a solution on the final mesh."""
    mesh = make_unit_square(6)
    driver = AdaptiveRefinement(mesh, _for(Projection(), Conditions(Source(1.0))), refine_near_centre,
                                max_triangles=400, max_iters=3)

    n_before = len(mesh.elements)
    solution = driver.run()

    assert len(driver.mesh.elements) > n_before, "mesh never grew"
    # The solution must belong to the *final* mesh, not the one we started on.
    u = solution.u
    assert solution.mesh is driver.mesh
    assert len(u) == len(driver.mesh.vertices)
    assert np.all(np.isfinite(u))


def test_adaptive_refinement_respects_max_triangles(make_unit_square):
    """The element cap stops the loop."""
    mesh = make_unit_square(6)
    cap = len(mesh.elements) + 1
    driver = AdaptiveRefinement(mesh, _for(Projection(), Conditions(Source(1.0))), refine_near_centre,
                                max_triangles=cap, max_iters=50)
    driver.run()
    # One round may overshoot the cap; the point is that it stops, not that it
    # lands exactly on it.
    assert len(driver.mesh.elements) < 400


def test_adaptive_refinement_respects_max_iters(make_unit_square):
    mesh = make_unit_square(6)
    problem_for = _for(Projection(), Conditions(Source(1.0)))

    first = AdaptiveRefinement(mesh, problem_for, refine_near_centre,
                               max_triangles=10**6, max_iters=1)
    first.run()
    after_one = len(first.mesh.elements)

    second = AdaptiveRefinement(first.mesh, problem_for, refine_near_centre,
                                max_triangles=10**6, max_iters=1)
    second.run()
    assert len(second.mesh.elements) > after_one, "max_iters=1 did no work"


def test_adaptive_refinement_carries_geometric_dirichlet_bcs(make_unit_square):
    """A Dirichlet condition described as a region is re-resolved on each refined mesh,
    so it keeps holding on nodes that did not exist when it was written."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(everywhere(), 0.0))
    driver = AdaptiveRefinement(mesh, _for(Poisson(), bc + Source(1.0)), refine_near_centre,
                                max_triangles=400, max_iters=3)

    n_before = len(mesh.vertices)
    solution = driver.run()

    final = driver.mesh
    assert len(final.vertices) > n_before, "mesh never grew"
    u = solution.u
    # Every boundary node of the *refined* mesh is pinned, including the new ones.
    assert np.allclose(u[final.boundary_idxs], 0.0, atol=1e-12)
    assert np.abs(u).max() > 0, "solution is trivially zero"


def test_adaptive_refinement_rejects_index_based_bcs(make_unit_square):
    """at_indices names nodes of one specific mesh, so the loop must refuse it
    rather than quietly relocate the condition after renumbering."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(at_indices(mesh.boundary_idxs), 0.0))

    with pytest.raises(NotImplementedError):
        AdaptiveRefinement(mesh, _for(Projection(), bc + Source(1.0)), refine_near_centre).run()


def test_adaptive_refinement_rejects_mismatched_estimator(make_unit_square):
    """An estimator sized to the wrong mesh is refused instead of indexing unrelated elements."""
    mesh = make_unit_square(6)

    def too_long(problem, solution):
        return np.ones(len(problem.space.mesh.elements) + 1)

    with pytest.raises(ValueError):
        AdaptiveRefinement(mesh, _for(Projection(), Conditions(Source(1.0))), too_long).run()


def test_bc_spec_is_reusable_across_meshes(make_unit_square):
    """The spec holds no mesh, so the same object resolves on any of them."""
    bc = Conditions(Dirichlet(everywhere(), 0.0))

    coarse, fine = make_unit_square(4), make_unit_square(9)
    assert len(bc.resolve(FunctionSpace(coarse, n_components=1)).fixed_idxs) == len(coarse.boundary_idxs)
    assert len(bc.resolve(FunctionSpace(fine, n_components=1)).fixed_idxs) == len(fine.boundary_idxs)


def test_adaptive_refinement_drives_a_finite_strain_problem(make_unit_square):
    """The driver takes any problem builder and picks the strategy its problem needs,
    so a Green-Lagrange equation is refined the same way as a linear one."""
    mesh = make_unit_square(5)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
        Dirichlet(on_plane(0, 1.0), [0.02, 0.0]),
    )
    equation = FiniteStrainElastic(E=200, nu=0.3)

    driver = AdaptiveRefinement(
        mesh, _for(equation, bc), refine_near_centre, max_triangles=200, max_iters=2,
    )
    n_before = len(mesh.elements)
    solution = driver.run()

    assert len(driver.mesh.elements) > n_before, "mesh never grew"
    assert solution.mesh is driver.mesh
    assert len(solution.u) == len(driver.mesh.vertices) * 2
    assert np.all(np.isfinite(solution.u))


def test_recovery_estimator_reads_the_flux_off_an_energy_problem(make_unit_square):
    """An EnergyForm names its stress as the recoverable flux, so the recovery estimator
    needs no physics argument to estimate a nonlinear solve."""
    mesh = make_unit_square(5)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
        Dirichlet(on_plane(0, 1.0), [0.02, 0.0]),
    )
    space = LinearElastic(E=200, nu=0.3).space(mesh)
    problem = Problem(space, EnergyForm(StVenantKirchhoff(200, 0.3)), bc)
    solution = problem.solve()

    eta = RecoveryEstimator().estimate(problem, solution)
    assert eta.shape == (len(mesh.elements),)
    assert np.all(np.isfinite(eta)) and np.all(eta >= 0)
