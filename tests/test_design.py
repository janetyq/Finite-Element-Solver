"""The general design optimizer over the adjoint core.

`DesignOptimizer` with a compliance objective, configured to match `TopologyOptimizer`
(same filter, move limit, volume target), produces the same density trajectory.
"""
import numpy as np

from fem.boundary import BCType, BoundaryConditions
from fem.design import DesignOptimizer, SIMPModel, optimality_criteria_update
from fem.equations import LinearElastic
from fem.regions import on_plane
from fem.sensitivity import Compliance
from fem.space import FunctionSpace
from fem.topology import TopologyOptimizer, calculate_smoothing_matrix


def _cantilever_bc():
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return bc


def test_design_optimizer_matches_topology_optimizer_on_compliance(make_unit_square):
    mesh = make_unit_square(8)
    bc = _cantilever_bc()
    volume_frac, penalty, radius, iters = 0.5, 3.0, 0.1, 5

    topo = TopologyOptimizer(
        mesh, LinearElastic(E=1.0, nu=0.3), bc,
        iters=iters, volume_frac=volume_frac, penalty=penalty, smoothing_radius=radius,
    )
    topo_history = topo.solve()

    space = FunctionSpace(mesh, n_components=2)
    model = SIMPModel(
        space, base_E=1.0, nu=0.3, bc=bc, penalty=penalty,
        sensitivity_filter=calculate_smoothing_matrix(mesh, radius),
    )
    # Match TopologyOptimizer's OC move limit (0.1) so the two trajectories coincide.
    design = DesignOptimizer(model, Compliance(), volume_frac=volume_frac, iters=iters, move=0.1)
    design_history = design.solve()

    for topo_rho, design_rho in zip(topo_history.rho, design_history.rho):
        np.testing.assert_allclose(design_rho, topo_rho, rtol=1e-10, atol=1e-12)


def test_design_optimizer_reduces_compliance(make_unit_square):
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=2)
    model = SIMPModel(
        space, base_E=1.0, nu=0.3, bc=_cantilever_bc(), penalty=3.0,
        sensitivity_filter=calculate_smoothing_matrix(mesh, 0.12),
    )
    history = DesignOptimizer(model, Compliance(), volume_frac=0.5, iters=12).solve()

    assert history.objective[-1] < history.objective[0]


def test_design_optimizer_meets_the_volume_target(make_unit_square):
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=2)
    model = SIMPModel(space, base_E=1.0, nu=0.3, bc=_cantilever_bc(), penalty=3.0)
    design = DesignOptimizer(model, Compliance(), volume_frac=0.4, iters=15)
    design.solve()

    volumes = space.element_volumes
    achieved = float((volumes * design.rho).sum() / volumes.sum())
    assert abs(achieved - 0.4) < 1e-2


def test_optimality_criteria_update_rejects_a_signed_sensitivity():
    """OC needs a nonnegative (compliance-type) sensitivity; a signed one, such as a raw
    point-displacement objective would give, must fail loudly rather than take a NaN
    step. The point-value adjoint gradient itself is validated in test_sensitivity.py."""
    import pytest

    rho = np.full(10, 0.5)
    volumes = np.ones(10)
    sensitivity = np.linspace(-1.0, 1.0, 10)

    with pytest.raises(ValueError, match='nonnegative'):
        optimality_criteria_update(rho, sensitivity, volumes, volume_frac=0.4)


def test_optimality_criteria_update_hits_the_volume_fraction():
    rho = np.full(20, 0.5)
    sensitivity = np.linspace(1.0, 2.0, 20)
    volumes = np.ones(20)

    updated = optimality_criteria_update(rho, sensitivity, volumes, volume_frac=0.4)

    assert abs(updated.mean() - 0.4) < 1e-3
    assert np.all(updated >= 1e-6) and np.all(updated <= 1.0)
