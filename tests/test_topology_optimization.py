"""Tests for the density-based topology optimizer.

Distinct from `test_topology.py`, which is about mesh topology -- edges and
boundary facets -- despite the name collision.
"""
import numpy as np

from fem.boundary import BCType, BoundaryConditions
from fem.regions import on_plane
from fem.solver import LinearElastic
from fem.topology import TopologyOptimizer


def _optimizer(mesh, penalty):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return TopologyOptimizer(
        mesh, LinearElastic(E=1.0, nu=0.3), bc,
        iters=1, volume_frac=0.5, penalty=penalty,
    )


def test_simp_penalty_drives_the_modulus_scaling(make_unit_square):
    """E(rho) = rho^p E_0, with p the configured exponent rather than a
    literal 3 buried in set_rho."""
    optimizer = _optimizer(make_unit_square(5), penalty=2.0)
    rho = np.full(len(optimizer.mesh.elements), 0.5)

    optimizer.set_rho(rho)

    assert np.allclose(optimizer.equation.E, 0.5**2.0 * 1.0)


def test_compliance_gradient_uses_the_same_penalty_as_set_rho(make_unit_square):
    """The sensitivity p/rho * c is only the derivative of the compliance if p
    is the exponent set_rho actually raised rho to. The two were independent
    literal 3s, so changing the penalty would have silently left the optimizer
    descending the wrong gradient."""
    penalty = 2.0
    optimizer = _optimizer(make_unit_square(5), penalty=penalty)
    optimizer.solver.solve()

    compliance = optimizer.solver.solution.values['compliance']
    expected = compliance * penalty / optimizer.rho

    assert np.allclose(optimizer.compliance_gradient(None), expected)
