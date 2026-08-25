"""Quadratic (P2) elements on the nonlinear energy path.

The St-Venant-Kirchhoff energy is quartic in the displacement gradient, so on P2 it
needs a degree-4 rule where the linear stiffness's default is 2. The energy, residual,
and tangent share that rule, which a finite-difference gradient check pins.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.energies import SmallStrain, StVenantKirchhoff
from fem.equations import LinearElastic, StrainMeasure
from fem.forms import EnergyForm
from fem.mesh.structured import create_rect_mesh
from fem.regions import on_plane
from fem.energy_solver import EnergySolver
from fem.solution import ElasticSolution
from fem.space import FunctionSpace


def _stvk_form():
    return EnergyForm(StVenantKirchhoff(200.0, 0.3))


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_energy_assembly_is_its_own_derivative_chain(element_type):
    """The definitive consistency check: the assembled residual is the gradient of the
    assembled energy, and the tangent is the gradient of the residual, at an arbitrary
    deformed state. This is what makes Newton converge, and it holds only if the energy,
    residual, and tangent share one quadrature rule (degree 4 on P2 St-VK)."""
    mesh = create_rect_mesh([[0.0, 0.0], [2.0, 1.0]], [5, 4])
    space = FunctionSpace(mesh, element_type, n_components=2)
    form = _stvk_form()
    rng = np.random.default_rng(0)
    u = 0.05 * rng.standard_normal(space.n_dofs)   # a moderate, non-homogeneous state
    direction = rng.standard_normal(space.n_dofs)
    eps = 1e-6

    d_energy = (space.total_energy(form, u + eps * direction)
                - space.total_energy(form, u - eps * direction)) / (2 * eps)
    residual = space.assemble_residual(form, u)
    assert d_energy == pytest.approx(residual @ direction, rel=1e-5)

    d_residual = (space.assemble_residual(form, u + eps * direction)
                  - space.assemble_residual(form, u - eps * direction)) / (2 * eps)
    tangent = space.assemble_tangent(form, u)
    assert np.linalg.norm(d_residual - tangent @ direction) < 1e-5 * np.linalg.norm(d_residual)


def test_the_energy_rule_is_degree_aware():
    """The rule follows the energy's nonlinearity, not just the element: quartic St-VK
    jumps to degree 4 on P2, quadratic small strain stays at the default 2, and P1 (a
    constant integrand) needs nothing beyond its default."""
    stvk, small = _stvk_form(), EnergyForm(SmallStrain(200.0, 0.3))
    assert stvk.quadrature_degree(2) == 4        # P2 quartic energy
    assert small.quadrature_degree(2) == 2       # P2 quadratic energy
    assert stvk.quadrature_degree(1) == 0        # P1: element-constant, default rule stands


def test_full_integration_changes_the_p2_stvk_energy():
    """Integrating the quartic St-VK energy on P2 at the default degree-2 rule
    under-integrates it by over 10 percent."""
    mesh = create_rect_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 4])
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    form = _stvk_form()
    u_elements = (0.1 * np.random.default_rng(1).standard_normal(space.n_dofs)
                  ).reshape(-1, 2)[space.element_nodes]

    reduced = form.element_energies(space.geometry_at(2), u_elements).sum()
    full = form.element_energies(space.geometry_at(4), u_elements).sum()
    assert abs(reduced - full) > 0.05 * abs(full)


def test_a_p2_hyperelastic_solve_converges_and_carries_its_element_type():
    """End to end: a Green-Lagrange solve on P2 converges and comes back as a P2
    ElasticSolution, so it knows its space and its recovered stress reaches the edge
    nodes too."""
    mesh = create_rect_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 4])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 2.0), [0.3, 0.15])
    equation = LinearElastic(200.0, 0.3, kinematics=StrainMeasure.GREEN_LAGRANGE)

    solution = EnergySolver(mesh, equation, bc,
                            element_type=QuadraticTriangleElement).solve()

    assert isinstance(solution, ElasticSolution)
    assert solution.element_type is QuadraticTriangleElement
    assert solution.space.n_nodes > len(mesh.vertices)          # edge nodes exist
    assert solution.nodal_von_mises().shape == (solution.space.n_nodes,)
