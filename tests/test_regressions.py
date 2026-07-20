"""Regressions for bugs a type checker found in paths the suite never exercised.

Each of these raised on any call: a misspelled attribute, a dropped argument, a
signature that drifted when Mesh.plot was simplified. They are grouped here
because what they have in common is how they were found, and because each one
marks a path worth keeping covered now that it works.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.mesh.refinement import RedGreenRefiner
from fem.regions import on_plane
from fem.solution import Solution
from fem.solver import LinearElastic
from fem.topology import TopologyOptimizer


def test_get_values_converts_vertex_field_to_element_field(make_unit_square):
    """get_values(mode=...) called self._convert_*, which never existed -- the
    conversions live on the mesh, without the underscore."""
    femesh = make_unit_square(6)
    solution = Solution(femesh, dim=1)
    solution.set_values('u', np.ones(len(femesh.vertices)))

    element_values = solution.get_values('u', mode='element')

    assert len(element_values) == len(femesh.elements)
    assert np.allclose(element_values, 1.0)


def test_get_values_converts_element_field_to_vertex_field(make_unit_square):
    femesh = make_unit_square(6)
    solution = Solution(femesh, dim=1)
    solution.set_values('rho', np.ones(len(femesh.elements)))

    vertex_values = solution.get_values('rho', mode='vertex')

    assert len(vertex_values) == len(femesh.vertices)
    assert np.allclose(vertex_values, 1.0)


def test_get_values_rejects_an_unknown_mode(make_unit_square):
    """An unrecognised mode fell out of the if/elif chain as None, which only
    failed wherever the caller went on to index it."""
    femesh = make_unit_square(4)
    solution = Solution(femesh, dim=1)
    solution.set_values('u', np.ones(len(femesh.vertices)))

    with pytest.raises(ValueError, match='unknown mode'):
        solution.get_values('u', mode='nodal')


def test_target_compliance_objective_is_callable(make_unit_square):
    """target_compliance_objective/gradient called self.compliance() with no
    argument, so selecting the objective was a guaranteed TypeError."""
    femesh = make_unit_square(5)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])

    optimizer = TopologyOptimizer(
        femesh, LinearElastic(E=1.0, nu=0.3), bc, iters=1, volume_frac=0.5
    )
    optimizer.solver.solve()

    objective, gradient = optimizer._select_objective('target_compliance')

    assert np.isfinite(objective([0.0]))
    assert len(gradient([0.0])) == len(femesh.elements)


def test_refinement_mesh_plot_draws(make_unit_square):
    """RedGreenRefiner.plot passed ax/linewidth/color to Mesh.plot, which takes
    no arguments since the Plotter decoupling."""
    femesh = make_unit_square(4)
    refiner = RedGreenRefiner(femesh)
    refiner.refine([0])

    ax = refiner.plot(title='refinement')

    assert ax.has_data()


def test_douglas_peucker_handles_a_collinear_run():
    """No interior point beats a zero distance on a straight run, so the index
    stayed None and the recursion sliced with None + 1."""
    # fem.mesh.svg imports svg.path, which is the optional `svg` extra.
    pytest.importorskip('svg.path')
    from fem.mesh.svg import douglas_peucker

    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    simplified = douglas_peucker(points, epsilon=0.0)

    assert len(simplified) == 2
