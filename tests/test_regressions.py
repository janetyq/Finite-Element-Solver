"""Regressions for bugs a type checker found in paths the suite never exercised.

Each of these raised on any call: a misspelled attribute, a dropped argument, a
signature that drifted when Mesh.plot was simplified. They are grouped here
because what they have in common is how they were found, and because each one
marks a path worth keeping covered now that it works.
"""
import numpy as np
import pytest

from fem.mesh.refinement import RedGreenRefiner
from fem.solution import Solution
from fem.topology import TargetCompliance


def test_get_values_converts_vertex_field_to_element_field(make_unit_square):
    """get_values(mode=...) called self._convert_*, which never existed -- the
    conversions live on the mesh, without the underscore."""
    mesh = make_unit_square(6)
    solution = Solution(mesh, n_components=1)
    solution.set_values('u', np.ones(len(mesh.vertices)))

    element_values = solution.get_values('u', mode='element')

    assert len(element_values) == len(mesh.elements)
    assert np.allclose(element_values, 1.0)


def test_get_values_converts_element_field_to_vertex_field(make_unit_square):
    mesh = make_unit_square(6)
    solution = Solution(mesh, n_components=1)
    solution.set_values('rho', np.ones(len(mesh.elements)))

    vertex_values = solution.get_values('rho', mode='vertex')

    assert len(vertex_values) == len(mesh.vertices)
    assert np.allclose(vertex_values, 1.0)


def test_get_values_rejects_an_unknown_mode(make_unit_square):
    """An unrecognised mode fell out of the if/elif chain as None, which only
    failed wherever the caller went on to index it."""
    mesh = make_unit_square(4)
    solution = Solution(mesh, n_components=1)
    solution.set_values('u', np.ones(len(mesh.vertices)))

    with pytest.raises(ValueError, match='unknown mode'):
        solution.get_values('u', mode='nodal')


def test_target_compliance_objective_gradient_is_well_formed():
    """The target-compliance objective yields a finite per-element sensitivity.

    It replaces the old target_compliance_objective/gradient, which called
    self.compliance() with no argument through a string-dispatched _select_objective
    -- a guaranteed TypeError. As an injected object it is just a gradient formula.
    """
    rng = np.random.default_rng(0)
    n = 12
    compliance = rng.random(n) + 0.1
    rho = rng.random(n) + 0.1

    gradient = TargetCompliance(target=1.0).gradient(compliance, rho, penalty=3.0)

    assert np.all(np.isfinite(gradient))
    assert len(gradient) == n


def test_refinement_plot_draws(make_unit_square):
    """plot_refinement colours red/green leaves on a refined mesh."""
    from fem.plot.helpers import plot_refinement
    from fem.plot.plotter import Plotter

    mesh = make_unit_square(4)
    refiner = RedGreenRefiner(mesh)
    mesh = refiner.refine([0])

    ax = Plotter().get_ax()
    plot_refinement(ax, mesh, refiner.leaf_classifications())

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
