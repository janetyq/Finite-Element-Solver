"""Regressions for bugs a type checker found in paths the suite never exercised.

Each of these raised on any call: a misspelled attribute, a dropped argument, a
signature that drifted when Mesh.plot was simplified. They are grouped here
because what they have in common is how they were found, and because each one
marks a path worth keeping covered now that it works.
"""
import numpy as np

from fem.mesh.refinement import RedGreenRefiner
from fem.topology import TargetCompliance


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
    from fem.mesh.svg import douglas_peucker

    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    simplified = douglas_peucker(points, epsilon=0.0)

    assert len(simplified) == 2
