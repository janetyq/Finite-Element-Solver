"""The quadrature-sampled load beats the nodal shortcut, and both stay second order."""
import numpy as np

from mms import ConvergenceStudy, load_comparison_convergence


def test_sampled_load_beats_nodal_at_every_resolution():
    solves = load_comparison_convergence((11, 21, 41))
    for solve in solves:
        assert solve.sampled_error < solve.nodal_error


def test_both_loads_are_second_order():
    solves = load_comparison_convergence((11, 21, 41))
    steps = np.array([s.h for s in solves])
    nodal = ConvergenceStudy(steps, np.array([s.nodal_error for s in solves]))
    sampled = ConvergenceStudy(steps, np.array([s.sampled_error for s in solves]))
    # The under-resolved coarse mesh pulls the nodal rate slightly below 2; both are
    # comfortably second order by the fit.
    assert nodal.fitted_order > 1.7
    assert sampled.fitted_order > 1.8
