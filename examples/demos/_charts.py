"""Matplotlib helpers shared by more than one demo's figures."""


def share_panel_limits(plotter, n_panels):
    """Give the panels in a row one shared view: the union of the x and y limits each set
    for its own shape, so they share a scale and their baselines and titles line up."""
    axes = [plotter.get_ax((0, c)) for c in range(n_panels)]
    xlo = min(a.get_xlim()[0] for a in axes)
    xhi = max(a.get_xlim()[1] for a in axes)
    ylo = min(a.get_ylim()[0] for a in axes)
    yhi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_xlim(xlo, xhi)
        a.set_ylim(ylo, yhi)
