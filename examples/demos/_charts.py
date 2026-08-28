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


def tidy_log_axis(ax, steps):
    """Label the axis with the steps actually used.

    These sequences span well under a decade, where a log axis falls back to minor
    ticks like 2x10^-2, which run into each other.
    """
    ax.grid(True, which='both', alpha=0.3)
    # Plain decimals below a thousandth run to more digits than they are worth.
    fmt = '{:.1e}' if min(steps) < 1e-3 else '{:g}'
    ax.set_xticks(steps, [fmt.format(s) for s in steps])
    ax.set_xticks([], minor=True)


def hide_x_ticks(plotter, idx):
    """Drop the x-axis ticks on a tall, thin panel, where the labels only collide; the
    y-axis carries the scale."""
    plotter.get_ax(idx).tick_params(axis='x', labelbottom=False, bottom=False)
