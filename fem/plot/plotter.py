from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation

from fem.typing import FloatArray

if TYPE_CHECKING:
    from fem.boundary import BoundaryConditions
    from fem.mesh.mesh import Mesh

from fem.plot.helpers import (
    plot_mesh,
    plot_boundary,
    plot_highlight,
    plot_arrows,
    setup_colorbar,
    plot_colored,
    change_ax_to_ax3d,
    plot_surface,
    plot_refinement,
    plot_solid,
    plot_bc,
)


class PlotMode(Enum):
    MESH = "mesh"
    BOUNDARY = "boundary"
    COLORED = "colored"
    SURFACE = "surface"
    ARROWS = "arrows"
    REFINEMENT = "refinement"
    BC = "bc"
    # The boundary surface of a 3D mesh, coloured. Distinct from SURFACE, which lifts
    # a scalar field over a *2D* mesh into the z direction.
    SOLID = "solid"


# Figures are read on screens, not printed: at matplotlib's default 100 a 5-inch panel
# is 500 px, which upscales blurrily in the gallery and in the README. Set on the
# figure rather than at save time so what is measured on screen is what is written.
DEFAULT_DPI = 150

# A frame sequence pays this per frame, and frames are viewed at a fraction of the size
# of a still, so they are written at matplotlib's default instead.
FRAME_DPI = 100


class Plotter:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        axis_labels: bool = True,
        panel_aspect: float = 1.0,
    ) -> None:
        if figsize is None:
            # `panel_aspect` is the width:height of what each panel draws. The axes are
            # equal-aspect, so a 4:1 beam in a square cell is a thin strip with the rest
            # of the cell empty; sizing the figure by the domain keeps the drawing big.
            # The floor is for the furniture -- title, ticks, colorbar -- which does not
            # shrink with the domain and would otherwise crowd out a very flat panel.
            figsize = (5*ncols, max(3.0, 5/panel_aspect)*nrows)

        # Constrained layout, so panels, their colorbars, and the suptitle are given
        # room rather than overlapping at the default spacing.
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize,
                                          dpi=DEFAULT_DPI, layout='constrained')
        if title is not None:
            self.fig.suptitle(title)
        if nrows == 1 and ncols == 1:
            self.axs = np.array([self.axs])
        self.axs = self.axs.reshape(nrows, ncols)

        # Whether the axes carry x/y/z labels. Off for figures whose axes are the
        # domain itself -- an outline in SVG user units gains nothing from being told
        # its horizontal axis is x, and gains a false suggestion the numbers mean
        # something.
        self.axis_labels = axis_labels

        # Panels whose axes are not the domain's, and the labels they carry instead;
        # see `chart_ax`.
        self._charts: dict[tuple[int, int], tuple[str, str]] = {}

        self.anims = {}
        # The frame-update callable behind each animation, kept because a FuncAnimation
        # renders only through show()/save(); `save_frames` steps these directly.
        self._anim_updates: dict[tuple[int, int], tuple[Callable[[int], None], int]] = {}
        self.cbar_infos = {}

    # function for plotting at a specific index
    def plot(
        self,
        mesh: 'Mesh',
        values: FloatArray | Sequence[float] | None = None,
        mode: PlotMode | str = PlotMode.MESH,
        idx: tuple[int, int] = (0, 0),
        title: str | None = None,
        bc: 'BoundaryConditions | None' = None,
        clear: bool = False,
        empty: bool = False,
        label: str | None = None,
    ) -> None:
        """Draw `values` on `mesh` into the subplot at `idx`.

        `label` names the quantity on the colorbar (colored mode); a colorbar is built
        once per subplot, so it is read on the call that first draws there and ignored
        by later ones redrawing the same axes.
        """
        mode = PlotMode(mode)  # accepts PlotMode or its value; unknown raises ValueError
        ax = self.axs[idx]
        if clear:
            ax.clear()

        if values is not None:
            values = np.array(values)

        # TODO: check that values/bc are provided for intended mode
        if mode is PlotMode.MESH:
            plot_mesh(ax, mesh)
        elif mode is PlotMode.BOUNDARY:
            plot_boundary(ax, mesh)
        elif mode is PlotMode.COLORED:
            cbar_info = plot_colored(ax, mesh, values, cbar_info=self.cbar_infos.get(idx, None),
                                     label=label)
            self.cbar_infos[idx] = cbar_info
        elif mode is PlotMode.SURFACE:
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            plot_surface(ax, mesh, values)
        elif mode is PlotMode.SOLID:
            # The colorbar is set up on the 3D axes, after the swap: attaching it to the
            # 2D one it replaces is what left a stray bar beside a surface animation.
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            if values is not None and idx not in self.cbar_infos:
                self.cbar_infos[idx] = setup_colorbar(
                    ax, (float(np.min(values)), float(np.max(values))), label)
            plot_solid(ax, mesh, values, self.cbar_infos.get(idx))
        elif mode is PlotMode.REFINEMENT:
            plot_refinement(ax, mesh, values)
        elif mode is PlotMode.ARROWS:
            plot_arrows(ax, mesh, values) # inside arrows, assert the correct shape
        elif mode is PlotMode.BC:
            plot_bc(ax, mesh, bc)

        ax.set_title(title) # overrides any existing title
        if empty:
            ax.axis('off')

    def plot_highlights(
        self,
        mesh: 'Mesh',
        idxs_list: Sequence[Any],
        color_list: Sequence[str],
        label_list: Sequence[str],
        mode: str = 'vertices',
        idx: tuple[int, int] = (0, 0),
    ) -> None:
        if not (len(idxs_list) == len(color_list) == len(label_list)):
            raise ValueError("idxs_list, color_list, and label_list must have the same length.")

        ax = self.axs[idx] if isinstance(self.axs, np.ndarray) else self.axs
        plot_highlight(ax, mesh, idxs_list, color_list, label_list, mode=mode)


    # Specialty plotting
    def plot_animation(
        self,
        mesh: 'Mesh',
        values: Sequence[FloatArray],
        mode: PlotMode | str = PlotMode.COLORED,
        idx: tuple[int, int] = (0, 0),
        titles: Sequence[str] | None = None,
        cbar_lims: tuple[float, float] | None = None,
        label: str | None = None,
    ) -> None:
        mode = PlotMode(mode)
        # Bound to a local list so the nested `update` closure keeps the
        # non-optional type; a narrowed parameter does not survive capture.
        frame_titles = list(titles) if titles is not None else [str(i) for i in range(len(values))]

        # Only the colored mode reads a colorbar. A surface animation drew one anyway,
        # onto the 2D axes that change_ax_to_ax3d then replaces -- leaving a stray
        # legend beside a plot that never used it.
        if mode is PlotMode.COLORED:
            # Fixed across frames so they stay comparable, and spanning the series
            # rather than a caller-supplied guess: the default used to be (0, 1),
            # against which any field outside that range rendered as one flat block.
            if cbar_lims is None:
                cbar_lims = (min(np.min(v) for v in values), max(np.max(v) for v in values))
            self.cbar_infos[idx] = setup_colorbar(self.axs[idx], cbar_lims, label=label)

        self.plot(mesh, values[0], mode=mode, idx=idx, title=frame_titles[0])

        def update(frame: int) -> None:
            self.plot(mesh, values[frame], mode=mode, idx=idx, title=frame_titles[frame], clear=True)

        self.anims[idx] = FuncAnimation(self.fig, update, frames=range(len(values)), blit=False, repeat=True)
        self._anim_updates[idx] = (update, len(values))

    def get_ax(self, idx: tuple[int, int] = (0, 0)) -> Axes:
        return self.axs[idx]

    def chart_ax(self, idx: tuple[int, int] = (0, 0), xlabel: str = '',
                 ylabel: str = '') -> Axes:
        """An axes for a plot whose two axes are not the domain's, to draw on directly.

        Every other panel here shows a field over a mesh, so it is given equal aspect
        and labelled x/y. A convergence curve is neither: equal aspect distorts a
        log-log plot, `ticklabel_format` raises on a log scale, and the quantities are
        named by `xlabel`/`ylabel` instead.
        """
        self._charts[idx] = (xlabel, ylabel)
        return self.axs[idx]

    def format_axs(self) -> None:
        for idx, ax in np.ndenumerate(self.axs):
            if idx in self._charts:
                xlabel, ylabel = self._charts[idx]
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            else:
                ax.ticklabel_format(useOffset=False)
                if self.axis_labels:
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                if hasattr(ax, 'get_zlim'):
                    if self.axis_labels:
                        ax.set_zlabel('z')
                    ax.set_aspect('equalxy')
                else:
                    ax.set_aspect('equal')

            # Only where the caller has not placed one itself: `ax.legend()` replaces an
            # existing legend with a default-positioned one, so an explicit `loc` was
            # being discarded here rather than respected.
            if ax.get_legend() is None and any(ax.get_legend_handles_labels()[1]):
                ax.legend()

        self._fit_colorbars()

    def _fit_colorbars(self) -> None:
        """Resize each colorbar to the panel it annotates.

        Constrained layout sizes a colorbar to the whole subplot *cell*, while an
        equal-aspect axes fills only part of that cell -- so a 4:1 domain got a bar
        three times the height of the plot beside it.

        A panel's drawn box is only known once the layout has run, and the layout
        would undo any position set by hand, so it is run once here and then switched
        off. Later calls (a frame sequence makes one per frame) reapply the same
        boxes, which is what `ax.clear()` would otherwise lose.
        """
        if not self.cbar_infos:
            return

        if self.fig.get_layout_engine() is not None:
            self.fig.draw_without_rendering()
            self.fig.set_layout_engine('none')

        for idx, info in self.cbar_infos.items():
            ax: Any = self.axs[idx]
            panel = ax.get_position()
            bar = info.bar.ax.get_position()
            info.bar.ax.set_position([bar.x0, panel.y0, bar.width, panel.height])

    def show(self) -> None:
        self.format_axs()
        plt.show()

    def save(self, path: str, dpi: float | None = None) -> None:
        """Write the figure to `path`, at `DEFAULT_DPI` unless `dpi` overrides it.

        The override is for figures whose weight matters more than their sharpness: a
        grid of 3D surfaces runs to megabytes at the default, which is a lot to ask of
        a README.
        """
        # self.fig, not plt.savefig: pyplot writes the *current* figure, which is
        # whichever was created last, so a caller holding two Plotters saved the second
        # one under both names.
        self.format_axs()
        self.fig.savefig(path, dpi=dpi if dpi is not None else self.fig.dpi)

    def save_frames(self, path_template: str) -> list[str]:
        '''Write each animation frame as a still image; returns the paths written.

        `path_template` is formatted with the frame number, e.g. `'heat/{:03d}.png'`.
        Every animation on this figure is stepped together, which is why this lives
        here rather than on `FuncAnimation`: a figure with two animated panels has two
        of those, and saving through either one leaves the other panel frozen.
        '''
        if not self._anim_updates:
            raise ValueError('this figure has no animation to write frames for')

        paths = []
        for frame in range(self.frame_count()):
            for update, _ in self._anim_updates.values():
                update(frame)
            self.format_axs()
            path = path_template.format(frame)
            self.fig.savefig(path, dpi=FRAME_DPI)
            paths.append(path)
        return paths

    def frame_count(self) -> int:
        '''Frames the animations on this figure share -- the shortest, so every panel
        has something to draw at every step.'''
        return min((n for _, n in self._anim_updates.values()), default=0)

    def close(self) -> None:
        plt.close(self.fig)
        