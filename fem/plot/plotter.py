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


class Plotter:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> None:
        if figsize is None:
            figsize = (5*ncols, 5*nrows)
        
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize)
        if title is not None:
            self.fig.suptitle(title)
        if nrows == 1 and ncols == 1:
            self.axs = np.array([self.axs])
        self.axs = self.axs.reshape(nrows, ncols)
        
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
    ) -> None:
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
            cbar_info = plot_colored(ax, mesh, values, cbar_info=self.cbar_infos.get(idx, None))
            self.cbar_infos[idx] = cbar_info
        elif mode is PlotMode.SURFACE:
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            plot_surface(ax, mesh, values)
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
            self.cbar_infos[idx] = setup_colorbar(self.axs[idx], cbar_lims, label=None)

        self.plot(mesh, values[0], mode=mode, idx=idx, title=frame_titles[0])

        def update(frame: int) -> None:
            self.plot(mesh, values[frame], mode=mode, idx=idx, title=frame_titles[frame], clear=True)

        self.anims[idx] = FuncAnimation(self.fig, update, frames=range(len(values)), blit=False, repeat=True)
        self._anim_updates[idx] = (update, len(values))

    def get_ax(self, idx: tuple[int, int] = (0, 0)) -> Axes:
        return self.axs[idx]

    def format_axs(self) -> None:
        for ax in self.axs.ravel():
            ax.ticklabel_format(useOffset=False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if hasattr(ax, 'get_zlim'):
                ax.set_label('z')
                ax.set_aspect('equalxy')
            else:
                ax.set_aspect('equal')

            if any(ax.get_legend_handles_labels()[1]):
                ax.legend()

    def show(self) -> None:
        self.format_axs()
        plt.show()

    def save(self, path: str) -> None:
        # self.fig, not plt.savefig: pyplot writes the *current* figure, which is
        # whichever was created last, so a caller holding two Plotters saved the second
        # one under both names.
        self.format_axs()
        self.fig.savefig(path)

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
            self.fig.savefig(path)
            paths.append(path)
        return paths

    def frame_count(self) -> int:
        '''Frames the animations on this figure share -- the shortest, so every panel
        has something to draw at every step.'''
        return min((n for _, n in self._anim_updates.values()), default=0)

    def close(self) -> None:
        plt.close(self.fig)
        