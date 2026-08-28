from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from fem.typing import FloatArray

if TYPE_CHECKING:
    from fem.boundary import BoundaryConditions
    from fem.mesh.mesh import Mesh
    from fem.post.solution import Solution
    from fem.space import FunctionSpace

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
)
from fem.plot.bc import overlay_supports, plot_bc
from fem.plot.tessellation import field_view


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
# GIFs carry every frame in one file, so they render lighter than the player frames.
GIF_DPI = 80


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
            # The floor is for the furniture (title, ticks, colorbar), which does not
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
        # domain itself: an outline in SVG user units gains nothing from being told
        # its horizontal axis is x, and gains a false suggestion the numbers mean
        # something.
        self.axis_labels = axis_labels

        # Panels whose axes are not the domain's, and the labels they carry instead;
        # see `chart_ax`.
        self._charts: dict[tuple[int, int], tuple[str, str]] = {}

        # Panels of boundary conditions. Their axes are the domain's, so they keep
        # equal aspect and their ticks, but not the x/y labels: `plot_bc` puts a legend
        # under the panel, which is where those words sit.
        self._bc_panels: set[tuple[int, int]] = set()

        self.anims = {}
        # The frame-update callable behind each animation, kept because a FuncAnimation
        # renders only through show()/save(); `save_frames` steps these directly.
        self._anim_updates: dict[tuple[int, int], tuple[Callable[[int], None], int]] = {}
        self.cbar_infos = {}
        # Whether the one-off layout pass in `_fit_colorbars` has run. Tracked here
        # rather than read back off the figure because freezing the layout leaves a
        # placeholder engine in place, not `None`, so the figure cannot report it.
        self._layout_frozen = False

    def plot(
        self,
        target: "Mesh | Solution",
        values: FloatArray | Sequence[float] | Sequence[str] | None = None,
        mode: PlotMode | str = PlotMode.MESH,
        idx: tuple[int, int] = (0, 0),
        title: str | None = None,
        bc: 'BoundaryConditions | None' = None,
        clear: bool = False,
        empty: bool = False,
        label: str | None = None,
        clim: tuple[float, float] | None = None,
        cmap: str | None = None,
        log_scale: bool = False,
        colorbar: bool = True,
        contour: int | None = None,
        space: 'FunctionSpace | None' = None,
        subdivisions: int = 3,
        warp: 'FloatArray | bool | None' = None,
    ) -> Any:
        """Draw `values` on `target` into the subplot at `idx`.

        `target` is a `Mesh` or a `Solution`. A solution supplies both its mesh and its
        `space`, so a P2 or curved solve renders faithfully without passing `space=` by
        hand (an explicit `space` still overrides it), and `warp=True` deforms the field
        by the solution's own displacement. A raw mesh keeps the low-level,
        field-agnostic path. Either way the panel draws the `FieldView` that
        `fem.plot.tessellation.field_view` builds.

        `label` names the quantity on the colorbar (colored mode); a colorbar is built
        once per subplot, so it is read on the call that first draws there and ignored
        by later ones redrawing the same axes. `colorbar=False` colours by value but
        draws no bar, for a qualitative or arbitrary-amplitude field (a mode shape).

        `clim` fixes the colour range instead of taking it from `values`, so a grid of
        panels can be compared. Each panel otherwise renormalizes to its own extremes,
        and six snapshots of a decaying field draw as six near-identical squares.

        `cmap` selects the colormap (default 'viridis'); `log_scale` uses logarithmic
        normalization. `contour=n` overlays n isolines of the field on the colored
        panel (the level sets of a scalar, e.g. a potential's equipotentials).

        `space` opts a P2 or curved solve into a faithful render: the mesh, colored,
        surface, and arrow panels draw on a `subdivisions`-fine tessellation of each
        element, so a curved boundary follows its true curve and a quadratic field shows
        its within-element curvature instead of being flattened to one triangle. Omitted,
        the P1 path draws the straight-sided mesh. `values` for the colored and surface
        modes must then be a per-node field (length `space.n_nodes`, e.g. a solution
        vector); arrows take a per-node vector field.

        `warp` is an optional nodal displacement `(n_nodes, spatial)` that draws the
        deformed configuration, so a stress field draws on the warped shape. Pass
        `warp=True` (with a Solution as the target) to deform by that solution's own
        displacement.

        Returns the recolourable collection for the colored and solid modes (the
        artist an animation updates in place across frames), and `None` otherwise.
        """
        mode = PlotMode(mode)  # accepts PlotMode or its value; unknown raises ValueError
        # The refinement mode takes the red/green classifications, not a field.
        field = (None if values is None or mode is PlotMode.REFINEMENT
                 else np.asarray(values, dtype=float))
        view = field_view(target, field, space=space, warp=warp, subdivisions=subdivisions)
        mesh = view.mesh
        ax = self.axs[idx]
        if clear:
            ax.clear()

        artist = None
        # TODO: check that values/bc are provided for intended mode
        if mode is PlotMode.MESH:
            plot_mesh(ax, view)
        elif mode is PlotMode.BOUNDARY:
            plot_boundary(ax, view)
        elif mode is PlotMode.COLORED:
            cmap_name = cmap if cmap is not None else 'viridis'
            if clim is not None and idx not in self.cbar_infos:
                self.cbar_infos[idx] = setup_colorbar(ax, clim, label, cmap_name, log_scale, colorbar)
            cbar_info, artist = plot_colored(ax, view, cbar_info=self.cbar_infos.get(idx, None),
                                             label=label, cmap_name=cmap_name, log_scale=log_scale,
                                             colorbar=colorbar, contour=contour)
            self.cbar_infos[idx] = cbar_info
        elif mode is PlotMode.SURFACE:
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            plot_surface(ax, view, clim=clim)
        elif mode is PlotMode.SOLID:
            # The colorbar is set up on the 3D axes, after the swap.
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            if view.values is not None and idx not in self.cbar_infos:
                self.cbar_infos[idx] = setup_colorbar(
                    ax, (float(np.min(view.values)), float(np.max(view.values))), label)
            artist = plot_solid(ax, view, self.cbar_infos.get(idx))
        elif mode is PlotMode.REFINEMENT:
            plot_refinement(ax, mesh, values)
        elif mode is PlotMode.ARROWS:
            plot_arrows(ax, view, values)
        elif mode is PlotMode.BC:
            plot_bc(ax, mesh, bc)
            self._bc_panels.add(idx)

        ax.set_title(title) # overrides any existing title
        if empty:
            ax.axis('off')
        return artist

    def overlay_supports(
        self,
        mesh: 'Mesh',
        bc: 'BoundaryConditions',
        idx: tuple[int, int] = (0, 0),
        coords: FloatArray | None = None,
    ) -> None:
        """Overlay support/load glyphs on the panel at `idx` (see `fem.plot.bc`).

        `coords` places them at deformed vertex positions (for a buckled shape, so a
        load follows the material) while the conditions are still read off `mesh`.
        """
        overlay_supports(self.axs[idx], mesh, bc, coords=coords)

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
        target: "Mesh | Solution",
        values: Sequence[FloatArray],
        mode: PlotMode | str = PlotMode.COLORED,
        idx: tuple[int, int] = (0, 0),
        titles: Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        label: str | None = None,
        meshes: "Sequence['Mesh'] | None" = None,
        cmap: str | None = None,
        space: 'FunctionSpace | None' = None,
        subdivisions: int = 3,
    ) -> None:
        """Animate `values` over the panel at `idx`.

        `target`, `space`, and `subdivisions` are as in `plot`, so a P2 or curved field
        animates on the same tessellation it is drawn on. `clim` fixes the colour range
        across the series (default: the extremes over every frame).

        `meshes` supplies one mesh per frame when the geometry moves (a vibration
        mode flexing, say) rather than a field changing over fixed geometry. The
        colours still come from `values`, but a moving mesh cannot be recoloured in
        place, so each frame redraws within bounds fixed across the series (so the shape
        flexes in view instead of the axes rescaling to follow it). Leave it `None` for
        the fixed-mesh case, whose recolour path is left untouched.
        """
        mode = PlotMode(mode)
        # Bound to local lists so the nested `update` closure keeps the non-optional
        # type; a narrowed parameter does not survive capture.
        frame_titles = list(titles) if titles is not None else [str(i) for i in range(len(values))]
        frame_meshes = list(meshes) if meshes is not None else None
        cmap_name = cmap if cmap is not None else 'viridis'

        # Colored and solid are the two modes that read a colorbar; surface draws one
        # anyway onto the 2D axes that change_ax_to_ax3d then replaces, leaving a stray
        # legend beside a plot that never used it.
        if mode in (PlotMode.COLORED, PlotMode.SOLID):
            # Fixed across frames so they stay comparable, spanning the whole series.
            if clim is None:
                clim = (min(np.min(v) for v in values), max(np.max(v) for v in values))
            ax = self.axs[idx]
            if mode is PlotMode.SOLID:
                # Built up front, so the colorbar spans the whole series rather than
                # just frame 0; `plot`'s own SOLID branch only sets one up when idx
                # has none yet, which this pre-empts. The swap to 3D axes has to happen
                # first: a colorbar anchored to the 2D axes orphans when `plot` swaps
                # it out from under it.
                ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
                self.axs[idx] = ax
            self.cbar_infos[idx] = setup_colorbar(ax, clim, label=label, cmap_name=cmap_name)

        base = frame_meshes[0] if frame_meshes is not None else target
        artist = self.plot(base, values[0], mode=mode, idx=idx, title=frame_titles[0],
                           cmap=cmap, space=space, subdivisions=subdivisions)

        if frame_meshes is not None:
            # Moving geometry: a fixed collection cannot be recoloured into a new shape,
            # so redraw the deformed mesh each frame. Bounds are frozen over the whole
            # series (the union of every frame's extent, with a margin) so the shape
            # swings within a still frame rather than the axes chasing it.
            all_vertices = np.concatenate([m.vertices for m in frame_meshes])
            lo, hi = all_vertices.min(axis=0), all_vertices.max(axis=0)
            margin = 0.05 * (hi - lo).max()
            xlim = (float(lo[0] - margin), float(hi[0] + margin))
            ylim = (float(lo[1] - margin), float(hi[1] + margin))

            def update(frame: int) -> None:
                self.plot(frame_meshes[frame], values[frame], mode=mode, idx=idx,
                          title=frame_titles[frame], clear=True, cmap=cmap)
                ax = self.axs[idx]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal')
        # Colored and solid over a fixed mesh draw one collection, so a frame only
        # changes its colours; recolour that artist in place rather than clearing the
        # axes and rebuilding it, which re-lays out every tick and label each frame and
        # was the bulk of an animated demo's render cost. Surface lifts the field into
        # z, so its geometry changes frame to frame and it has to be redrawn.
        elif mode in (PlotMode.COLORED, PlotMode.SOLID) and artist is not None:
            ax = self.axs[idx]
            view = field_view(target, values[0], space=space, subdivisions=subdivisions)

            def update(frame: int) -> None:
                artist.set_array(view.with_values(values[frame]).face_values)
                ax.set_title(frame_titles[frame])
        else:
            def update(frame: int) -> None:
                self.plot(target, values[frame], mode=mode, idx=idx, title=frame_titles[frame],
                          clear=True, space=space, subdivisions=subdivisions)

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
                if self.axis_labels and idx not in self._bc_panels:
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                if isinstance(ax, Axes3D):
                    if self.axis_labels:
                        ax.set_zlabel('z')
                    ax.set_aspect('equalxy')  # pyright: ignore[reportArgumentType]
                else:
                    ax.set_aspect('equal')

            # Only where the caller has not placed one itself: `ax.legend()` would
            # replace an existing legend with a default-positioned one.
            if ax.get_legend() is None and any(ax.get_legend_handles_labels()[1]):
                ax.legend()

        self._fit_colorbars()

    def _fit_colorbars(self) -> None:
        """Resize each colorbar to the panel it annotates.

        Constrained layout sizes a colorbar to the whole subplot cell, while an
        equal-aspect axes fills only part of that cell, so a 4:1 domain got a bar
        three times the height of the plot beside it.

        A panel's drawn box is only known once the layout has run, and the layout
        would undo any position set by hand, so it is run once here and then switched
        off. Later calls (a frame sequence makes one per frame) reapply the same
        boxes, which `ax.clear()` would otherwise lose.
        """
        # Nothing to reposition when no panel drew a bar (an all-`colorbar=False` figure
        # still records mappings here). Skip the layout freeze too, so constrained layout
        # keeps managing the figure and reserves room for a suptitle or supxlabel.
        if not any(info.bar is not None for info in self.cbar_infos.values()):
            return

        if not self._layout_frozen:
            self.fig.draw_without_rendering()
            self.fig.set_layout_engine('none')
            self._layout_frozen = True

        for idx, info in self.cbar_infos.items():
            if info.bar is None:   # colour drawn without a bar (colorbar=False)
                continue
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
        # self.fig, not plt.savefig: pyplot writes the current figure, which is
        # whichever was created last, so a caller holding two Plotters saved the second
        # one under both names.
        self.format_axs()
        self.fig.savefig(path, dpi=dpi if dpi is not None else self.fig.dpi)

    def save_frames(self, path_template: str, max_frames: int | None = None) -> list[str]:
        '''Write animation frames as still images; returns the paths written.

        `path_template` is formatted with the frame number, e.g. `'heat/{:03d}.png'`.
        Every animation on this figure is stepped together, so a figure with two
        animated panels writes both. `max_frames` samples the run down to at most that
        many images.
        '''
        paths = []
        for image in self._step_frames(max_frames):
            # Numbered by image rather than by frame, so a sampled run still writes a
            # contiguous 000, 001, 002 for the player to step through.
            path = path_template.format(image)
            self.fig.savefig(path, dpi=FRAME_DPI)
            paths.append(path)
        return paths

    def save_gif(self, path: str, max_frames: int | None = None, frame_ms: int = 80,
                 dpi: float = GIF_DPI) -> None:
        '''Write the animation to `path` as a looping GIF, `frame_ms` per frame.

        `max_frames` samples the run as `save_frames` does. Each frame is quantized to
        its own palette, so a GIF of a colour map stays faithful frame to frame.
        '''
        import io
        from PIL import Image

        images = []
        for _ in self._step_frames(max_frames):
            buffer = io.BytesIO()
            self.fig.savefig(buffer, format='png', dpi=dpi)
            buffer.seek(0)
            images.append(Image.open(buffer).convert('RGB').quantize(colors=256))
        images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms,
                       loop=0, optimize=False)

    def _step_frames(self, max_frames: int | None):
        '''Step every animation on this figure through the run, yielding the image
        number after each frame is drawn; sampled down to `max_frames` evenly, keeping
        both ends, so the last frame (a topology optimization's result) is never
        dropped.'''
        if not self._anim_updates:
            raise ValueError('this figure has no animation to write frames for')

        frames = range(self.frame_count())
        if max_frames is not None and self.frame_count() > max_frames:
            frames = np.unique(np.linspace(0, self.frame_count() - 1, max_frames).astype(int))

        for image, frame in enumerate(frames):
            for update, _ in self._anim_updates.values():
                update(int(frame))
            self.format_axs()
            yield image

    def frame_count(self) -> int:
        '''Frames the animations on this figure share: the shortest, so every panel
        has something to draw at every step.'''
        return min((n for _, n in self._anim_updates.values()), default=0)

    def close(self) -> None:
        plt.close(self.fig)
        