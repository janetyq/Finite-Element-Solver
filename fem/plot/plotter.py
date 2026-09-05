from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from fem.field import NodalField
from fem.plot.bc import overlay_supports, plot_bc
from fem.plot.helpers import (
    ColorbarInfo,
    change_ax_to_ax3d,
    plot_arrows,
    plot_boundary,
    plot_colored,
    plot_mesh,
    plot_refinement,
    plot_solid,
    plot_surface,
    setup_colorbar,
)
from fem.plot.tessellation import PanelView, panel_view
from fem.typing import FloatArray

if TYPE_CHECKING:
    from fem.conditions import Conditions
    from fem.mesh.mesh import Mesh
    from fem.post.solution import Solution
    from fem.space import FunctionSpace


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


@dataclass(frozen=True)
class Style:
    """How a panel colours what it draws.

    The colouring arguments of `Plotter.plot` collected into one value, so the panel,
    the mode handlers, and an animation's frames all read the same thing instead of
    each forwarding six keywords by hand.
    """
    cmap: str = 'viridis'
    clim: tuple[float, float] | None = None
    label: str | None = None
    log_scale: bool = False
    colorbar: bool = True
    contour: int | None = None


@dataclass(frozen=True)
class Extras:
    """The inputs only one mode reads.

    `conditions` is the BC panel's subject. `values` is the caller's array as given,
    for the two modes that read it rather than the view's field: refinement takes
    red/green classifications, and arrows take the per-node vectors themselves.
    """
    conditions: 'Conditions | None' = None
    values: Any = None


class Panel:
    """One cell of the figure's grid: its axes, and what has been drawn there.

    The axes stay in the plotter's `axs` array, which callers index and which the swap
    to 3D replaces, so a panel reads and writes through that array rather than holding
    an axes of its own.
    """

    def __init__(self, plotter: 'Plotter', idx: tuple[int, int]) -> None:
        self._plotter = plotter
        self.idx = idx
        self.cbar_info: ColorbarInfo | None = None
        # Fixed on the first colour-bearing draw; see `fix_style`.
        self.style: Style | None = None

        # Set when the panel's axes are not the domain's, to the labels they carry
        # instead; see `Plotter.chart_ax`.
        self.chart_labels: tuple[str, str] | None = None

        # A panel of boundary conditions. Its axes are the domain's, so it keeps equal
        # aspect and its ticks, but not the x/y labels: `plot_bc` puts a legend under
        # the panel, which is where those words sit.
        self.is_bc = False

        self.animation: FuncAnimation | None = None
        # The frame-update callable behind the animation, kept because a FuncAnimation
        # renders only through show()/save(); `save_frames` steps these directly.
        self.update: Callable[[int], None] | None = None
        self.n_frames = 0

    @property
    def ax(self) -> Any:
        return self._plotter.axs[self.idx]

    @ax.setter
    def ax(self, ax: Any) -> None:
        self._plotter.axs[self.idx] = ax

    def to_3d(self) -> Any:
        """Swap this panel's axes for 3D ones, in place in the plotter's array."""
        self.ax = change_ax_to_ax3d(self.ax, self._plotter.fig,
                                    self._plotter.axs.shape, self.idx)
        return self.ax

    def fix_style(self, style: Style) -> Style:
        """The style to draw with, fixed on the call that first coloured this panel.

        A colorbar is built once per panel, so the mapping (colormap, limits, label,
        scale) is read off that first call and later calls redrawing the same axes keep
        it. Isolines are drawn afresh each time, so `contour` follows the call.
        """
        if self.style is None:
            self.style = style
        return replace(self.style, contour=style.contour)

    def format(self, axis_labels: bool) -> None:
        ax = self.ax
        if self.chart_labels is not None:
            xlabel, ylabel = self.chart_labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.ticklabel_format(useOffset=False)
            if axis_labels and not self.is_bc:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            if isinstance(ax, Axes3D):
                if axis_labels:
                    ax.set_zlabel('z')
                ax.set_aspect('equalxy')  # pyright: ignore[reportArgumentType]
            else:
                ax.set_aspect('equal')

        # Only where the caller has not placed one itself: `ax.legend()` would
        # replace an existing legend with a default-positioned one.
        if ax.get_legend() is None and any(ax.get_legend_handles_labels()[1]):
            ax.legend()

    def fit_colorbar(self) -> None:
        """Resize this panel's colorbar to the drawn box of the panel it annotates."""
        if self.cbar_info is None or self.cbar_info.bar is None:
            return   # no bar: colour drawn without one (colorbar=False)
        ax: Any = self.ax
        box = ax.get_position()
        bar = self.cbar_info.bar.ax.get_position()
        self.cbar_info.bar.ax.set_position((bar.x0, box.y0, bar.width, box.height))


# -- mode handlers -----------------------------------------------------------------
#
# One handler per `PlotMode`, each drawing into the panel it is given and returning the
# artist an animation can update in place, or `None`. What a mode needs from the call,
# and whether its axes are 3D or it carries a colour mapping, is declared beside it in
# `ModeSpec` rather than read off the branch that draws it.


def _draw_mesh(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_mesh(panel.ax, view)


def _draw_boundary(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_boundary(panel.ax, view)


def _draw_colored(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    if style.clim is not None and panel.cbar_info is None:
        panel.cbar_info = setup_colorbar(panel.ax, style.clim, style.label, style.cmap,
                                         style.log_scale, style.colorbar)
    panel.cbar_info, artist = plot_colored(
        panel.ax, view, cbar_info=panel.cbar_info, label=style.label, cmap_name=style.cmap,
        log_scale=style.log_scale, colorbar=style.colorbar, contour=style.contour)
    return artist


def _draw_surface(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_surface(panel.ax, view, clim=style.clim)


def _draw_solid(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    # The colorbar spans the field as drawn, and is set up on the 3D axes: the swap has
    # already happened, so a bar anchored here is not orphaned by it.
    if view.values is not None and panel.cbar_info is None:
        panel.cbar_info = setup_colorbar(
            panel.ax, (float(np.min(view.values)), float(np.max(view.values))), style.label)
    return plot_solid(panel.ax, view, panel.cbar_info)


def _draw_refinement(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_refinement(panel.ax, view.mesh, extras.values)


def _draw_arrows(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_arrows(panel.ax, view, extras.values)


def _draw_bc(panel: Panel, view: PanelView, style: Style, extras: Extras) -> Any:
    plot_bc(panel.ax, view.mesh, extras.conditions)
    panel.is_bc = True


@dataclass(frozen=True)
class ModeSpec:
    """What one mode draws, and what the call has to supply for it.

    The declaration is checked before anything is drawn, so a mode asked for without
    its field or its conditions is reported as that, rather than surfacing as a shape
    error out of matplotlib several frames later.
    """
    draw: Callable[[Panel, PanelView, Style, Extras], Any]
    needs_values: bool = False
    needs_conditions: bool = False
    axes_3d: bool = False       # the panel's axes are swapped to 3D before drawing
    colored: bool = False       # carries a colour mapping, and so a colorbar


_MODE_SPECS: dict[PlotMode, ModeSpec] = {
    PlotMode.MESH: ModeSpec(_draw_mesh),
    PlotMode.BOUNDARY: ModeSpec(_draw_boundary),
    PlotMode.COLORED: ModeSpec(_draw_colored, needs_values=True, colored=True),
    PlotMode.SURFACE: ModeSpec(_draw_surface, needs_values=True, axes_3d=True),
    # A solid with no field draws its boundary surface plain, so values are optional.
    PlotMode.SOLID: ModeSpec(_draw_solid, axes_3d=True, colored=True),
    PlotMode.REFINEMENT: ModeSpec(_draw_refinement, needs_values=True),
    PlotMode.ARROWS: ModeSpec(_draw_arrows, needs_values=True),
    PlotMode.BC: ModeSpec(_draw_bc, needs_conditions=True),
}


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

        self._panels = {idx: Panel(self, idx) for idx in np.ndindex(nrows, ncols)}

        # Whether the one-off layout pass in `_fit_colorbars` has run. Tracked here
        # rather than read back off the figure because freezing the layout leaves a
        # placeholder engine in place, not `None`, so the figure cannot report it.
        self._layout_frozen = False

    @property
    def anims(self) -> dict[tuple[int, int], FuncAnimation]:
        """The animation on each animated panel, by index."""
        return {idx: p.animation for idx, p in self._panels.items() if p.animation is not None}

    @property
    def cbar_infos(self) -> dict[tuple[int, int], ColorbarInfo]:
        """The colour mapping each coloured panel drew with, by index."""
        return {idx: p.cbar_info for idx, p in self._panels.items() if p.cbar_info is not None}

    def plot(
        self,
        target: "Mesh | Solution | NodalField",
        values: "FloatArray | Sequence[float] | Sequence[str] | NodalField | None" = None,
        mode: PlotMode | str = PlotMode.MESH,
        idx: tuple[int, int] = (0, 0),
        title: str | None = None,
        conditions: 'Conditions | None' = None,
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

        `target` is a `Mesh`, a `Solution`, or a `NodalField`. A field or solution
        supplies both its mesh and its `space`, so a P2 or curved solve renders faithfully
        without passing `space=` by hand (an explicit `space` still overrides it), and
        `warp=True` deforms the field by the target's own displacement. `values` may be
        a `NodalField` too, drawn by node on its own space. A raw mesh keeps the
        low-level, field-agnostic path. Either way the panel draws the `PanelView` that
        `fem.plot.tessellation.panel_view` builds.

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
        style = Style(cmap=cmap if cmap is not None else 'viridis', clim=clim, label=label,
                      log_scale=log_scale, colorbar=colorbar, contour=contour)
        return self._draw(target, values, mode=PlotMode(mode), idx=idx, style=style,
                          title=title, conditions=conditions, clear=clear, empty=empty,
                          space=space, subdivisions=subdivisions, warp=warp)

    def _draw(
        self,
        target: "Mesh | Solution | NodalField",
        values: "FloatArray | Sequence[float] | Sequence[str] | NodalField | None",
        *,
        mode: PlotMode,
        idx: tuple[int, int],
        style: Style,
        title: str | None = None,
        conditions: 'Conditions | None' = None,
        clear: bool = False,
        empty: bool = False,
        space: 'FunctionSpace | None' = None,
        subdivisions: int = 3,
        warp: 'FloatArray | bool | None' = None,
    ) -> Any:
        """Draw one panel with an already-resolved `Style`; `plot` is the public form."""
        spec = _MODE_SPECS[mode]
        if spec.needs_values and values is None:
            raise ValueError(f"the {mode.value} mode draws a field: pass values=")
        if spec.needs_conditions and conditions is None:
            raise ValueError(f"the {mode.value} mode draws boundary conditions: "
                             f"pass conditions=")

        # The refinement mode takes the red/green classifications, not a field.
        field = (None if values is None or mode is PlotMode.REFINEMENT
                 else values if isinstance(values, NodalField)
                 else np.asarray(values, dtype=float))
        view = panel_view(target, field, space=space, warp=warp, subdivisions=subdivisions)

        panel = self._panels[idx]
        if clear:
            panel.ax.clear()
        if spec.axes_3d:
            panel.to_3d()
        if spec.colored:
            style = panel.fix_style(style)

        artist = spec.draw(panel, view, style, Extras(conditions=conditions, values=values))

        panel.ax.set_title(title)  # overrides any existing title
        if empty:
            panel.ax.axis('off')
        return artist

    def overlay_supports(
        self,
        mesh: 'Mesh',
        conditions: 'Conditions',
        idx: tuple[int, int] = (0, 0),
        coords: FloatArray | None = None,
    ) -> None:
        """Overlay support/load glyphs on the panel at `idx` (see `fem.plot.bc`).

        `coords` places them at deformed vertex positions (for a buckled shape, so a
        load follows the material) while the conditions are still read off `mesh`.
        """
        overlay_supports(self.axs[idx], mesh, conditions, coords=coords)

    # Specialty plotting
    def plot_animation(
        self,
        target: "Mesh | Solution",
        values: "Sequence[FloatArray] | FloatArray",
        mode: PlotMode | str = PlotMode.COLORED,
        idx: tuple[int, int] = (0, 0),
        titles: Sequence[str] | None = None,
        clim: tuple[float, float] | None = None,
        label: str | None = None,
        meshes: "Sequence['Mesh'] | None" = None,
        cmap: str | None = None,
        log_scale: bool = False,
        contour: int | None = None,
        space: 'FunctionSpace | None' = None,
        subdivisions: int = 3,
    ) -> None:
        """Animate `values` over the panel at `idx`.

        `target`, `space`, and `subdivisions` are as in `plot`, so a P2 or curved field
        animates on the same tessellation it is drawn on. `clim` fixes the colour range
        across the series (default: the extremes over every frame); `cmap`, `label`,
        `log_scale`, and `contour` colour the panel as they do a still. On the
        fixed-mesh path a frame only recolours the one collection drawn, so `contour`
        isolines there are the first frame's and stay put.

        `meshes` supplies one mesh per frame when the geometry moves (a vibration
        mode flexing, say) rather than a field changing over fixed geometry. The
        colours still come from `values`, but a moving mesh cannot be recoloured in
        place, so each frame redraws within bounds fixed across the series (so the shape
        flexes in view instead of the axes rescaling to follow it). Leave it `None` for
        the fixed-mesh case, whose recolour path is left untouched.
        """
        mode = PlotMode(mode)
        spec = _MODE_SPECS[mode]
        # Bound to local lists so the nested `update` closure keeps the non-optional
        # type; a narrowed parameter does not survive capture.
        frame_titles = list(titles) if titles is not None else [str(i) for i in range(len(values))]
        frame_meshes = list(meshes) if meshes is not None else None
        style = Style(cmap=cmap if cmap is not None else 'viridis', clim=clim, label=label,
                      log_scale=log_scale, contour=contour)
        panel = self._panels[idx]

        # Colored and solid are the two modes that read a colorbar; surface draws one
        # anyway onto the 2D axes that the swap to 3D then replaces, leaving a stray
        # legend beside a plot that never used it.
        if spec.colored:
            # Fixed across frames so they stay comparable, spanning the whole series.
            if style.clim is None:
                style = replace(style, clim=(min(np.min(v) for v in values),
                                             max(np.max(v) for v in values)))
            if spec.axes_3d:
                # The swap to 3D has to happen first: a colorbar anchored to the 2D
                # axes orphans when the axes are swapped out from under it.
                panel.to_3d()
            # Built up front, so the colorbar spans the whole series rather than just
            # frame 0, and fixes the style every frame then draws with.
            panel.style = style
            panel.cbar_info = setup_colorbar(panel.ax, style.clim, label=style.label,
                                             cmap_name=style.cmap, log_scale=style.log_scale,
                                             colorbar=style.colorbar)

        base = frame_meshes[0] if frame_meshes is not None else target
        artist = self._draw(base, values[0], mode=mode, idx=idx, style=style,
                            title=frame_titles[0], space=space, subdivisions=subdivisions)

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
                self._draw(frame_meshes[frame], values[frame], mode=mode, idx=idx,
                           style=style, title=frame_titles[frame], clear=True)
                ax = self.axs[idx]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal')
        # Colored and solid over a fixed mesh draw one collection, so a frame only
        # changes its colours; recolour that artist in place rather than clearing the
        # axes and rebuilding it, which re-lays out every tick and label each frame and
        # was the bulk of an animated demo's render cost. Surface lifts the field into
        # z, so its geometry changes frame to frame and it has to be redrawn.
        elif spec.colored and artist is not None:
            ax = self.axs[idx]
            view = panel_view(target, values[0], space=space, subdivisions=subdivisions)

            def update(frame: int) -> None:
                artist.set_array(view.with_values(values[frame]).face_values)
                ax.set_title(frame_titles[frame])
        else:
            def update(frame: int) -> None:
                self._draw(target, values[frame], mode=mode, idx=idx, style=style,
                           title=frame_titles[frame], clear=True, space=space,
                           subdivisions=subdivisions)

        panel.animation = FuncAnimation(self.fig, update, frames=range(len(values)),
                                        blit=False, repeat=True)
        panel.update = update
        panel.n_frames = len(values)

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
        self._panels[idx].chart_labels = (xlabel, ylabel)
        return self.axs[idx]

    def format_axs(self) -> None:
        for panel in self._panels.values():
            panel.format(self.axis_labels)
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

        for panel in self._panels.values():
            panel.fit_colorbar()

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
        animated = [p for p in self._panels.values() if p.update is not None]
        if not animated:
            raise ValueError('this figure has no animation to write frames for')

        frames = range(self.frame_count())
        if max_frames is not None and self.frame_count() > max_frames:
            frames = np.unique(np.linspace(0, self.frame_count() - 1, max_frames).astype(int))

        for image, frame in enumerate(frames):
            for panel in animated:
                assert panel.update is not None
                panel.update(int(frame))
            self.format_axs()
            yield image

    def frame_count(self) -> int:
        '''Frames the animations on this figure share: the shortest, so every panel
        has something to draw at every step.'''
        return min((p.n_frames for p in self._panels.values() if p.update is not None),
                   default=0)

    def close(self) -> None:
        plt.close(self.fig)
