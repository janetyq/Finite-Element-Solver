from enum import Enum
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fem.plot.helpers import (
    plot_mesh,
    plot_boundary,
    plot_highlight,
    plot_arrows,
    plot_colored,
    plot_surface,
    plot_bc,
)


class PlotMode(Enum):
    MESH = "mesh"
    BOUNDARY = "boundary"
    COLORED = "colored"
    SURFACE = "surface"
    ARROWS = "arrows"
    BC = "bc"


class Plotter:
    def __init__(self, nrows=1, ncols=1, figsize=None, title=None):
        self.nrows, self.ncols = nrows, ncols
        if figsize is None:
            figsize = (400*ncols, 400*nrows)

        # Every cell is a 3D ('scene') subplot -- a 2D mesh is just a 3D mesh
        # viewed from above (z=0), which is what lets 2D and 3D meshes (and
        # tet-mesh boundary surfaces) share one rendering path.
        specs = [[{'type': 'scene'} for _ in range(ncols)] for _ in range(nrows)]
        self.fig = make_subplots(rows=nrows, cols=ncols, specs=specs)
        self.fig.update_layout(title=title, width=figsize[0], height=figsize[1])

        self.anims = []  # truthy/falsy only -- examples/cli.py depends on that
        self._anim_specs = []
        self._trace_idxs = {}            # idx -> [fig.data indices currently shown there]
        self._title_annotation_idx = {}  # idx -> position within fig.layout.annotations
        self._surface_idxs = set()       # idx cells that used 'surface' mode (want the angled default camera to show height; other cells are flat and read better top-down)

    def _scene_key(self, idx):
        n = idx[0]*self.ncols + idx[1] + 1
        return 'scene' if n == 1 else f'scene{n}'

    def _build_trace(self, mesh, values, mode, bc=None, cmin=None, cmax=None):
        # TODO: check that values/bc are provided for intended mode
        if mode is PlotMode.MESH:
            return plot_mesh(mesh)
        elif mode is PlotMode.BOUNDARY:
            return plot_boundary(mesh)
        elif mode is PlotMode.COLORED:
            return plot_colored(mesh, values, cmin=cmin, cmax=cmax)
        elif mode is PlotMode.SURFACE:
            return plot_surface(mesh, values, cmin=cmin, cmax=cmax)
        elif mode is PlotMode.ARROWS:
            return plot_arrows(mesh, values)  # inside arrows, assert the correct shape
        elif mode is PlotMode.BC:
            return plot_bc(mesh, bc)

    def _hide_traces(self, idx):
        for trace_idx in self._trace_idxs.get(idx, []):
            self.fig.data[trace_idx].visible = False
        self._trace_idxs[idx] = []

    def _set_subplot_title(self, idx, title):
        if title is None:
            return
        if idx in self._title_annotation_idx:
            self.fig.layout.annotations[self._title_annotation_idx[idx]].text = title
            return
        domain = self.fig.layout[self._scene_key(idx)].domain
        x_center = (domain.x[0] + domain.x[1]) / 2
        self.fig.add_annotation(x=x_center, y=min(domain.y[1] + 0.04, 1.0),
                                 xref='paper', yref='paper', text=title,
                                 showarrow=False, font=dict(size=13), yanchor='bottom')
        self._title_annotation_idx[idx] = len(self.fig.layout.annotations) - 1

    def _set_empty(self, idx):
        self.fig.layout[self._scene_key(idx)].update(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        )

    # function for plotting at a specific index
    def plot(self, mesh, values=None, mode=PlotMode.MESH, idx=(0, 0), title=None, bc=None, clear=False, empty=False):
        mode = PlotMode(mode)  # accepts PlotMode or its value; unknown raises ValueError
        if values is not None:
            values = np.asarray(values)
        if mode is PlotMode.SURFACE:
            self._surface_idxs.add(idx)

        if clear:
            self._hide_traces(idx)

        trace_or_traces = self._build_trace(mesh, values, mode, bc=bc)
        traces = trace_or_traces if isinstance(trace_or_traces, list) else [trace_or_traces]
        for trace in traces:
            self.add_trace(trace, idx=idx)

        self._set_subplot_title(idx, title)  # overrides any existing title
        if empty:
            self._set_empty(idx)

    def plot_highlights(self, mesh, idxs_list, color_list, label_list, mode='vertices', idx=(0, 0)):
        if not (len(idxs_list) == len(color_list) == len(label_list)):
            raise ValueError("idxs_list, color_list, and label_list must have the same length.")

        for trace in plot_highlight(mesh, idxs_list, color_list, label_list, mode=mode):
            self.add_trace(trace, idx=idx)

    # Specialty plotting
    def plot_animation(self, mesh, values, mode=PlotMode.COLORED, idx=(0, 0), titles=None, cbar_lims=None):
        mode = PlotMode(mode)
        if mode is PlotMode.SURFACE:
            self._surface_idxs.add(idx)
        values = [np.asarray(v) for v in values]
        if titles is None:
            titles = [str(i) for i in range(len(values))]
        if cbar_lims is None:
            # Fixed across all frames, computed from the actual data instead of a
            # placeholder default -- the matplotlib version's (0, 1) default only
            # ever applied to 'colored' mode (plot_trisurf never took a norm at
            # all), so an unheeded caller got a colorbar with no relation to the
            # data (e.g. a ~300-350 temperature field clipped to (0, 1)).
            flat = np.concatenate([v.flatten() for v in values])
            cbar_lims = (float(flat.min()), float(flat.max()))

        trace = self._build_trace(mesh, values[0], mode, cmin=cbar_lims[0], cmax=cbar_lims[1])
        self.add_trace(trace, idx=idx)
        trace_idx = self._trace_idxs[idx][-1]
        self._set_subplot_title(idx, titles[0])

        self._anim_specs.append(dict(
            mesh=mesh, values=values, mode=mode, titles=titles,
            cmin=cbar_lims[0], cmax=cbar_lims[1], trace_idx=trace_idx,
        ))
        self.anims.append(idx)

    def add_trace(self, trace, idx=(0, 0)):
        self.fig.add_trace(trace, row=idx[0] + 1, col=idx[1] + 1)
        self._trace_idxs.setdefault(idx, []).append(len(self.fig.data) - 1)

    def format_axs(self):
        for row in range(self.nrows):
            for col in range(self.ncols):
                idx = (row, col)
                layout_kwargs = dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data')
                if idx not in self._surface_idxs:
                    # Flat content (mesh/colored/arrows/bc) reads as a 2D plot
                    # should -- straight down the z-axis -- rather than
                    # foreshortened by Plotly's default angled 3D camera. Cells
                    # that used 'surface' mode keep that default, since an
                    # overhead view would hide the very height it's plotting.
                    layout_kwargs['camera'] = dict(eye=dict(x=0, y=0, z=2.0), up=dict(x=0, y=1, z=0))
                self.fig.layout[self._scene_key(idx)].update(**layout_kwargs)

    def _finalize(self):
        self.format_axs()
        if not self._anim_specs:
            return

        n_frames = len(self._anim_specs[0]['values'])
        frames = []
        for i in range(n_frames):
            data = [
                self._build_trace(spec['mesh'], spec['values'][i], spec['mode'],
                                   cmin=spec['cmin'], cmax=spec['cmax'])
                for spec in self._anim_specs
            ]
            frames.append(go.Frame(name=str(i), data=data,
                                    traces=[spec['trace_idx'] for spec in self._anim_specs]))
        self.fig.frames = frames

        # Per-frame text (e.g. "Color t=0.03") lives on the slider step, not a
        # title annotation -- the subplot title stays static (set from frame 0),
        # and the slider always shows exactly which frame is current, playing or not.
        step_labels = self._anim_specs[0]['titles']
        self.fig.update_layout(
            updatemenus=[dict(
                type='buttons', showactive=False, x=0, y=0, xanchor='left', yanchor='top',
                buttons=[
                    dict(label='Play', method='animate', args=[None, dict(
                        frame=dict(duration=200, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='Pause', method='animate', args=[[None], dict(
                        frame=dict(duration=0, redraw=False), mode='immediate')]),
                ],
            )],
            sliders=[dict(
                active=0, x=0.1, y=0, len=0.9,
                steps=[dict(method='animate', label=step_labels[i], args=[[str(i)], dict(
                    mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                    for i in range(n_frames)],
            )],
        )

    def show(self):
        self._finalize()
        self.fig.show()

    def save(self, path):
        self._finalize()
        if Path(path).suffix.lower() in ('.html', '.htm'):
            self.fig.write_html(path)
        else:
            self.fig.write_image(path)  # static only -- one frame, via kaleido
