"""Meshing demos: turning a shape into a triangulation, and naming parts of it.

Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_from_svg
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

from fem.geometry import calculate_triangle_min_angle
from fem.plot.plotter import Plotter
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker
from fem.regions import in_box, intersect, on_plane

from demo_registry import Demo, DemoResult, Figure
from domains import beam, plate_with_hole_pslg

# Resolved against the repo rather than the working directory: the input files ship
# with the project, so a demo should not depend on where it was launched from. Output
# paths stay relative, and so follow the caller's directory.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count, so simplifying
# first is what keeps the demo interactive.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

# Ruppert's: any triangle whose area exceeds this fraction of the outline's total area
# gets its circumcenter inserted. The angle bound controls element *shape* but says
# nothing about size, so without this a large region comes back as a handful of
# enormous triangles.
DEFAULT_MAX_AREA_FRACTION = 0.005

def demo_regions(mesh):
    """Name parts of a domain by position, which is how a boundary condition says where
    it applies -- and works the same for vertices and for elements."""
    # The alternative is naming vertex indices, and an index means nothing after a
    # remesh renumbers them. Everything here is written against coordinates, so the
    # same three lines select the same three places on any mesh of this beam -- which
    # is what lets a generated mesh carry boundary conditions at all.
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    clamped = on_plane(0, 0.0)
    loaded = intersect(on_plane(0, w), in_box([None, 0.2*h], [None, 0.8*h]))
    far_half = in_box([w/2, None], [None, None])

    # Sized for the domain plus a row of labels under it: the axes are equal-aspect, so
    # a 4:1 beam in the default square-ish figure is a thin strip, and a legend inside
    # one covers the mesh it is annotating.
    figsize = (9.0, 3.2)

    boundary = Plotter(title='The mesh and its boundary', axis_labels=False,
                       figsize=figsize)
    boundary.plot(mesh, mode='mesh')
    boundary.plot(mesh, mode='boundary')
    boundary.plot_highlights(mesh, [mesh.boundary_idxs], ['red'],
                             [f'boundary ({len(mesh.boundary_idxs)} vertices)'])
    boundary.get_ax().legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                             frameon=False)

    # A region is a callable from coordinates to a mask, so it reads element centroids
    # as happily as it reads vertices.
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    selected = Plotter(title='Three regions, selected by position', axis_labels=False,
                       figsize=figsize)
    selected.plot(mesh, mode='mesh')
    selected.plot_highlights(mesh, [np.flatnonzero(far_half(centroids))], ['lightblue'],
                             ['in_box: far half'], mode='elements')
    selected.plot_highlights(
        mesh,
        [np.flatnonzero(clamped(mesh.vertices)), np.flatnonzero(loaded(mesh.vertices))],
        ['red', 'green'],
        ['on_plane: clamped edge', 'intersect: loaded patch'],
    )
    selected.get_ax().legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
                             frameon=False)
    return DemoResult([
        Figure(boundary,
               'The boundary a mesh knows about: the facets it carries, and the vertices '
               'on them. Every condition in the gallery is placed somewhere on this.',
               'boundary'),
        Figure(selected,
               'The same three regions the cantilever demos use, drawn rather than solved '
               'on. Note the regions are geometric, not boundary-aware: resolving one into '
               'a boundary condition intersects it with the boundary, so a plane through '
               'the middle of the domain yields only the two vertices where it emerges.',
               'regions'),
    ])

def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)

def simplify_curve(curve, save_file='douglas_peucker_output.json',
                   tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE, interactive=False):
    """Simplify `curve` with Douglas-Peucker, returning the simplified curve.

    `tolerance` is a fraction of the curve's extent. `interactive=True` opens a slider
    to explore it instead, starting from `tolerance`, and returns whatever it was left
    on. Simplifying directly is the default so that every caller -- including ones with
    nobody watching -- gets the same result without having to ask for it.
    """
    d = max(np.max(curve, axis=0) - np.min(curve, axis=0))
    if not interactive:
        return douglas_peucker(curve, tolerance * d)

    fig, ax = plt.subplots()  # a widget figure, not a Plotter: this path is interactive
    ax.plot(curve[:, 0], curve[:, 1], color='gray', alpha=0.5)
    plt.subplots_adjust(bottom=0.15)

    initial = douglas_peucker(curve, tolerance * d)
    sampled_plot = plt.plot(initial[:, 0], initial[:, 1], 'b-')[0]
    # Starting at zero would leave an untouched slider handing the full outline
    # downstream, which Ruppert's does not finish triangulating.
    slider = Slider(plt.axes([0.15, 0.04, 0.6, 0.03]), 'Epsilon ', 0, d/20,
                    valinit=tolerance * d)
    button = plt.Button(plt.axes([0.85, 0.04, 0.1, 0.04]), 'Save')

    def update(val):
        epsilon = slider.val
        dp = douglas_peucker(curve, epsilon)
        sampled_plot.set_xdata(dp[:, 0])
        sampled_plot.set_ydata(dp[:, 1])
        fig.canvas.draw_idle()

    def save(event):
        epsilon = slider.val
        sampled_curve = douglas_peucker(curve, epsilon)
        with open(save_file, 'w') as f:
            json.dump(sampled_curve.tolist(), f)
        print(f'Saved points to {save_file}')

    slider.on_changed(update)
    button.on_clicked(save)
    ax.set_aspect('equal')
    plt.show()

    return douglas_peucker(curve, slider.val)

def rupperts_mesh(pslg, min_angle=20, max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Triangulate a PSLG with Ruppert's algorithm; returns (mesh, algorithm)."""
    pslg.validate()
    max_area = None
    if max_area_fraction is not None:
        max_area = max_area_fraction * pslg.area()
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle, max_area=max_area)
    return rupperts.refine(), rupperts

def rupperts_figure(mesh, rupperts, min_angle, slug=''):
    """The triangulation, its input segments, and what the angle bound actually held to."""
    plotter = Plotter(title='Triangulated mesh', axis_labels=False)
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    # One collection rather than a plot call per segment: an outline that has been
    # refined runs to hundreds of them.
    ax.add_collection(LineCollection(rupperts.vertices[rupperts.segments],
                                     colors='blue', linewidths=1.0))
    outlines = len(np.unique(rupperts.segment_loops))
    # An input corner sharper than the bound keeps its own angle, so read the
    # bound off the mesh rather than claiming the one that was asked for.
    worst = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
    held = (f'every angle at least {min_angle} degrees' if worst >= min_angle else
            f'every angle at least {min_angle} degrees bar the input corners already '
            f'sharper than that, the worst {worst:.0f}')
    return Figure(
        plotter,
        f"Ruppert's refinement of {outlines} outlines (blue) into {len(mesh.elements)} "
        f'triangles, {held}. The mesh covers what the '
        f'outlines enclose and nothing else, and carries the {len(mesh.boundary)} boundary '
        'edges a solver needs to put conditions on.',
        slug)

def demo_plate_with_hole(min_angle=25, max_area_fraction=0.004):
    """Mesh a plate with a hole in it, colouring the boundary by which outline it came from.

    The shape a flow-around-an-obstacle problem needs: one loop inside another is
    a hole under the even-odd rule, and the two boundaries have to be separable
    for the obstacle and the outer wall to take different conditions."""
    # The same geometry `stress_concentration` solves on, from one definition: this
    # demo builds the mesh, that one puts conditions on the boundaries it separates.
    pslg = plate_with_hole_pslg()
    mesh, rupperts = rupperts_mesh(pslg, min_angle=min_angle,
                                   max_area_fraction=max_area_fraction)

    plotter = Plotter(title='Plate with a hole', axis_labels=False, panel_aspect=2.0)
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    for loop_id, colour, label in ((0, 'blue', 'outer wall'), (1, 'red', 'obstacle')):
        facets = np.asarray(mesh.boundary)[rupperts.boundary_loops == loop_id]
        ax.add_collection(LineCollection(mesh.vertices[facets], colors=colour,
                                         linewidths=2.0))
        ax.plot([], [], color=colour, linewidth=2.0, label=f'{label} ({len(facets)} edges)')
    ax.legend(loc='upper right')
    return DemoResult([Figure(
        plotter,
        f'{len(mesh.elements)} triangles between the two outlines. The hole is absent from '
        'the mesh but present in its boundary, and every boundary edge knows which outline '
        'it came from -- which is what lets Dirichlet on the obstacle and Neumann on the '
        'wall be written separately.')])

def demo_mesh_from_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE,
                       interactive=False, min_angle=20,
                       max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Turn an SVG drawing into a mesh: simplify each outline with Douglas-Peucker, then
    triangulate them with Ruppert's algorithm.

    --interactive opens a slider over the simplification tolerance, previewing it on the
    largest outline before meshing. The tolerance used for meshing is always `tolerance`,
    applied per-loop by `read_svg_to_pslg`."""
    # The two steps are one demo because the first exists for the second: Ruppert's cost
    # is superlinear in the point count it is handed, and an SVG outline traced at screen
    # resolution has thousands. Simplification is what makes the triangulation finish.
    curve = get_curve_from_svg(svg_file)
    simplified = simplify_curve(curve, tolerance=tolerance, interactive=interactive)

    # The interactive path has already had its say on screen; this is the result either
    # way, and the only thing a saved gallery can show of it.
    simplify_plotter = Plotter(title='Douglas-Peucker simplification', axis_labels=False)
    ax = simplify_plotter.get_ax()
    ax.plot(curve[:, 0], curve[:, 1], color='gray', linewidth=1.0,
            label=f'original ({len(curve)} pts)')
    ax.plot(simplified[:, 0], simplified[:, 1], 'b-',
            label=f'simplified ({len(simplified)} pts)')

    pslg = read_svg_to_pslg(svg_file, tolerance=tolerance)
    mesh, rupperts = rupperts_mesh(pslg, min_angle=min_angle,
                                   max_area_fraction=max_area_fraction)
    return DemoResult([
        Figure(simplify_plotter,
               f'{len(curve)} outline points reduced to {len(simplified)} on the largest '
               'loop, at a tolerance set as a fraction of the outline\'s extent.',
               'simplified'),
        rupperts_figure(mesh, rupperts, min_angle, 'meshed'),
    ])


DEMOS = [
    # Both mesh to a size cap, which is what makes the figures worth looking at and
    # also most of their cost; the smoke run only needs the code paths. Loosen the cap
    # and nothing else -- simplifying the outline further is *not* reliably cheaper,
    # because it sharpens corners, and refinement spends extra elements around those.
    Demo('mesh_from_svg', demo_mesh_from_svg, section='Meshing a domain',
         smoke_kwargs={'max_area_fraction': 0.05}),
    Demo('plate_with_hole', demo_plate_with_hole, section='Meshing a domain',
         smoke_kwargs={'max_area_fraction': 0.05}),
    # Coarse, so individual edges and the selected vertices stay legible, and a beam
    # so the regions are the cantilever's own.
    Demo('regions', demo_regions, section='Meshing a domain',
         domain=partial(beam, 4.0, 1.0, 24)),
]
