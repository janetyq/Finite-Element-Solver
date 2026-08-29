"""Reading an SVG path into an `Outline` of pieces: `L` commands as `Line`s, `C`
commands as `CubicBezier`s with their control points kept, in a y-up frame."""
import sys
from pathlib import Path

import numpy as np
import pytest

from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.mesh.curves import CubicBezier, Line
from fem.mesh.outline import Outline, _simplify_indices, douglas_peucker
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.space import FunctionSpace

CLOUD_SVG = str(Path(__file__).resolve().parents[1] / 'files' / 'cloud.svg')


def _write_svg(tmp_path, d):
    svg_file = tmp_path / 'shape.svg'
    svg_file.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        f'<path d="{d}"/></svg>'
    )
    return str(svg_file)


def test_commands_become_pieces_in_a_y_up_frame(tmp_path):
    """A cubic keeps its four control points (mirrored about the document height) and
    a line is a `Line`; the pieces join end to end."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,1 1,1 1,0 L2,0 Z')
    (loop,) = Outline.from_svg(svg_file).loops
    assert [type(p) for p in loop] == [CubicBezier, Line, Line]
    cubic = loop[0]
    assert isinstance(cubic, CubicBezier)
    np.testing.assert_allclose(cubic.controls, [[0, 10], [0, 9], [1, 9], [1, 10]])
    np.testing.assert_allclose(loop[1].end, [2, 10])


def test_a_path_that_closes_exactly_gets_no_closing_line(tmp_path):
    """The final cubic lands on the Move point (as in `files/cloud.svg`), so `Z` adds
    nothing: no zero-length piece."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,1 1,1 1,0 C1,-1 0,-1 0,0 Z')
    (loop,) = Outline.from_svg(svg_file).loops
    assert len(loop) == 2 and all(isinstance(p, CubicBezier) for p in loop)


def test_a_path_that_does_not_close_exactly_is_closed_by_a_line(tmp_path):
    """The artwork's last point misses the Move point (as in `files/california.svg`);
    `Z` draws the straight closing edge, as the SVG spec says it does."""
    svg_file = _write_svg(tmp_path, 'M0,0 L1,0 L1,1 L0.1,0.9 Z')
    (loop,) = Outline.from_svg(svg_file).loops
    assert len(loop) == 4
    np.testing.assert_allclose(loop[-1].start, [0.1, 10 - 0.9])
    np.testing.assert_allclose(loop[-1].end, [0, 10])


def test_unsupported_commands_are_refused_not_dropped(tmp_path):
    svg_file = _write_svg(tmp_path, 'M0,0 Q1,1 2,0 L2,-1 Z')
    with pytest.raises(NotImplementedError, match='QuadraticBezier'):
        Outline.from_svg(svg_file)


def test_an_open_path_reads_as_no_outline(tmp_path):
    svg_file = _write_svg(tmp_path, 'M0,0 L1,0 L1,1')
    with pytest.raises(ValueError, match='no closed path'):
        Outline.from_svg(svg_file)


def test_a_traced_cubic_outline_meshes_onto_its_true_curve():
    """End to end: meshing `cloud.svg` and building an isoparametric space places the
    boundary midside nodes on the true cubics, where straight P2 leaves them on chords."""
    outline = Outline.from_svg(CLOUD_SVG)
    assert any(isinstance(p, CubicBezier) for loop in outline.loops for p in loop)
    graph = outline.sample(resolution=0.05)   # coarse chords, so P2 visibly misses the curve
    mesh = RuppertsAlgorithm(graph, min_angle=20, max_area=0.03 * graph.area()).refine()
    assert mesh.boundary_curves is not None

    def max_midside_distance(element_type):
        space = FunctionSpace(mesh, element_type, n_components=1)
        worst = 0.0
        for facet_nodes, curve in zip(space.boundary_nodes, mesh.boundary_curves):
            if curve is None:
                continue
            midside = space.node_coords[facet_nodes[2]]
            worst = max(worst, float(np.linalg.norm(curve.project(midside) - midside)))
        return worst

    assert max_midside_distance(IsoparametricTriangleElement) < 1e-9
    assert max_midside_distance(QuadraticTriangleElement) > 1e-3


def test_simplifying_a_trace_keeps_its_cubics(tmp_path):
    """Douglas-Peucker touches only the straight runs; a barely curved cubic survives
    simplification as the same piece, so refinement can still recover its shape."""
    svg_file = _write_svg(tmp_path, 'M0,0 C1.33,0.05 2.67,0.05 4,0 L4,-1 L4,-2 L4,-3 L0,-3 Z')
    outline = Outline.from_svg(svg_file)
    simplified = outline.simplified(0.05)
    assert sum(isinstance(p, CubicBezier) for p in simplified.loops[0]) == 1
    assert simplified.n_pieces < outline.n_pieces, 'the collinear run down the side collapsed'


# --- Douglas-Peucker ---

def test_douglas_peucker_matches_its_index_form():
    rng = np.random.default_rng(0)
    for _ in range(20):
        points = rng.uniform(0, 1, size=(30, 2))
        keep = _simplify_indices(points, 0.1)
        np.testing.assert_array_equal(douglas_peucker(points, 0.1), points[keep])
        assert keep[0] == 0 and keep[-1] == len(points) - 1
        assert keep == sorted(set(keep))


def test_douglas_peucker_is_iterative_and_survives_a_deep_split():
    t = np.linspace(0, 1, 4000)
    points = np.column_stack([t, t ** 8])
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        simplified = douglas_peucker(points, 1e-6)
    finally:
        sys.setrecursionlimit(original)
    assert 2 < len(simplified) < len(points)


def test_douglas_peucker_handles_a_collinear_run():
    """No interior point beats a zero distance on a straight run, so only the endpoints
    survive, even at epsilon 0."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert len(douglas_peucker(points, epsilon=0.0)) == 2
