"""Reading an SVG path into the point rings `PSLG` and `douglas_peucker` expect.

A loop is returned without a repeated closing vertex -- the wraparound in
`PSLG.from_loops` supplies that edge -- which the Close-segment handling in
`read_svg_to_list_of_path_points` has to preserve regardless of whether the
artwork's own last segment happens to land exactly back on its start point.
"""
from pathlib import Path

import numpy as np
import pytest

import sys

from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.mesh.curves import CubicBezier
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import (
    _find_crossing_segments,
    _simplify_indices,
    douglas_peucker,
    read_svg_to_list_of_path_points,
    read_svg_to_pslg,
)
from fem.space import FunctionSpace

CLOUD_SVG = str(Path(__file__).resolve().parents[1] / 'files' / 'cloud.svg')


def _write_svg(tmp_path, d):
    svg_file = tmp_path / 'shape.svg'
    svg_file.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        f'<path d="{d}"/></svg>'
    )
    return str(svg_file)


def test_a_bezier_segment_is_sampled_through_its_own_true_endpoint(tmp_path):
    """Every curve segment's endpoint should appear in the outline, not just the
    points strictly between it and its neighbour -- otherwise every joint in a
    traced SVG (not only the one closing the loop) cuts the corner short by
    however far the last sample point falls short of it."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,1 1,1 1,0 L2,0 Z')
    loop = np.array(read_svg_to_list_of_path_points(svg_file)[0])
    true_endpoint = np.array([1, 10 - 0])  # the C command's own end, y-mirrored
    assert np.min(np.linalg.norm(loop - true_endpoint, axis=1)) < 1e-9


def test_a_curve_that_closes_exactly_is_returned_without_a_duplicate_vertex(tmp_path):
    """The final Bezier's endpoint is written to coincide with the Move point, as a
    real closed-artwork path does -- e.g. `files/cloud.svg`."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,1 1,1 1,0 C1,-1 0,-1 0,0 Z')
    loop = np.array(read_svg_to_list_of_path_points(svg_file)[0])
    assert not np.allclose(loop[0], loop[-1])


def test_a_curve_that_does_not_close_exactly_keeps_its_real_gap(tmp_path):
    """The artwork's own last point does not land back on the Move point -- e.g.
    `files/california.svg`. Nothing should be fabricated to close it; the gap is
    closed implicitly by `PSLG.from_loops`'s wraparound instead."""
    svg_file = _write_svg(tmp_path, 'M0,0 L1,0 L1,1 L0.1,0.9 Z')
    loop = np.array(read_svg_to_list_of_path_points(svg_file)[0])
    assert loop[-1].tolist() == pytest.approx([0.1, 10 - 0.9])
    assert not np.allclose(loop[0], loop[-1])


def test_a_closed_curve_simplifies_and_meshes_without_a_degenerate_chord(tmp_path):
    """Regression: sampling every Bezier through its true endpoint (rather than
    stopping short) once left an exact start==end duplicate in the ring, which is
    a zero-length chord to `douglas_peucker` and a zero-length edge to `PSLG`."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,1 1,1 1,0 C1,-1 0,-1 0,0 Z')
    loop = np.array(read_svg_to_list_of_path_points(svg_file)[0])

    with np.errstate(invalid='raise'):
        simplified = douglas_peucker(loop, 0.01)
    assert not np.isnan(simplified).any()

    pslg = read_svg_to_pslg(svg_file, tolerance=0.01)
    pslg.validate()  # raises on a zero-length or duplicated vertex


def test_bezier_segments_keep_a_cubic_curve_and_lines_stay_straight(tmp_path):
    """A cubic path segment tags its segments with a `CubicBezier`; a `Line` tags `None`,
    so meshing rounds the cubic and leaves the straight edge alone."""
    svg_file = _write_svg(tmp_path, 'M0,0 C0,4 4,4 4,0 L4,-3 L0,-3 Z')
    pslg = read_svg_to_pslg(svg_file, tolerance=0.01)

    curved = [c for c in pslg.segment_curves if isinstance(c, CubicBezier)]
    assert curved, 'the cubic contributed no curved segments'
    # Every curved segment's endpoints lie on its own cubic (they were sampled from it).
    for seg, curve in zip(pslg.segments, pslg.segment_curves):
        if curve is None:
            continue
        ends = pslg.vertices[seg]
        assert np.max(np.linalg.norm(curve.project(ends) - ends, axis=1)) < 1e-9
    # The two straight edges (and the closing edge) carry no curve.
    assert sum(c is None for c in pslg.segment_curves) >= 2


def test_a_nearly_flat_cubic_collapses_to_a_chord_but_keeps_its_curve(tmp_path):
    """Douglas-Peucker drops the samples of a barely-curved cubic, yet the surviving
    chord keeps the `CubicBezier`, so refinement can still recover the true shape."""
    # Sagitta ~0.05 across a span of 4: well under the simplification tolerance.
    svg_file = _write_svg(tmp_path, 'M0,0 C1.33,0.05 2.67,0.05 4,0 L4,-3 L0,-3 Z')
    pslg = read_svg_to_pslg(svg_file, tolerance=0.05)

    curved = [c for c in pslg.segment_curves if isinstance(c, CubicBezier)]
    assert len(curved) == 1, 'the flat cubic should collapse to a single curved chord'


def test_a_traced_cubic_outline_meshes_onto_its_true_curve():
    """End to end: meshing `cloud.svg` and building an isoparametric space places the
    boundary midside nodes on the true cubics, where straight P2 leaves them on chords."""
    pslg = read_svg_to_pslg(CLOUD_SVG, tolerance=0.01)
    assert any(isinstance(c, CubicBezier) for c in pslg.segment_curves)
    mesh = RuppertsAlgorithm(pslg, min_angle=20, max_area=0.03 * pslg.area()).refine()
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
    assert max_midside_distance(QuadraticTriangleElement) > 1e-2


def test_douglas_peucker_matches_its_index_form():
    """`douglas_peucker` is a thin wrapper over `_simplify_indices`, so it returns
    exactly the kept points, in order."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        points = np.cumsum(rng.standard_normal((rng.integers(3, 60), 2)), axis=0)
        eps = float(rng.uniform(0.05, 2.0))
        keep = _simplify_indices(points, eps)
        assert np.array_equal(douglas_peucker(points, eps), points[keep])
        assert keep[0] == 0 and keep[-1] == len(points) - 1   # endpoints always kept
        assert keep == sorted(set(keep))                      # sorted, no duplicates


def test_douglas_peucker_is_iterative_and_survives_a_deep_split():
    """The simplifier uses an explicit stack, so a densely sampled outline whose splits
    nest thousands deep cannot overflow the interpreter's recursion limit."""
    # A curve of increasing curvature forces very unbalanced splits, the recursion's
    # worst case; 4000 points would blow a recursion-limited implementation.
    t = np.linspace(0, 1, 4000)
    points = np.column_stack([t, t ** 8])
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        simplified = douglas_peucker(points, 1e-9)
    finally:
        sys.setrecursionlimit(original)
    assert 2 < len(simplified) < len(points)


def _crossing_by_all_pairs(vertices, segments):
    '''The reference the grid must match: the old quadratic all-pairs scan, returning
    the lexicographically first properly crossing pair.'''
    vertices, segments = np.asarray(vertices, float), np.asarray(segments)

    def side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if set(segments[i]) & set(segments[j]):
                continue
            ai, bi = vertices[segments[i]]
            aj, bj = vertices[segments[j]]
            if ((side(ai, bi, aj) > 0) != (side(ai, bi, bj) > 0)
                    and (side(aj, bj, ai) > 0) != (side(aj, bj, bi) > 0)):
                return (i, j)
    return None


def test_grid_crossing_detection_matches_the_all_pairs_reference():
    """The spatial-grid crossing search is a speed knob only: on random segment soups
    it returns exactly the pair the old all-pairs scan would."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(2, 25))
        vertices = rng.uniform(0, 10, size=(2 * n, 2))
        segments = np.arange(2 * n).reshape(n, 2)
        expected = _crossing_by_all_pairs(vertices, segments)
        got = _find_crossing_segments(vertices, segments)
        assert got == expected


def test_grid_crossing_finds_a_crossing_among_far_apart_clusters():
    """A long spanning segment plus a distant crossing pair: the grid must still find
    the crossing even though the two clusters share no small cell."""
    vertices = np.array([
        [-100.0, 0.0], [100.0, 0.001],   # a long, near-horizontal spanning segment
        [0.0, -1.0], [0.0, 1.0],         # a short vertical segment that it crosses
        [50.0, 50.0], [51.0, 50.0],      # a far-off pair of parallel, non-crossing segments
        [50.0, 51.0], [51.0, 51.0],
    ])
    segments = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert _find_crossing_segments(vertices, segments) == (0, 1)
