"""Reading an SVG path into the point rings `PSLG` and `douglas_peucker` expect.

A loop is returned without a repeated closing vertex -- the wraparound in
`PSLG.from_loops` supplies that edge -- which the Close-segment handling in
`read_svg_to_list_of_path_points` has to preserve regardless of whether the
artwork's own last segment happens to land exactly back on its start point.
"""
import numpy as np
import pytest

from fem.mesh.svg import douglas_peucker, read_svg_to_list_of_path_points, read_svg_to_pslg


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
