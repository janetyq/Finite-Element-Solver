"""`close_ring`, the draw-time fix for a curve that reads back without a repeated
closing vertex (see tests/test_svg.py): `ax.plot` never connects a polyline's last
point back to its first on its own, so a ring plotted as given looks open there.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

from demos.outline_to_mesh.physics import close_ring  # noqa: E402


def test_close_ring_appends_the_first_point():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    closed = close_ring(points)
    assert len(closed) == len(points) + 1
    assert np.array_equal(closed[-1], points[0])
    assert np.array_equal(closed[:-1], points)
