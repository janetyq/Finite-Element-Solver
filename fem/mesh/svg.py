import re
import xml.etree.ElementTree as ET

import numpy as np
import svg.path  # pyright: ignore[reportMissingImports]

from fem.geometry import calculate_polygon_area


def _document_height(root):
    '''Height of the SVG user-space box, or None if the file does not say.

    Needed to mirror the artwork: SVG's y axis points down the page, so a path
    read literally arrives upside down in any y-up frame.
    '''
    height = root.get('height')
    if height is not None:
        # Lengths may carry a unit ('737.6px'); the number is what matters here.
        number = re.match(r'\s*([0-9.eE+-]+)', height)
        if number:
            return float(number.group(1))

    view_box = root.get('viewBox')
    if view_box is not None:
        bounds = view_box.replace(',', ' ').split()
        if len(bounds) == 4:
            return float(bounds[1]) + float(bounds[3])
    return None


def read_svg_to_list_of_path_points(svg_file):
    '''
    Reads svg file and returns a list of closed loop paths, where each path is a list of points.

    Points come back in a y-up frame, mirrored about the document height, so the
    artwork plots the way it looks in a browser rather than flipped.
    '''
    # Read the SVG file and parse it
    tree = ET.parse(svg_file)
    root = tree.getroot()
    list_of_path_points = []

    # Iterate over all path elements
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        d = path.get("d")
        if d is None:
            continue
        svg_path = svg.path.parse_path(d)
        path_points = []
        for segment in svg_path:
            start = [segment.start.real, segment.start.imag]
            end = [segment.end.real, segment.end.imag]
            if isinstance(segment, svg.path.path.Move):
                assert len(path_points) == 0
                path_points.append(start)
            elif isinstance(segment, svg.path.path.Line):
                path_points.append(end)
            elif isinstance(segment, svg.path.path.Close):
                list_of_path_points.append(path_points)
                path_points = []
            elif isinstance(segment, svg.path.path.CubicBezier):
                # Approximate the cubic Bezier curve with line segments
                control1 = (segment.control1.real, segment.control1.imag)
                control2 = (segment.control2.real, segment.control2.imag)
                num_segments = 10
                for t in range(1, num_segments):
                    t_normalized = t / num_segments
                    x = (1 - t_normalized)**3 * start[0] + 3 * (1 - t_normalized)**2 * t_normalized * control1[0] + 3 * (1 - t_normalized) * t_normalized**2 * control2[0] + t_normalized**3 * end[0]
                    y = (1 - t_normalized)**3 * start[1] + 3 * (1 - t_normalized)**2 * t_normalized * control1[1] + 3 * (1 - t_normalized) * t_normalized**2 * control2[1] + t_normalized**3 * end[1]
                    path_points.append([x, y])

    if not list_of_path_points:
        return list_of_path_points

    # Fall back to the artwork's own extent when the file declares no size: the
    # mirror line only shifts the result, and shape is what callers use.
    height = _document_height(root)
    if height is None:
        height = max(y for path_points in list_of_path_points for _, y in path_points)
    return [[[x, height - y] for x, y in path_points] for path_points in list_of_path_points]

def douglas_peucker(points, epsilon):
    '''
    Simplifies a curve by reducing the number of points while preserving the overall shape.

    Input:
        points - Nx2 array of points describing the path of a curve (in order)
        epsilon - the distance from a line between two points at which a point will be kept
    
    Output:
        a nx2 array of points describing a simplified curve using the Douglas-Peucker algorithm, n < N

    The algorithm recursively keeps points that are furthest away from the line segment between pairs of points
    and stops when the furthest points are less than epsilon distance away
    '''
    if len(points) <= 2:
        return points

    def perp_distance(start, end, point):
        se_vector = start - end
        sp_vector = start - point
        # The 2D cross product written out: np.cross on 2-vectors is deprecated in
        # NumPy 2 and slated for removal.
        cross = se_vector[0]*sp_vector[1] - se_vector[1]*sp_vector[0]
        return np.abs(cross) / np.linalg.norm(se_vector)

    furthest_dist = 0
    furthest_p_idx = None
    start, end = points[0], points[-1]
    for p_idx, point in enumerate(points[1:-1], start=1):
        dist = perp_distance(start, end, point)
        if dist > furthest_dist:
            furthest_dist = dist
            furthest_p_idx = p_idx
    
    # furthest_p_idx stays None when no interior point beat furthest_dist, which
    # a collinear run does; the recursion below would slice with None+1.
    if furthest_p_idx is None or furthest_dist < epsilon:
        return np.array([start, end])
    else:
        return np.concatenate([douglas_peucker(points[:furthest_p_idx+1], epsilon)[:-1], douglas_peucker(points[furthest_p_idx:], epsilon)])


def read_svg_to_pslg(svg_file, tolerance=0.005):
    '''Read an SVG file and return a PSLG of its closed outlines.

    Each outline is simplified with Douglas-Peucker against its own bounding-box
    extent, so small features survive even when the drawing spans a wide range of
    scales.  `tolerance` is the fraction of each loop's extent below which points
    are dropped.
    '''
    loops = []
    for path_points in read_svg_to_list_of_path_points(svg_file):
        loop = np.array(path_points)
        extent = max(np.max(loop, axis=0) - np.min(loop, axis=0))
        simplified = np.asarray(douglas_peucker(loop, tolerance * extent))
        if len(simplified) >= 3:
            loops.append(simplified)
    return PSLG.from_loops(loops)


def _find_crossing_segments(vertices, segments):
    '''The first pair of segments that properly cross, or None.

    Pairs sharing an endpoint are allowed to touch there, so they are skipped.
    Compares every pair, which is quadratic in the *input* size -- the outline,
    not the mesh refined from it.
    '''
    starts, ends = vertices[segments[:, 0]], vertices[segments[:, 1]]

    def side_of(line_start, line_end, point):
        '''Sign of which side of a directed line each point falls on.'''
        return ((line_end[..., 0] - line_start[..., 0]) * (point[..., 1] - line_start[..., 1])
                - (line_end[..., 1] - line_start[..., 1]) * (point[..., 0] - line_start[..., 0]))

    rows, cols = np.asarray(starts[:, None]), np.asarray(starts[None, :])
    row_ends, col_ends = ends[:, None], ends[None, :]
    # A proper crossing puts each segment's endpoints on opposite sides of the other.
    straddles_row = (side_of(rows, row_ends, cols) > 0) != (side_of(rows, row_ends, col_ends) > 0)
    straddles_col = (side_of(cols, col_ends, rows) > 0) != (side_of(cols, col_ends, row_ends) > 0)
    crossing = straddles_row & straddles_col

    shares_endpoint = (segments[:, None, :, None] == segments[None, :, None, :]).any(axis=(2, 3))
    crossing &= ~shares_endpoint
    crossing &= np.triu(np.ones_like(crossing), k=1).astype(bool)

    found = np.argwhere(crossing)
    return tuple(found[0]) if len(found) else None


class PSLG:
    '''A planar straight-line graph: vertices, plus the segments a mesh must respect.

    `loop_ids` says which closed outline each segment came from, so a caller can
    tell an obstacle's boundary from an enclosing box's after meshing. It is all
    zeros unless the graph was built by `from_loops`.
    '''

    def __init__(self, vertices, segments=None, loop_ids=None):
        self.vertices = vertices
        if segments is None:
            self.segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
        else:
            self.segments = np.asarray(segments)
        self.loop_ids = (np.zeros(len(self.segments), dtype=int) if loop_ids is None
                         else np.asarray(loop_ids))

    @classmethod
    def from_loops(cls, loops):
        '''A PSLG spanning several closed outlines.

        What each loop *means* is decided when meshing, by the even-odd rule: a
        loop inside another is a hole, a loop beside it is a separate piece. So
        the caller draws the outlines and does not also have to label them.
        '''
        vertices, segments, loop_ids = [], [], []
        for loop_id, loop in enumerate(loops):
            points = np.asarray(loop, dtype=float)
            offset = len(vertices)
            vertices.extend(points.tolist())
            segments.extend([[offset + i, offset + (i + 1) % len(points)]
                             for i in range(len(points))])
            loop_ids.extend([loop_id] * len(points))
        return cls(np.array(vertices), np.array(segments), np.array(loop_ids))

    def area(self):
        '''Total polygon area across all loops.'''
        total = 0.0
        for loop_id in np.unique(self.loop_ids):
            verts = self.vertices[self.segments[self.loop_ids == loop_id, 0]]
            total += calculate_polygon_area(verts)
        return total

    def __repr__(self):
        return f'PSLG(vertices={self.vertices}, segments={self.segments})'

    def validate(self):
        '''Raise if these segments do not describe a planar straight-line graph.

        Segments may share endpoints and may not otherwise touch. Meshing an
        input that breaks this does not fail, it quietly produces a mesh of the
        wrong region, so it is worth refusing up front.
        '''
        vertices = np.asarray(self.vertices, dtype=float)
        segments = np.asarray(self.segments)

        starts, ends = vertices[segments[:, 0]], vertices[segments[:, 1]]
        degenerate = np.flatnonzero(np.all(starts == ends, axis=1))
        if len(degenerate):
            raise ValueError(
                f'segment {segments[degenerate[0]].tolist()} has zero length')

        _, first, counts = np.unique(vertices, axis=0, return_index=True, return_counts=True)
        if np.any(counts > 1):
            duplicate = vertices[first[counts > 1][0]]
            raise ValueError(f'vertex {duplicate.tolist()} appears more than once')

        crossing = _find_crossing_segments(vertices, segments)
        if crossing is not None:
            first_seg, second = crossing
            raise ValueError(
                f'segments {segments[first_seg].tolist()} and {segments[second].tolist()} '
                'cross away from a shared endpoint'
            )

    def add_bounding_box(self, buffer=0.2):
        '''Enclose the graph in a box, as its own loop.

        Under the even-odd rule this makes whatever was already here a hole, so
        an outline plus a box is a plate with that outline cut out of it.
        '''
        x_min, y_min = np.min(self.vertices, axis=0)
        x_max, y_max = np.max(self.vertices, axis=0)
        width = x_max - x_min
        height = y_max - y_min

        corner_vertices = [
            [x_min - buffer*width, y_min - buffer*height],
            [x_min - buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_min - buffer*height]
        ]
        num_vertices = len(self.vertices)

        box_loop = int(self.loop_ids.max()) + 1 if len(self.loop_ids) else 0
        self.vertices = np.append(self.vertices, corner_vertices, axis=0)
        for i in range(4):
            self.segments = np.append(self.segments, [[num_vertices + i, num_vertices + (i + 1) % 4]], axis=0)
            self.loop_ids = np.append(self.loop_ids, box_loop)

