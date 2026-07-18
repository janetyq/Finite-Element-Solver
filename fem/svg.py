import numpy as np

import svg.path
import xml.etree.ElementTree as ET

def read_svg_to_list_of_path_points(svg_file):
    '''
    Reads svg file and returns a list of closed loop paths, where each path is a list of points.
    '''
    # Read the SVG file and parse it
    tree = ET.parse(svg_file)
    root = tree.getroot()
    list_of_path_points = []

    # Iterate over all path elements
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        d = path.get("d")
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
            
    return list_of_path_points

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
        return np.abs(np.cross(se_vector, sp_vector)) / np.linalg.norm(se_vector)

    furthest_dist = 0
    furthest_p_idx = None
    start, end = points[0], points[-1]
    for p_idx, point in enumerate(points[1:-1], start=1):
        dist = perp_distance(start, end, point)
        if dist > furthest_dist:
            furthest_dist = dist
            furthest_p_idx = p_idx
    
    if furthest_dist < epsilon:
        return [start, end]
    else:
        return np.concatenate([douglas_peucker(points[:furthest_p_idx+1], epsilon)[:-1], douglas_peucker(points[furthest_p_idx:], epsilon)])


class PSLG:
    def __init__(self, vertices, segments=None):
        self.vertices = vertices
        if segments is None:
            self.segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
        else:
            self.segments = segments
    
    def __repr__(self):
        return f'PSLG(vertices={self.vertices}, segments={self.segments})'

    def add_bounding_box(self, buffer=0.2):
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

        self.vertices = np.append(self.vertices, corner_vertices, axis=0)
        for i in range(4):
            self.segments = np.append(self.segments, [[num_vertices + i, num_vertices + (i + 1) % 4]], axis=0)

