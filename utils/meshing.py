import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import Delaunay

sys.path.append('..')
from Mesh import Mesh
from utils.helper import *


class RuppertsAlgorithm:
    def __init__(self, pslg, min_angle=30):
        self.vertices = np.array(pslg.vertices)
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.triangulation = Delaunay(self.vertices)
        self.min_angle = min_angle

    def get_encroached_segments(self):
        encroached_segments = []
        for segment in self.segments:
            if self.is_segment_encroached(segment):
                encroached_segments.append(segment)
        return encroached_segments

    def get_bad_triangles(self):
        bad_triangles = []
        for triangle in self.triangulation.simplices:
            min_angle = calculate_triangle_min_angle(self.vertices[triangle])
            if min_angle < self.min_angle:
                bad_triangles.append(triangle)
        return bad_triangles

    def run_algo(self):
        encroached_segments = self.get_encroached_segments()
        bad_triangles = self.get_bad_triangles()

        while True:
            new_encroached_segments = []

            # check if there are any encroached segments and split them
            if len(encroached_segments) > 0:
                segment = encroached_segments.pop()
                self.split_segment(segment)
            # check if there are any bad triangles and refine them
            elif len(bad_triangles) > 0:
                triangle = bad_triangles.pop()
                circumcenter = calculate_circumcenter(self.vertices[triangle])
                vertex_encroaches = False
                for segment in self.segments:
                    if self.is_segment_encroached(segment, circumcenter):
                        new_encroached_segments.append(segment)
                        vertex_encroaches = True
                if not vertex_encroaches:
                    self.add_vertex(circumcenter)
            # if no encroached segments or bad triangles, we are done
            else:
                break
            self.triangulation = Delaunay(self.vertices)
            encroached_segments = self.get_encroached_segments() + new_encroached_segments
            bad_triangles = self.get_bad_triangles()

        return Mesh(self.vertices, self.triangulation.simplices, [])

    def is_segment_encroached(self, segment, vertex=None):
        center = 0.5 * (self.vertices[segment[0]] + self.vertices[segment[1]])
        radius = np.linalg.norm(self.vertices[segment[0]] - self.vertices[segment[1]]) / 2
        if vertex is not None:
            vec = vertex - center
            if vec[0]**2 + vec[1]**2 < radius ** 2 - 1e-6:
                return True
        for vertex in self.vertices:
            vec = vertex - center
            if vec[0]**2 + vec[1]**2 < radius ** 2 - 1e-6:
                return True

    def del_segment(self, segment):
        segment_idx = np.where((self.segments == segment).all(axis=1))[0][0]
        self.segments = np.delete(self.segments, segment_idx, axis=0)
    
    def add_vertex(self, vertex):
        self.vertices = np.append(self.vertices, [vertex], axis=0)
    
    def add_segment(self, segment):
        self.segments = np.append(self.segments, [segment], axis=0)

    def split_segment(self, segment):
        midpoint = 0.5 * (self.vertices[segment[0]] + self.vertices[segment[1]])
        new_vertex_idx = len(self.vertices)
        new_segments = [[segment[0], new_vertex_idx], [segment[1], new_vertex_idx]]
        self.del_segment(segment)
        self.add_vertex(midpoint)
        self.add_segment(new_segments[0])
        self.add_segment(new_segments[1])
        return new_segments

# Simple meshing functions
def create_rect_mesh(corners, resolution):
    x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
    y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*resolution[0] + i

    for i in range(resolution[0]-1):
        for j in range(resolution[1]-1):
            if (i + j) % 2 == 0:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            else:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i, j+1)])
                elements.append([get_index(i+1, j), get_index(i+1, j+1), get_index(i, j+1)])

    boundary = get_boundary_from_vertices_elements(vertices, elements)
    mesh = Mesh(vertices, elements, boundary)

    return mesh

def create_approx_mesh(outline, approx_triangles=100):
    dx = np.sqrt(2 * calculate_polygon_area(outline) / approx_triangles)
    x_min, x_max = np.min(outline[:, 0]), np.max(outline[:, 0])
    y_min, y_max = np.min(outline[:, 1]), np.max(outline[:, 1])
    x_range = np.arange(x_min, x_max, dx)
    y_range = np.arange(y_min, y_max, dx)
    x_range += (x_max - x_range[-1])/2
    y_range += (y_max - y_range[-1])/2

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*len(x_range) + i

    # first mesh everything
    for i, x in enumerate(x_range[:-1]):
        for j, y in enumerate(y_range[:-1]):
            elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
            elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            
    # second remove elements with centers outside of outline
    removed_elements = []
    for element in elements:
        center = np.mean(vertices[element], axis=0)
        offcenters = [(center + vertices[i])/2 for i in element]
        for offcenter in offcenters:
            if not point_in_polygon(offcenter, outline):
                removed_elements.append(element)
                break
    for element in removed_elements:
        elements.remove(element)

    # remove unnecessary vertices
    used_v_idxs = np.unique(np.array(elements).flatten())
    # map old indices to new indices
    v_idx_map = {old: new for new, old in enumerate(used_v_idxs)}
    vertices = vertices[used_v_idxs]
    elements = [[v_idx_map[e_idx] for e_idx in element] for element in elements]
    boundary = get_boundary_from_vertices_elements(vertices, elements)
    mesh = Mesh(vertices, elements, boundary)

    plotter = Plotter(title='Approximate mesh')
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    ax.plot(outline[:, 0], outline[:, 1], 'r-')
    plotter.show()

    return mesh