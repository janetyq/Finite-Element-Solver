import pickle
import numpy as np
from utils.helper import *

from Plotter import *

class Mesh:
    def __init__(self, points, faces, boundary):
        self.points = np.array(points)
        self.faces = np.array(faces)
        self.boundary = np.array(boundary)

        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.areas = np.array([calculate_triangle_area(self.points[face]) for face in self.faces])
        self.edges = self._get_all_edges()

    # TODO: Save and load to better formats - off, obj
    @classmethod
    def load(cls, path='test_mesh.pkl'):
        with open(path, 'rb') as f:
            return cls(*pickle.load(f))

    def save(self, path='test_mesh.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_info(), f)
        print(f'Saved mesh to {path}')

    def get_info(self):
        return self.points, self.faces, self.boundary

    def __repr__(self):
        return f'Mesh(points={self.points}, faces={self.faces}, boundary={self.boundary})'

    def copy(self):
        return Mesh(self.points.copy(), self.faces.copy(), self.boundary.copy())

    def plot(self):
        return Plotter(self).plot_mesh()

    # METRICS
    def calculate_total_value(self, u):
        if len(u) == len(self.faces):       # u defined on faces
            return sum([self.areas[face_idx] * u[face_idx] for face_idx in range(len(self.faces))])
        elif len(u) == len(self.points):    # u defined on vertices
            return sum([self.areas[face_idx] * np.mean(u[self.faces[face_idx]]) for face_idx in range(len(self.faces))])

    def calculate_mean_value(self, u):
        return self.calculate_total_value(u) / sum(self.areas)

    def calculate_dirichlet_energy(self, u):
        energy = 0
        # u_gradient = self.calculate_gradient(u)
        # for face_idx, face in enumerate(self.faces):
        #     area = calculate_triangle_area(self.points[face])
        #     energy += 1/2 * area * calc_dot(u_gradient[face_idx], u_gradient[face_idx])
        return energy

    def calculate_energy(self, u, dudt):
        # TODO: not conserved in wave eqn for some reason
        dirichlet_energy = self.calculate_dirichlet_energy(u)
        kinetic_energy = 1/2 * self.calculate_total_value(dudt**2)
        return dirichlet_energy + kinetic_energy
        
    def calculate_face_neighbors(self):
        self.face_neighbors = {face_idx: [] for face_idx in range(len(self.faces))}
        for face_idx, face in enumerate(self.faces):
            for neighbor_idx in face:
                self.face_neighbors[face_idx].extend(np.where(self.faces == neighbor_idx)[0])
            # keep only elements that appear twice
            self.face_neighbors[face_idx] = [idx for idx in self.face_neighbors[face_idx] if self.face_neighbors[face_idx].count(idx) == 2]
        return self.face_neighbors

    def _get_all_edges(self):
        all_edges = set()
        for face in self.faces:
            for i in range(3):
                edge = [face[i], face[(i+1)%3]]
                all_edges.add(tuple(sorted(edge)))
        all_edges = np.array(list(all_edges))
        return all_edges

    def get_edges_in_idxs(self, vertices_idxs, exclude_corners=False):
        in_edges = []
        for edge in self.edges:
            if edge[0] in vertices_idxs and edge[1] in vertices_idxs:
                if exclude_corners:
                    x1, y1 = self.points[edge[0]]
                    x2, y2 = self.points[edge[1]]
                    if (x1 - x2) != 0 and (y1 - y2) != 0:
                        continue
                in_edges.append(edge)
        return in_edges

    def get_boundary_idxs_in_rect(self, rect):
        x_min, y_min, x_max, y_max = rect
        in_boundary_idxs = []
        for idx in self.boundary_idxs:
            x, y = self.points[idx]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_boundary_idxs.append(idx)
        return in_boundary_idxs

class EasyMesher:
    '''
    Creates an approximate uniform mesh of a given outline
    '''
    def __init__(self, outline, approx_triangles=100):
        self.outline = outline
        self.dx = np.sqrt(2 * calculate_polygon_area(outline) / approx_triangles)

    def mesh(self, plot=True):
        x_min, x_max = np.min(self.outline[:, 0]), np.max(self.outline[:, 0])
        y_min, y_max = np.min(self.outline[:, 1]), np.max(self.outline[:, 1])
        x_range = np.arange(x_min, x_max, self.dx)
        y_range = np.arange(y_min, y_max, self.dx)
        x_range += (x_max - x_range[-1])/2
        y_range += (y_max - y_range[-1])/2
        points = np.array([[x, y] for y in y_range for x in x_range])
        
        faces = []

        def get_index(i, j):
            return j*len(x_range) + i

        # first mesh everything
        for i, x in enumerate(x_range[:-1]):
            for j, y in enumerate(y_range[:-1]):
                faces.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                faces.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
                
        # second remove faces with centers outside of outline
        removed_faces = []
        for face in faces:
            center = np.mean(points[face], axis=0)
            offcenters = [(center + points[i])/2 for i in face]
            for offcenter in offcenters:
                if not point_in_polygon(offcenter, self.outline):
                    removed_faces.append(face)
                    break
        for face in removed_faces:
            faces.remove(face)

        # remove unnecessary points
        used_point_idxs = np.unique(np.array(faces).flatten())
        # map old indices to new indices
        idx_map = {old: new for new, old in enumerate(used_point_idxs)}
        points = points[used_point_idxs]
        faces = [[idx_map[idx] for idx in face] for face in faces]
        boundary = get_boundary_from_points_faces(points, faces)
        mesh = Mesh(points, faces, boundary)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.outline[:, 0], self.outline[:, 1], 'r-')
            Plotter(mesh, fig=fig, ax=ax, options={'title': 'EasyMesher mesh'}).plot_mesh(mode='wireframe')
        
        return mesh

class RectMesher:
    def __init__(self, corners, resolution):
        self.corners = corners
        self.resolution = resolution

    def mesh(self, plot=True):
        x_range = np.linspace(self.corners[0][0], self.corners[1][0], self.resolution[0])
        y_range = np.linspace(self.corners[0][1], self.corners[1][1], self.resolution[1])

        points = np.array([[x, y] for y in y_range for x in x_range])

        faces = []

        def get_index(i, j):
            return j*self.resolution[0] + i

        for i in range(self.resolution[0]-1):
            for j in range(self.resolution[1]-1):
                faces.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                faces.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])

        boundary = get_boundary_from_points_faces(points, faces)
        mesh = Mesh(points, faces, boundary)

        if plot:
            Plotter(mesh, options={'title': 'RectMesher mesh'}).plot_mesh(mode='wireframe')

        return Mesh(points, faces, boundary)

if __name__ == '__main__':
    # EasyMesher for a simple mesh of any polygon
    outline = np.array([[0, 0], [1, 2], [3, 2], [2, 0], [0, 0]])
    mesher = EasyMesher(outline, approx_triangles=500)
    mesh = mesher.mesh()
    mesh.save('meshes/easymesher_demo.pkl')

    # RectMesher to make a uniform rectangular mesh
    corners = [[0, 0], [2, 1]]
    resolution = (40, 20)
    mesher = RectMesher(corners, resolution=resolution)
    mesh = mesher.mesh()
    mesh.save(f'meshes/{resolution[0]}x{resolution[1]}.pkl')

    # Mesh plotting examples with color
    face_list = []
    for face_idx, face in enumerate(mesh.faces):
        center = np.mean(mesh.points[face], axis=0)
        if center[0] > corners[1][0]/2:
            face_list.append(face_idx)
    color_faces = [('blue', face_list, 'right blue faces')]
    vert_list = []
    for vert_idx, vert in enumerate(mesh.points):
        if vert[0] < 1e-3:
            vert_list.append(vert_idx)
    color_vertices = [('red', vert_list, 'left red vertices')]
    Plotter(mesh, options={'title': 'Labeled mesh plot'}).plot_mesh(mode='wireframe', color_faces=color_faces, color_vertices=color_vertices)

    values = []
    for face_idx, face in enumerate(mesh.faces):
        x, y = np.mean(mesh.points[face], axis=0)
        value = abs(((y-0.5) - 0.3*np.sin(10*x)))
        values.append(value)
    Plotter(mesh, options={'title': 'Colorful value plot'}).plot_values(values, mode='colored')

    # Neighbors plotting example
    face_neighbors = mesh.calculate_face_neighbors()
    fig, ax = plt.subplots()
    for face_idx, neighbors in face_neighbors.items():
        if face_idx % 2 == 0 or face_idx % 3 == 0:
            continue
        center = np.mean(mesh.points[mesh.faces[face_idx]], axis=0)
        random_color = np.random.rand(3)
        for neighbor_idx in neighbors:
            neighbor_center = np.mean(mesh.points[mesh.faces[neighbor_idx]], axis=0)
            ax.plot([center[0], neighbor_center[0]], [center[1], neighbor_center[1]], color=random_color)
    Plotter(mesh, fig=fig, ax=ax, options={'title': 'Face neighbors'}).plot_mesh(mode='wireframe')
    