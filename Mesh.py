import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from utils.helper import *
from matplotlib.colorbar import Colorbar


class Mesh:
    def __init__(self, points, faces, boundary):
        self.points = np.array(points)
        self.faces = np.array(faces)
        self.boundary = np.array(boundary)

        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.areas = np.array([calculate_triangle_area(self.points[face]) for face in self.faces])
        self.edges = self._get_all_edges()

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

    def get_deformed_mesh(self, u):
        points = self.points + u.reshape(-1, 2)
        return Mesh(points, self.faces, self.boundary)

    # PLOTTING
    def plot(self, title=None, ax=None, show=True, linewidth=0.2, mode=None, fill_regions=None, save=None, idxs=None):
        if ax is None:
            fig, ax = plt.subplots()

        if mode is None or mode == "wireframe":
            # triangulation and boundary
            ax.triplot(self.points[:, 0], self.points[:, 1], self.faces, color='black', linewidth=linewidth)
            for seg in self.boundary:
                ax.plot(self.points[seg, 0], self.points[seg, 1], color='black', linewidth=linewidth*2)
                # for testing normals
                # vec = self.points[seg[1]] - self.points[seg[0]]
                # point = np.mean(self.points[seg], axis=0) + 0.5 * np.array([-vec[1], vec[0]])            
                # ax.scatter(point[0], point[1], color='green', s=5)
        elif mode == "boundary":
            for seg in self.boundary:
                ax.plot(self.points[seg, 0], self.points[seg, 1], color='gray', linewidth=linewidth*2)
        elif mode == "solid":
            # plot mesh
            for face in self.faces:
                vertices = self.points[face]
                center = np.mean(vertices, axis=0)
                vertices = center + 0.95 * (vertices - center)
                ax.fill(vertices[:, 0], vertices[:, 1], 'b-', alpha=0.2)
            for edge in self.boundary:
                ax.plot(self.points[[edge[0], edge[1]], 0], self.points[[edge[0], edge[1]], 1], 'k-')

        if fill_regions is not None:      
            for color, region, label in fill_regions:
                # region indexes into faces
                i = 0
                for face_idx in region:
                    vertices = self.points[self.faces[face_idx]]
                    ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.2, label=label if i == 0 else None)
                    i += 1

        if idxs:
            ax.scatter(self.points[idxs, 0], self.points[idxs, 1], color='red', s=5)

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.ticklabel_format(useOffset=False)

        if save:
            plt.savefig(save)
            plt.close()
        elif show:
            plt.show()
        return ax

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
        u_gradient = self.calculate_gradient(u)
        for face_idx, face in enumerate(self.faces):
            area = calculate_triangle_area(self.points[face])
            energy += 1/2 * area * calc_dot(u_gradient[face_idx], u_gradient[face_idx])
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

    def save_to_obj(self, filename):
        with open(filename, 'w') as f:
            for vertex in self.points:
                f.write("v {} {} {}\n".format(vertex[0], vertex[1], 0))
            
            for face in self.faces:
                f.write("f")
                for vertex_index in face:
                    f.write(" {}".format(vertex_index + 1))  # OBJ file indices start from 1
                f.write("\n")

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

    def mesh(self):
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

        fig, ax = plt.subplots()
        ax.plot(self.outline[:, 0], self.outline[:, 1], 'r-')
        mesh.plot(ax=ax, show=False, mode="solid")
        plt.show()
        
        return mesh

class RectMesher:
    def __init__(self, corners, resolution):
        self.corners = corners
        self.resolution = resolution

    def mesh(self):
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

        return Mesh(points, faces, boundary)

if __name__ == '__main__':

    # EasyMesher to make a rectangle mesh
    w, h = 3, 1
    outline = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    mesher = EasyMesher(outline, approx_triangles=500)
    mesh = mesher.mesh()
    mesh.save_to_obj("meshes/rect.obj")
    mesh.save('meshes/easy_rectangle.pkl')

    # # plot neighbors
    # face_neighbors = mesh.calculate_face_neighbors()
    # fig, ax = plt.subplots()
    # for face_idx, neighbors in face_neighbors.items():
    #     if face_idx % 2 == 0 or face_idx % 3 == 0:
    #         continue
    #     center = np.mean(mesh.points[mesh.faces[face_idx]], axis=0)
    #     random_color = np.random.rand(3)
    #     for neighbor_idx in neighbors:
    #         neighbor_center = np.mean(mesh.points[mesh.faces[neighbor_idx]], axis=0)
    #         ax.plot([center[0], neighbor_center[0]], [center[1], neighbor_center[1]], color=random_color)
    # mesh.plot(ax=ax, show=False)
    # plt.show()

    corners = [[0, 0], [2, 1]]
    mesher = RectMesher(corners, resolution=(80, 40))
    mesh2 = mesher.mesh()
    mesh2.save('meshes/spring_80x40.pkl')
    mesh2.plot(save='results/rect_mesh.png')