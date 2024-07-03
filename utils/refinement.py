import numpy as np
import matplotlib.pyplot as plt
from Mesh import *

# TODO: very inefficient

class Triangle:
    def __init__(self, vertex_idxs, parent=None, status='red child'):
        self.vertex_idxs = vertex_idxs
        self.status = status
        self.parent = parent        # ptr to triangle
        self.children = []          # ptrs to triangles

    def __repr__(self):
        return f'Triangle \n vertex_idxs: {self.vertex_idxs} \n  status: {self.status} \n  parent: {self.parent is not None} \n  children: {len(self.children)}'

class RefinementMesh:
    def __init__(self, mesh):
        # use this to get mesh
        self.mesh = mesh

        # internal representation - do not use outside of this class
        self.points = mesh.points
        self.faces = mesh.faces
        self.boundary = [list(edge) for edge in mesh.boundary]
        self.triangles = [Triangle(face) for face in self.faces]

        self.triangle_index_map = {idx: idx for idx in range(len(self.triangles))}

    def get_shared_triangle(self, edge, not_idxs=None):
        for idx, triangle in enumerate(self.triangles):
            if not_idxs is not None and idx in not_idxs:
                continue
            if triangle.status == 'gone':
                continue
            if edge[0] in triangle.vertex_idxs and edge[1] in triangle.vertex_idxs:
                return idx
    
    def get_triangle_idx(self, triangle):
        for idx, t in enumerate(self.triangles):
            if t == triangle:
                return idx

    def get_point_idx(self, point):
        for idx, p in enumerate(self.points):
            if (p == point).all():
                return idx

    def refine_triangles(self, refine_list):
        '''
        Refines triangles in refine_list
        and updates self.mesh
        '''
        for idx in refine_list:
            # print(self.triangles[triangle_idx].status)
            self.refine(self.triangle_index_map[idx])
        # self.plot(title=f'refined {triangle_idx}', triangle_idxs=refine_list)
        
        for triangle_idx in list(range(len(self.triangles)))[::-1]:
            if self.triangles[triangle_idx].status == 'gone':
                self.triangles.pop(triangle_idx)
        
        self.update_mesh()

    def refine(self, triangle_idx):
        # print('refine', triangle_idx)
        triangle = self.triangles[triangle_idx]
        if triangle.status == 'red parent':
            # already refined, do nothing
            return
        elif triangle.status == 'red child':
            # refine as red
            self.refine_red(triangle_idx)
        elif triangle.status == 'green parent':
            # if green, rollback to parent triangle and refine as red
            parent_idx = self.rollback_green(triangle_idx)
            self.refine_red(parent_idx)
        elif triangle.status == 'green child':
            parent_idx = self.rollback_green(triangle_idx)
            self.refine_red(parent_idx)
        elif triangle.status == 'gone':
            pass
        else:
            assert False, f'unknown triangle status {triangle.status}'

    def update_boundary(self, edge, idx):
        if edge in self.boundary:
            self.boundary.remove(edge)
            self.boundary.extend([[edge[0], idx], [idx, edge[1]]])
        elif edge[::-1] in self.boundary:
            self.boundary.remove(edge[::-1])
            self.boundary.extend([[edge[1], idx], [idx, edge[0]]])

    def refine_red(self, triangle_idx):
        # print('refine red', triangle_idx)
        triangle = self.triangles[triangle_idx]
        new_point_idxs = []
        for i in range(3):
            edge = [triangle.vertex_idxs[i], triangle.vertex_idxs[(i+1)%3]]
            midpoint = (self.points[edge[0]] + self.points[edge[1]]) / 2
            # if midpoint already exists, use that
            idx = self.get_point_idx(midpoint)
            if idx is None:
                self.points = np.vstack((self.points, midpoint))
                idx = len(self.points) - 1
            new_point_idxs.append(idx)

            self.update_boundary(edge, idx)

        new_triangles = [Triangle([triangle.vertex_idxs[0], new_point_idxs[0], new_point_idxs[2]], parent=triangle, status='red child'),
                        Triangle([triangle.vertex_idxs[1], new_point_idxs[1], new_point_idxs[0]], parent=triangle, status='red child'),
                        Triangle([triangle.vertex_idxs[2], new_point_idxs[2], new_point_idxs[1]], parent=triangle, status='red child'),
                        Triangle([new_point_idxs[0], new_point_idxs[1], new_point_idxs[2]], parent=triangle, status='red child')]
        
        triangle.children = new_triangles
        self.triangles.extend(new_triangles)
        triangle.status = 'red parent'
        new_triangle_idxs = [len(self.triangles) - 4, len(self.triangles) - 3, len(self.triangles) - 2, len(self.triangles) - 1]

        for i in range(3):
            edge = [triangle.vertex_idxs[i], triangle.vertex_idxs[(i+1)%3]]
            shared_idx = self.get_shared_triangle(edge, not_idxs=[triangle_idx])
            if shared_idx is not None:
                shared_triangle = self.triangles[shared_idx]
                if shared_triangle.status == 'red parent':
                    # already refined, do nothing
                    # self.plot(title=f'shared red parent', edge=edge, main_idx=triangle_idx, red_idx=shared_idx)
                    continue
                if shared_triangle.status == 'red child':
                    # self.plot(title=f'shared red child', edge=edge, main_idx=triangle_idx, red_idx=shared_idx)
                    self.refine_green(shared_idx, edge, new_point_idxs[i])
                elif shared_triangle.status == 'green parent':
                    # self.plot(title=f'shared green parent', edge=edge, main_idx=triangle_idx, green_idx=shared_idx)
                    parent_idx = self.rollback_green(shared_idx)
                    self.refine_red(parent_idx)
                elif shared_triangle.status == 'green child':
                    # self.plot(title=f'shared green CHILD', edge=edge, main_idx=triangle_idx, green_idx=shared_idx)
                    parent_idx = self.rollback_green(shared_idx)
                    new_red_triangle_idxs = self.refine_red(parent_idx)
                    for new_idx in new_red_triangle_idxs:
                        new_triangle = self.triangles[new_idx]
                        if edge[0] in new_triangle.vertex_idxs and edge[1] in new_triangle.vertex_idxs:
                            # self.plot(title=f'shared red CHILD', edge=edge, main_idx=triangle_idx, red_idx=new_idx)
                            self.refine_green(new_idx, edge, new_point_idxs[i])
                            break

        return new_triangle_idxs
    
    def rollback_green(self, triangle_idx):
        # print('rollback green', triangle_idx)
        triangle = self.triangles[triangle_idx]
        parent = triangle if len(triangle.children) != 0 else triangle.parent

        parent_idx = self.get_triangle_idx(parent)
        if parent_idx is None: # parent previously removed
            self.triangles.append(parent)
            parent_idx = len(self.triangles) - 1
        parent.status = 'red parent'
        for child in parent.children:
            child.status = 'gone'
        parent.children = []

        return parent_idx

    def refine_green(self, triangle_idx, edge, new_idx):
        # print('refine green', triangle_idx)
        triangle = self.triangles[triangle_idx]
        triangle.status = 'green parent'
        
        opposite_vertex = [v for v in triangle.vertex_idxs if v not in edge][0]
        green_triangle1 = Triangle([edge[0], opposite_vertex, new_idx], parent=triangle, status='green child')
        green_triangle2 = Triangle([edge[1], opposite_vertex, new_idx], parent=triangle, status='green child')
        triangle.children = [green_triangle1, green_triangle2]
        self.triangles.extend([green_triangle1, green_triangle2])

        self.update_boundary(edge, new_idx)

    def update_mesh(self):
        # set mesh as points, faces
        self.triangle_index_map = {}
        faces = []
        for triangle_idx, triangle in enumerate(self.triangles):
            if triangle.status != 'red child' and triangle.status != 'green child':
                continue
            faces.append(triangle.vertex_idxs)
            self.triangle_index_map[len(faces) - 1] = triangle_idx
        faces = np.array(faces)

        used_idxs = list(set(faces.flatten()))
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_idxs)}
        
        self.points = self.points[used_idxs]
        self.faces = np.vectorize(index_mapping.get)(faces)
        self.boundary = np.vectorize(index_mapping.get)(self.boundary)
        self.boundary = [list(edge) for edge in self.boundary]

        self.mesh = Mesh(self.points, self.faces, self.boundary)
    
    def get_mesh(self):
        return self.mesh

    def plot(self, title=None, edge=None, main_idx=None, green_idx=None, red_idx=None, triangle_idxs=None):
        ax = self.mesh.plot(title=title, show=False, linewidth=3)
        if edge is not None:
            edge_points = self.points[edge]
            ax.plot(edge_points[:, 0], edge_points[:, 1], linewidth=3, color='blue')
        if main_idx is not None:
            main_triangle = self.triangles[main_idx]
            main_center = np.mean(self.points[main_triangle.vertex_idxs], axis=0)
            ax.scatter(main_center[0], main_center[1], color='blue')
        if green_idx is not None:
            green_triangle = self.triangles[green_idx]
            green_center = np.mean(self.points[green_triangle.vertex_idxs], axis=0)
            ax.scatter(green_center[0], green_center[1], color='green')
        if red_idx is not None:
            red_triangle = self.triangles[red_idx]
            red_center = np.mean(self.points[red_triangle.vertex_idxs], axis=0)
            ax.scatter(red_center[0], red_center[1], color='red')
        if triangle_idxs is not None:
            for triangle_idx in triangle_idxs:
                triangle = self.triangles[triangle_idx]
                center = np.mean(self.points[triangle.vertex_idxs], axis=0)
                ax.scatter(center[0], center[1], color='black')
        plot_triangles = []
        for triangle in self.triangles:
            if triangle.status == 'red child' or triangle.status == 'green child':
                plot_triangles.append(triangle)
            
        plotting_mesh = Mesh(self.points, [triangle.vertex_idxs for triangle in plot_triangles], self.boundary)
        plotting_mesh.plot(ax=ax, linewidth=1, color='cyan')

    def get_mesh(self):
        return self.mesh

if __name__ == '__main__':
    # MESH
    # MESH_FILE = '../shared_meshes/square20_mesh.pkl'
    # mesh = Mesh.load(MESH_FILE)
    # mesh.plot()
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    mesh = Mesh(points, faces, boundary)
    original_mesh = mesh.copy()

    refinement_mesh = RefinementMesh(mesh)

    import random
    # refine_lists = [{4}, {16}]
    for i in range(8):
        refine_list = set(random.randint(0, len(mesh.faces)-1) for _ in range(5))
        # refine_list = refine_lists[i]
        print(refine_list)
        refinement_mesh.refine_triangles(refine_list)
        mesh = refinement_mesh.get_mesh()
        # mesh.plot()

    fig, ax = plt.subplots(1, 2)
    original_mesh.plot(ax=ax[0], title='Original Mesh', show=False)
    mesh.plot(ax=ax[1], title='Refined Mesh')
    
