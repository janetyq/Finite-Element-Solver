import numpy as np
import matplotlib.pyplot as plt

# TODO: refactor

class HalfEdge:
    def __init__(self, vertex, flip, next, face):
        self.he_vertex = vertex
        self.flip = flip
        self.next = next
        self.face = face

    def __repr__(self):
        return f'HalfEdge({self.he_vertex}, {self.next.he_vertex})'

class HEVertex:
    def __init__(self, point, half_edge, index):
        self.point = point
        self.he_edge = half_edge
        self.index = index

class HEFace:
    def __init__(self, vertices, half_edge, index):
        self.he_vertices = vertices
        self.he_edge = half_edge
        self.index = index

class HalfEdgeMesh:
    def __init__(self, mesh):
        self.he_vertices = []
        self.he_faces = []
        self.he_edges = []

        self.he_edges_dict = {}

        self.build_half_edge_mesh(mesh)

    def build_half_edge_mesh(self, mesh):
        points = mesh.points
        faces = mesh.faces
        self.he_vertices = [HEVertex(points[i], None, i) for i in range(len(points))]

        for i, face in enumerate(faces):
            self.he_faces.append(HEFace(None, None, i))
            face_half_edges = []
            edge_vec1 = points[face[1]] - points[face[0]]
            edge_vec2 = points[face[2]] - points[face[0]]
            orientation =  1 if np.cross(edge_vec1, edge_vec2) > 0 else -1
            self.he_faces[i].he_vertices = [self.he_vertices[v_idx] for v_idx in face[::orientation]]
            for v_idx in face[::orientation]:
                vertex = self.he_vertices[v_idx]
                half_edge = HalfEdge(vertex, None, None, self.he_faces[i])
                face_half_edges.append(half_edge)
                if vertex.he_edge is None:
                    # set vertex half edge
                    vertex.he_edge = half_edge
            
            # set all next pointers by rotating through face half edges
            for j in range(3):
                half_edge = face_half_edges[j]
                half_edge.next = face_half_edges[(j+orientation)%3]
                self.he_edges_dict[(half_edge.he_vertex.index, half_edge.next.he_vertex.index)] = half_edge
            self.he_faces[i].he_edge = face_half_edges[0]
            self.he_edges.extend(face_half_edges)
        
        # set all flip pointers
        for half_edge in self.he_edges:
            flip = self.find_half_edge(half_edge.next.he_vertex, half_edge.he_vertex)
            if flip is None:
                # boundary edge, add flip half edge (w/o face, next) for convenience
                flip = HalfEdge(half_edge.next.he_vertex, half_edge, None, None)
                self.he_edges_dict[(half_edge.next.he_vertex, half_edge.he_vertex)] = flip
            half_edge.flip = flip

    def find_half_edge(self, vertex1, vertex2):
        # find matching half edge that goes from vertex1 -> vertex2
        idxs = (vertex1.index, vertex2.index)
        if idxs in self.he_edges_dict:
            return self.he_edges_dict[idxs]

    # takes in and returns indices
    def get_v_neighbor_idxs(self, v_idx):
        # get all faces, vertices connected to vertex
        neighbor_v_idxs, neighbor_f_idxs = [], []
        vertex = self.he_vertices[v_idx]
        half_edge = vertex.he_edge # outgoing half edge
        stopped = False
        while True:
            neighbor_v_idxs.append(half_edge.flip.he_vertex.index) # vertex pointed to by half edge
            if half_edge.flip.face is not None:
                neighbor_f_idxs.append(half_edge.flip.face.index)
            if half_edge.flip.next is None:
                stopped=True 
                break
            # try to go to next outgoing half edge
            half_edge = half_edge.flip.next
            if half_edge is None or half_edge == vertex.he_edge:
                break
        if stopped and vertex.he_edge.next is not None: # need to check other side because of boundary
            # try other side half edge
            half_edge = vertex.he_edge.next.next.flip
            while True:
                neighbor_v_idxs.append(half_edge.flip.he_vertex.index)
                if half_edge.flip.face is not None:
                    neighbor_f_idxs.append(half_edge.flip.face.index)
                if half_edge.next is None:
                    break
                half_edge = half_edge.next.next.flip
                if half_edge is None or half_edge == vertex.he_edge:
                    break
        return neighbor_v_idxs, neighbor_f_idxs

    def get_v_neighbor_f_idxs(self, v_idx):
        # get all faces connected to vertex
        return self.get_v_neighbor_idxs(v_idx)[1]
    
    def get_v_neighbor_v_idxs(self, v_idx):
        # get all vertices connected to vertex
        return self.get_v_neighbor_idxs(v_idx)[0]

    def get_f_neighbor_f_idxs(self, face_idx):
        # get all faces connected to face
        face = self.he_faces[face_idx]
        neighbor_f_idxs = []
        he_edge = face.he_edge
        while True:
            if he_edge.flip.face is not None:
                neighbor_f_idxs.append(he_edge.flip.face.index)
            he_edge = he_edge.next
            if he_edge is None or he_edge == face.he_edge:
                break
        return list(set(neighbor_f_idxs))

# if __name__ == '__main__':
#     from mesh import *

#     np.set_printoptions(linewidth=200)

#     # MESH
#     MESH_FILE = '../shared_meshes/square20_mesh.pkl'
#     mesh = Mesh.load(MESH_FILE)

            
#     he_mesh = HalfEdgeMesh(mesh)

#     # TESTING HALF MESH DS
#     import random
#     for i in range(10):
#         f_idx = random.randint(0, len(he_mesh.he_faces)-1)
#         center_face = he_mesh.he_faces[f_idx]
#         neighbor_f_idxs = he_mesh.get_f_neighbor_f_idxs(f_idx)
#         ax = mesh.plot(show=False)
#         print(len(neighbor_f_idxs))
#         for f_idx in neighbor_f_idxs:
#             face = he_mesh.he_faces[f_idx]
#             face_points = np.mean([vertex.point for vertex in face.he_vertices], axis=0)
#             ax.plot(face_points[0], face_points[1], 'bo')
#         face_point = np.mean([vertex.point for vertex in center_face.he_vertices], axis=0)
#         ax.plot(face_point[0], face_point[1], 'ro')
#         plt.show()


#     for i in range(10):
#         v_idx = random.randint(0, len(he_mesh.he_vertices)-1)
#         vertex = he_mesh.he_vertices[v_idx]
#         neighbor_v_idxs = he_mesh.get_v_neighbor_v_idxs(v_idx)
#         neighbor_f_idxs = he_mesh.get_v_neighbor_f_idxs(v_idx)
#         ax = mesh.plot(show=False)
#         ax.plot(vertex.point[0], vertex.point[1], 'ro')
#         print(len(neighbor_v_idxs))
#         for v_idx in neighbor_v_idxs:
#             vertex = he_mesh.he_vertices[v_idx]
#             ax.plot(vertex.point[0], vertex.point[1], 'go')
#         for f_idx in neighbor_f_idxs:
#             face = he_mesh.he_faces[f_idx]
#             face_points = np.mean([vertex.point for vertex in face.he_vertices], axis=0)
#             ax.plot(face_points[0], face_points[1], 'bo')
#         plt.show()



