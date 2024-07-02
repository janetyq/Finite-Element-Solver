from datetime import datetime
import matplotlib.animation as animation 
import numpy as np
import pickle

from Mesh import *

class Solution:
    def __init__(self, mesh, values=None):
        self.mesh = mesh
        self.values = values if values is not None else {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return cls(pickle.load(f))
    
    def __copy__(self):
        values_copy = {k: v.copy() for k, v in self.values.items()}
        return self.__class__(self.mesh.copy(), values_copy)

    def __reduce__(self):
        return (self.__class__, (self.mesh, self.values))

    def get_values(self, name, idx=None, mode=None):
        if name is None:
            return np.zeros(len(self.mesh.faces))
        elif name not in self.values:
            print('--> contains:', self.values.keys())
            raise ValueError(f'{name} not found in solution')
        
        values = self.values[name][idx] if idx is not None else self.values[name]
        if mode is None:
            return values
        elif mode == 'face':
            if len(values) == len(self.mesh.faces):
                return values
            elif len(values) == len(self.mesh.points):
                return self._convert_vertex_values_to_face_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        elif mode == 'vertex':
            if len(values) == len(self.mesh.points):
                return values
            elif len(values) == len(self.mesh.faces):
                return self._convert_face_values_to_vertex_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')


    def set_values(self, name, value):
        self.values[name] = value

    def reset(self):
        self.values = {}

    def calc_gradient(self, name):
        values = self.get_values(name)
        gradient = np.zeros((len(self.mesh.faces), 2))
        for face_idx, face in enumerate(self.mesh.faces):
            element = self.mesh.points[face]
            for i in range(3):
                edge_i = element[(i+1)%3] - element[i]
                edge_j = element[(i+2)%3] - element[i]
                sign = calc_cross(edge_i, edge_j) / np.abs(calc_cross(edge_i, edge_j))
                gradient[face_idx] += sign * values[face[(i+2)%3]] * np.array([-edge_i[1], edge_i[0]]) / (2*self.mesh.areas[face_idx])
        self.values["grad_" + name] = gradient
        return gradient

    def _convert_vertex_values_to_face_values(self, vertex_values):
        assert len(vertex_values) == len(self.mesh.points)
        face_values = np.zeros(len(self.mesh.faces))
        for face_idx, face in enumerate(self.mesh.faces):
            face_values[face_idx] = np.mean([vertex_values[v_idx] for v_idx in face])
        return face_values

    def _convert_face_values_to_vertex_values(self, face_values):
        assert len(face_values) == len(self.mesh.faces)
        vertex_values = np.zeros(len(self.mesh.points))
        for face_idx, face in enumerate(self.mesh.faces):
            for v_idx in face:
                vertex_values[v_idx] = face_values[face_idx]
        return vertex_values

    