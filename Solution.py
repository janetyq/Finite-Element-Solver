from datetime import datetime
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

    def calculate_gradient(self, name): # TODO: rename, overloaded
        values = self.get_values(name)
        self.values["grad_" + name] = self.mesh.calculate_gradient(values)
        return self.values["grad_" + name]

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

    def plot(self, name, idx=None, deformed=False, mode=None, options=None): #TODO: support None name
        # special to mechanics solve
        values = self.get_values(name, idx=idx)
        options = options if options is not None else {}
        options['title'] = options.get('title', name)
        if deformed:
            points = self.mesh.points + self.get_values('u').reshape(-1, 2)
            deformed_mesh = Mesh(points, self.mesh.faces, self.mesh.boundary)
            return Plotter(deformed_mesh, options=options).plot_values(values, mode=mode)
        return Plotter(self.mesh, options=options).plot_values(values, mode=mode)

    @classmethod
    def combine_solutions(cls, solution_list):
        combined_solution = Solution(solution_list[0].mesh) # TODO: bit weird
        for name in solution_list[0].values.keys():
            combined_solution.values[name + '_list'] = np.array([s.get_values(name) for s in solution_list])
        return combined_solution
