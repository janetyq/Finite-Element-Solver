from datetime import datetime
import numpy as np
import pickle

from FEMesh import *
from Plotter import *

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

    def get_values(self, name, iter_idx=None, mode=None):
        if name is None:
            return np.zeros(len(self.mesh.elements))
        elif name not in self.values:
            print('--> contains:', self.values.keys())
            raise ValueError(f'{name} not found in solution')
        
        values = self.values[name][iter_idx] if iter_idx is not None else self.values[name]
        if mode is None:
            return values
        elif mode == 'element':
            if len(values) == len(self.mesh.elements):
                return values
            elif len(values) == len(self.mesh.vertices):
                return self._convert_vertex_values_to_element_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        elif mode == 'vertex':
            if len(values) == len(self.mesh.vertices):
                return values
            elif len(values) == len(self.mesh.elements):
                return self._convert_element_values_to_vertex_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')

    def set_values(self, name, value):
        self.values[name] = value

    def reset(self):
        self.values = {}

    def get_deformed_mesh(self):
        return FEMesh(self.mesh.vertices + self.get_values('u').reshape(-1, 2), self.mesh.elements, self.mesh.boundary)

    @classmethod
    def combine_solutions(cls, solution_list):
        combined_solution = Solution(solution_list[0].mesh) # TODO: bit weird
        for name in solution_list[0].values.keys():
            combined_solution.values[name + '_list'] = np.array([s.get_values(name) for s in solution_list])
        return combined_solution
