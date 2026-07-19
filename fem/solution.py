import numpy as np

class Solution:
    def __init__(self, femesh, dim):
        self.femesh = femesh
        self.values = {}
        self.dim = dim

    def save(self, filename):
        from fem.io import save_solution
        save_solution(self, filename)

    @classmethod
    def load(cls, filename):
        from fem.io import load_solution
        return load_solution(filename)
    
    def get_values(self, name, iter_idx=None, mode=None):
        if name is None:
            return np.zeros(len(self.femesh.elements))
        elif name not in self.values:
            raise ValueError(f'{name} not found in solution (has: {list(self.values.keys())})')
        
        values = self.values[name][iter_idx] if iter_idx is not None else self.values[name]
        if mode is None:
            return values
        elif mode == 'element':
            if len(values) == len(self.femesh.elements):
                return values
            elif len(values) == len(self.femesh.vertices):
                return self._convert_vertex_values_to_element_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        elif mode == 'vertex':
            if len(values) == len(self.femesh.vertices):
                return values
            elif len(values) == len(self.femesh.elements):
                return self._convert_element_values_to_vertex_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')

    def set_values(self, name, value):
        self.values[name] = value

    def reset(self):
        self.values = {}

    def get_deformed_mesh(self, u=None):
        if u is None:
            u = self.get_values('u')
        femesh_deformed = self.femesh.copy()
        femesh_deformed.vertices += u.reshape(-1, self.dim)
        return femesh_deformed

    @classmethod
    def combine_solutions(cls, solution_list):
        combined_solution = Solution(solution_list[0].femesh, 2) # TODO: bit weird
        for name in solution_list[0].values.keys():
            combined_solution.values[name + '_list'] = np.array([s.get_values(name) for s in solution_list])
        return combined_solution
