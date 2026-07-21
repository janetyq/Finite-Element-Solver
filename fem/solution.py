from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fem.typing import FloatArray

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh

# Which discretization a caller wants values on. `Solution` stores whatever the
# solver produced and converts on request.
ValueMode = Literal['element', 'vertex']


class Solution:
    def __init__(self, mesh: 'Mesh', n_components: int) -> None:
        self.mesh = mesh
        # Heterogeneous by design: "u" is a DofVector, "t_values" a list of
        # floats, "u_values" a list of arrays per timestep.
        self.values: dict[str, Any] = {}
        self.n_components = n_components

    def save(self, filename: str) -> None:
        from fem.io import save_solution
        save_solution(self, filename)

    @classmethod
    def load(cls, filename: str) -> 'Solution':
        from fem.io import load_solution
        return load_solution(filename)

    def get_values(
        self,
        name: str | None,
        iter_idx: int | None = None,
        mode: ValueMode | None = None,
    ) -> Any:
        if name is None:
            return np.zeros(len(self.mesh.elements))
        elif name not in self.values:
            raise ValueError(f'{name} not found in solution (has: {list(self.values.keys())})')
        
        values = self.values[name][iter_idx] if iter_idx is not None else self.values[name]
        if mode is None:
            return values
        elif mode == 'element':
            if len(values) == len(self.mesh.elements):
                return values
            elif len(values) == len(self.mesh.vertices):
                return self.mesh.convert_vertex_values_to_element_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        elif mode == 'vertex':
            if len(values) == len(self.mesh.vertices):
                return values
            elif len(values) == len(self.mesh.elements):
                return self.mesh.convert_element_values_to_vertex_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        # Falling through returned None, which only failed later where the caller
        # indexed it -- get_deformed_mesh reshaping a None being the usual way.
        raise ValueError(f"unknown mode {mode!r}: expected 'element', 'vertex', or None")

    def set_values(self, name: str, value: Any) -> None:
        self.values[name] = value

    def reset(self) -> None:
        self.values = {}

    def get_deformed_mesh(self, u: FloatArray | None = None) -> 'Mesh':
        displacement = self.get_values('u') if u is None else u
        mesh_deformed = self.mesh.copy()
        mesh_deformed.vertices += displacement.reshape(-1, self.n_components)
        return mesh_deformed

    @classmethod
    def combine_solutions(cls, solution_list: list['Solution']) -> 'Solution':
        combined_solution = Solution(solution_list[0].mesh, 2) # TODO: bit weird
        for name in solution_list[0].values.keys():
            combined_solution.values[name + '_list'] = np.array([s.get_values(name) for s in solution_list])
        return combined_solution
