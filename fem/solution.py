"""Typed solution containers -- one dataclass per solve shape.

Replaces a single dict of named arrays. The fields a solve produces are now typed
attributes: `solution.u` instead of `solution.get_values("u")`, discoverable and
checkable. A steady field (an array) and a time series (a list of arrays) are
different *types* rather than both being `values[...]` told apart by guessing at a
length, which is what the old `get_values(mode=...)` had to do.

The hierarchy follows the physics: a `FieldSolution` carries the unknown `u`;
`ElasticSolution` adds the recovered stress fields; `TransientSolution` is a time
series and `WaveSolution` adds the velocity series. `save`/`load` round-trip any of
them through `fem.io`, which reflects over the dataclass fields.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem import invariants
from fem.typing import DofVector, ElementField, FloatArray

if TYPE_CHECKING:
    from fem.forms import RecoversElasticFields
    from fem.mesh.mesh import Mesh
    from fem.space import FunctionSpace


@dataclass(frozen=True, eq=False)
class Solution:
    '''Base: every solution knows the discretization it was computed on.'''
    mesh: 'Mesh'
    n_components: int

    def save(self, path: str) -> None:
        from fem.io import save_solution
        save_solution(self, path)

    @staticmethod
    def load(path: str) -> 'Solution':
        from fem.io import load_solution
        return load_solution(path)


@dataclass(frozen=True, eq=False)
class FieldSolution(Solution):
    '''A single steady field u -- Projection, Poisson, and the base of elasticity.'''
    u: DofVector

    def deformed_mesh(self) -> 'Mesh':
        '''The mesh displaced by u (meaningful for a vector displacement field).'''
        mesh = self.mesh.copy()
        mesh.vertices = mesh.vertices + self.u.reshape(-1, self.n_components)
        return mesh


@dataclass(frozen=True, eq=False)
class ElasticSolution(FieldSolution):
    '''A displacement field plus the stress state recovered from it.

    `strain` and `stress` are full `(n_elements, 3, 3)` tensors, not pre-reduced
    scalars. Storing the tensor is what keeps every invariant available: a
    Frobenius norm cannot be turned back into a von Mises stress, so reducing at
    construction would decide, permanently and on the caller's behalf, which
    question the result can answer. The scalars are properties instead.
    '''
    strain: FloatArray       # (n_elements, 3, 3)
    stress: FloatArray       # (n_elements, 3, 3)
    compliance: ElementField  # (n_elements,)

    def __post_init__(self) -> None:
        # Every invariant below indexes the last two axes, so a field of the wrong
        # rank would be read as if it were a tensor field and quietly return
        # nonsense. `fem.io` reconstructs this class from stored arrays without
        # checking their shape, which is the path that makes the guard worth having.
        for name in ('strain', 'stress'):
            value = getattr(self, name)
            if np.ndim(value) != 3:
                raise ValueError(
                    f'{type(self).__name__}.{name} must be an (n_elements, 3, 3) '
                    f'tensor field, got shape {np.shape(value)}'
                )

    @classmethod
    def from_solve(
        cls,
        space: 'FunctionSpace',
        u: DofVector,
        form: 'RecoversElasticFields',
    ) -> 'ElasticSolution':
        '''Recover the elastic fields for `u` and package them.

        The one place a solved displacement becomes an `ElasticSolution`. Both a
        facade (`Solver`) and a driver (`TopologyOptimizer`) need this, and they
        used to each spell it out -- which is how a reduction that was not
        rotation invariant came to be written twice. The typed result owning its
        own derivation is the same shape `Problem` has to a specification.

        Takes the `space`, not a mesh and a component count and an element
        geometry: those are three views of one discretization, and passing them
        separately would let a caller hand over a geometry built for a different
        mesh than the one it names. That is the stale-index failure the rest of
        the package is built to prevent, so it is made unrepresentable here too.

        `form` is anything satisfying `RecoversElasticFields`, so the linear and
        energy elastic paths build their solution the same way.
        '''
        mesh, n_components = space.mesh, space.n_components
        # (n_elements, N, n_components) -- the layout RecoversElasticFields is
        # written against, and the same one FunctionSpace.assemble_residual gathers.
        u_elements = np.asarray(u).reshape(-1, n_components)[mesh.elements]
        fields = form.derived_fields(space.geometry, u_elements)
        return cls(mesh, n_components, u, fields.strain, fields.stress, fields.compliance)

    @property
    def von_mises(self) -> ElementField:
        '''Von Mises equivalent stress per element -- the usual scalar to plot.'''
        return invariants.von_mises(self.stress)

    @property
    def pressure(self) -> ElementField:
        '''Hydrostatic pressure per element, positive in compression.'''
        return invariants.pressure(self.stress)

    @property
    def principal_stress(self) -> FloatArray:
        '''(n_elements, 3) principal stresses, ascending.'''
        return invariants.principal(self.stress)

    @property
    def max_shear(self) -> ElementField:
        '''Maximum shear stress per element.'''
        return invariants.max_shear(self.stress)


@dataclass(frozen=True, eq=False)
class TransientSolution(Solution):
    '''A time series: the times t and the field u at each step.'''
    t: FloatArray
    u: list[DofVector]


@dataclass(frozen=True, eq=False)
class WaveSolution(TransientSolution):
    '''A time series that also carries the velocity du/dt at each step.'''
    dudt: list[DofVector]
