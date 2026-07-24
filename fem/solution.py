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

from fem.typing import DofVector, ElementField, FloatArray

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh


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
    '''A displacement field plus the stresses recovered from it.'''
    strain: ElementField
    stress: ElementField
    compliance: ElementField


@dataclass(frozen=True, eq=False)
class TransientSolution(Solution):
    '''A time series: the times t and the field u at each step.'''
    t: FloatArray
    u: list[DofVector]


@dataclass(frozen=True, eq=False)
class WaveSolution(TransientSolution):
    '''A time series that also carries the velocity du/dt at each step.'''
    dudt: list[DofVector]
