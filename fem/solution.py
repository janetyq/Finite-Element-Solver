"""Typed solution containers: one dataclass per solve shape.

Replaces a single dict of named arrays. The fields a solve produces are now typed
attributes: `solution.u` instead of `solution.get_values("u")`, discoverable and
checkable. A steady field (an array) and a time series (a list of arrays) are
different types rather than both being `values[...]` told apart by guessing at a
length, as the old `get_values(mode=...)` had to do.

The hierarchy follows the physics: a `FieldSolution` carries the unknown `u`;
`ElasticSolution` adds the recovered stress fields; `TransientSolution` is a time
series and `WaveSolution` adds the velocity series. `save`/`load` round-trip any of
them through `fem.io`, which reflects over the dataclass fields.
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from fem import invariants
from fem.typing import DofVector, ElementField, FloatArray

if TYPE_CHECKING:
    from fem.elements import Element
    from fem.forms import RecoversElasticFields
    from fem.mesh.mesh import Mesh
    from fem.space import FunctionSpace


@dataclass(frozen=True, eq=False)
class Solution:
    '''Base: every solution knows the discretization it was computed on.

    `element_type` is the element the DOFs live on (None means the linear element for
    the mesh). It is carried because `u` is a vector of nodal DOFs whose meaning (which
    node each entry is, P1 vs P2) is the space's, not the mesh's, so a solution is
    under-specified without it. Keyword-only, so every existing positional construction
    (and a P1 solve) keeps working unchanged.
    '''
    mesh: 'Mesh'
    n_components: int
    element_type: 'type[Element] | None' = field(default=None, kw_only=True)

    @cached_property
    def space(self) -> 'FunctionSpace':
        '''The `FunctionSpace` this solution's DOFs live on.

        Rebuilt from `(mesh, element_type, n_components)`, whose node numbering is
        deterministic, so it matches the space the solve used and reconstructs after
        `load` too. Cached, so nodal recovery and P2 plotting build it once.
        '''
        from fem.space import FunctionSpace
        return FunctionSpace(self.mesh, self.element_type, n_components=self.n_components)

    def save(self, path: str) -> None:
        from fem.io import save_solution
        save_solution(self, path)

    @staticmethod
    def load(path: str) -> 'Solution':
        from fem.io import load_solution
        return load_solution(path)


@dataclass(frozen=True, eq=False)
class FieldSolution(Solution):
    '''A single steady field u: Projection, and the base of Poisson and elasticity.'''
    u: DofVector

    def deformed_mesh(self) -> 'Mesh':
        '''The mesh displaced by u (meaningful for a vector displacement field).

        A P2 field carries edge-midpoint DOFs the mesh has no vertices for, so only the
        leading vertex DOFs move the geometry: the warp draws as its P1 restriction, the
        same simplification `mode_mesh` and the rest of the plot layer make for P2.
        '''
        mesh = self.mesh.copy()
        n_vertices = len(mesh.vertices)
        mesh.vertices = mesh.vertices + self.u.reshape(-1, self.n_components)[:n_vertices]
        return mesh


@dataclass(frozen=True, eq=False)
class ScalarFieldSolution(FieldSolution):
    '''A scalar field plus its recovered per-element flux `grad u` (Poisson's solution).

    `flux` is the element-constant gradient the solve recovers. `nodal_flux` turns it into
    a continuous per-node field by volume-weighted averaging, the smooth flux a P2 plot or
    a nodal consumer wants where the per-element field is piecewise-constant.
    '''
    flux: ElementField   # (n_elements, spatial_dim) element-constant grad u

    @classmethod
    def from_solve(cls, space: 'FunctionSpace', u: DofVector) -> 'ScalarFieldSolution':
        '''Package a scalar solve, recovering its per-element diffusion flux grad u.'''
        return cls(space.mesh, space.n_components, u,
                   flux=space.gradient(u), element_type=space.element_type)

    def nodal_flux(self) -> FloatArray:
        '''(n_nodes, spatial_dim) continuous flux recovered from the per-element gradient.'''
        return self.space.recover_nodal(self.flux)


@dataclass(frozen=True, eq=False)
class ElasticSolution(FieldSolution):
    '''A displacement field plus the stress state recovered from it.

    Stress and strain are stored as tensors; the scalar measures are properties.
    '''
    strain: FloatArray       # (n_elements, 3, 3)
    stress: FloatArray       # (n_elements, 3, 3)
    compliance: ElementField  # (n_elements,)

    def __post_init__(self) -> None:
        # `fem.io` rebuilds this from stored arrays without checking their rank.
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
        '''Recover the elastic fields for `u` and package them.'''
        mesh, n_components = space.mesh, space.n_components
        # (n_elements, N, n_components): the layout RecoversElasticFields takes,
        # and the same one FunctionSpace.assemble_residual gathers. Indexed by the
        # space's element nodes, not the mesh triangles: a P2 element has six nodes.
        u_elements = np.asarray(u).reshape(-1, n_components)[space.element_nodes]
        fields = form.derived_fields(space.geometry, u_elements)
        return cls(mesh, n_components, u, fields.strain, fields.stress, fields.compliance,
                   element_type=space.element_type)

    @property
    def von_mises(self) -> ElementField:
        '''Von Mises equivalent stress per element: the usual scalar to plot.'''
        return invariants.von_mises(self.stress)

    def nodal_stress(self) -> FloatArray:
        '''(n_nodes, 3, 3) continuous stress recovered from the per-element tensor.'''
        return self.space.recover_nodal(self.stress)

    def nodal_strain(self) -> FloatArray:
        '''(n_nodes, 3, 3) continuous strain recovered from the per-element tensor.'''
        return self.space.recover_nodal(self.strain)

    def nodal_von_mises(self) -> FloatArray:
        '''(n_nodes,) von Mises stress at the nodes, the smooth field to plot.

        Recover-then-reduce: the stress tensor is recovered to the nodes first, then the
        von Mises scalar formed there. Reducing first and averaging the per-element von
        Mises is a different, less faithful number, since the reduction is nonlinear.
        '''
        return invariants.von_mises(self.nodal_stress())

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
class BucklingSolution(Solution):
    '''Linearised buckling result: critical load factors and their mode shapes.

    `load_factors[i]` is λ_i, the multiplier on the reference load at which the
    structure buckles into `modes[i]`, the eigenvalues of `K φ = -λ K_g φ`, in
    ascending order, so `load_factors[0]` is the critical (lowest) one and its mode
    is the shape the structure buckles into first. A mode is a shape, not a
    displacement: its amplitude is arbitrary (the eigenproblem is homogeneous), so
    only its form and the factor scaling the reference load are physical.
    '''
    load_factors: FloatArray   # (n_modes,) ascending λ
    modes: FloatArray          # (n_modes, n_dofs) mode-shape displacement vectors

    @property
    def critical_load_factor(self) -> float:
        '''The lowest buckling factor λ_1: the one a real structure reaches first.'''
        return float(self.load_factors[0])

    def mode_mesh(self, i: int, scale: float = 1.0) -> 'Mesh':
        '''The mesh displaced by `scale` times buckling mode `i`, for drawing it.

        The amplitude is meaningless on its own, so `scale` is a display choice: a
        caller picks it to make the shape legible against the structure's size.

        A P2 mode carries edge-midpoint DOFs the mesh has no vertices for, so only the
        leading vertex DOFs move the geometry: the mode draws as its P1 restriction,
        the same simplification the rest of the plot layer makes for P2 fields.
        '''
        mesh = self.mesh.copy()
        n_vertices = len(mesh.vertices)
        displacement = self.modes[i].reshape(-1, self.n_components)[:n_vertices]
        mesh.vertices = mesh.vertices + scale * displacement
        return mesh


@dataclass(frozen=True, eq=False)
class ModalSolution(Solution):
    '''Free-vibration result: natural frequencies and their mode shapes.

    `angular_frequencies[i]` is omega_i (rad/s), ascending, and `modes[i]` the shape the
    structure oscillates in at that frequency: the eigenpairs of `K phi = omega^2 M phi`.
    Like a buckling mode, a mode shape has arbitrary amplitude (the eigenproblem is
    homogeneous): only its form and its frequency are physical, and any real free
    vibration is a superposition of the modes, weighted by how the structure was set
    moving. `frequencies` (Hz) and `periods` (s) are the same data in engineering units.
    '''
    angular_frequencies: FloatArray   # (n_modes,) ascending omega, rad/s
    modes: FloatArray                 # (n_modes, n_dofs) mode-shape displacement vectors

    @property
    def frequencies(self) -> FloatArray:
        '''The natural frequencies in Hz (cycles per second): omega / 2pi.'''
        return self.angular_frequencies / (2 * np.pi)

    @property
    def periods(self) -> FloatArray:
        '''The natural periods in seconds: 1 / f (infinite for a zero-frequency mode).'''
        with np.errstate(divide='ignore'):
            return 1.0 / self.frequencies

    def mode_mesh(self, i: int, scale: float = 1.0) -> 'Mesh':
        '''The mesh displaced by `scale` times mode `i`, for drawing it.

        The amplitude is arbitrary, so `scale` is a display choice a caller picks to make
        the shape legible. A P2 mode carries edge-midpoint DOFs the mesh has no vertices
        for, so only the leading vertex DOFs move the geometry: the mode draws as its P1
        restriction, the same simplification the rest of the plot layer makes for P2.
        '''
        mesh = self.mesh.copy()
        n_vertices = len(mesh.vertices)
        displacement = self.modes[i].reshape(-1, self.n_components)[:n_vertices]
        mesh.vertices = mesh.vertices + scale * displacement
        return mesh


@dataclass(frozen=True, eq=False)
class TransientSolution(Solution):
    '''A time series: the times t and the field u at each step.'''
    t: FloatArray
    u: list[DofVector]


@dataclass(frozen=True, eq=False)
class WaveSolution(TransientSolution):
    '''A time series that also carries the velocity du/dt at each step.'''
    dudt: list[DofVector]
