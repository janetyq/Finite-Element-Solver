"""Typed solution containers: one dataclass per solve shape.

A `FieldSolution` carries the unknown `u`; `ElasticSolution` adds the recovered
stress fields; `TransientSolution` is a time series and `WaveSolution` adds the
velocity series. `save`/`load` round-trip any of them through `fem.io`, which
reflects over the dataclass fields.
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
    the mesh). Without it `u` is under-specified: which node each entry belongs to
    is the space's numbering, not the mesh's.
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

        Only the leading vertex DOFs move the geometry: a P2 field's edge-midpoint
        DOFs have no mesh vertices, so the warp is the field's P1 restriction.
        '''
        mesh = self.mesh.copy()
        n_vertices = len(mesh.vertices)
        mesh.vertices = mesh.vertices + self.u.reshape(-1, self.n_components)[:n_vertices]
        return mesh


@dataclass(frozen=True, eq=False)
class ScalarFieldSolution(FieldSolution):
    '''A scalar field plus its recovered per-element flux `grad u` (Poisson's solution).

    `flux` is one gradient per element (the element mean). `nodal_flux` gives the
    continuous per-node field a P2 plot or a nodal consumer wants, re-evaluated from `u`
    at the nodes so a P2 gradient's variation within the element is kept.
    '''
    flux: ElementField   # (n_elements, spatial_dim) per-element grad u

    @classmethod
    def from_solve(cls, space: 'FunctionSpace', u: DofVector) -> 'ScalarFieldSolution':
        '''Package a scalar solve, recovering its per-element diffusion flux grad u.'''
        return cls(space.mesh, space.n_components, u,
                   flux=space.gradient(u), element_type=space.element_type)

    def nodal_flux(self, method: str = 'average') -> FloatArray:
        '''(n_nodes, spatial_dim) continuous flux at the nodes.

        `method` is the recovery (`'average'` or `'l2'`); see `FunctionSpace.nodal_gradient`.
        '''
        return self.space.nodal_gradient(self.u, method=method)


@dataclass(frozen=True, eq=False)
class ElasticSolution(FieldSolution):
    '''A displacement field plus the stress state recovered from it.

    Stress and strain are stored as one tensor per element (the element mean); the
    scalar measures are properties. `form` is the elastic form that recovered them,
    kept so the nodal recoveries can re-evaluate the fields at the nodes and
    quadrature points, where a P2 stress varies within the element. It is not
    saved: a loaded solution recovers from the per-element tensors alone, exact for
    P1 and a coarser reading for P2.
    '''
    strain: FloatArray       # (n_elements, 3, 3)
    stress: FloatArray       # (n_elements, 3, 3)
    compliance: ElementField  # (n_elements,)
    form: 'RecoversElasticFields | None' = field(
        default=None, kw_only=True, repr=False, metadata={'persist': False})

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
                   element_type=space.element_type, form=form)

    @property
    def von_mises(self) -> ElementField:
        '''Von Mises equivalent stress per element: the usual scalar to plot.'''
        return invariants.von_mises(self.stress)

    def nodal_stress(self, method: str = 'average') -> FloatArray:
        '''(n_nodes, 3, 3) continuous stress at the nodes.

        `'average'` evaluates each element's stress at its own nodes and volume-averages
        the elements sharing a node; `'l2'` projects the stress sampled at quadrature
        points onto the nodal space. Both keep a P2 stress's variation within the
        element, so a boundary node gets the boundary value. Without `form` (a loaded
        solution) they fall back to recovering the per-element tensor.
        '''
        return self._nodal_field('stress', method)

    def nodal_strain(self, method: str = 'average') -> FloatArray:
        '''(n_nodes, 3, 3) continuous strain at the nodes; see `nodal_stress`.'''
        return self._nodal_field('strain', method)

    def _nodal_field(self, name: str, method: str) -> FloatArray:
        if self.form is None:
            return self.space.recover_nodal(getattr(self, name), method=method)
        space = self.space
        u_elements = np.asarray(self.u).reshape(-1, self.n_components)[space.element_nodes]
        if method == 'average':
            fields = self.form.fields_at(space.geometry_at_nodes, u_elements)
            return space.average_to_nodal(getattr(fields, name))
        if method == 'l2':
            # A degree-p field's gradient is degree p - 1; the rule that integrates its
            # product with a shape function exactly is 2p - 1, and 2p is the cached one.
            geometry = space.geometry_at(2 * space.element_type.SHAPE_DEGREE)
            fields = self.form.fields_at(geometry, u_elements)
            return space.project_to_nodal(getattr(fields, name), geometry)
        raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")

    def nodal_von_mises(self, method: str = 'average') -> FloatArray:
        '''(n_nodes,) von Mises stress at the nodes, the smooth field to plot.

        Recover-then-reduce: the stress tensor is recovered to the nodes first, then the
        von Mises scalar formed there. Reducing first and averaging the per-element von
        Mises is a different, less faithful number, since the reduction is nonlinear.
        `method` is the tensor recovery (`'average'` or `'l2'`).
        '''
        return invariants.von_mises(self.nodal_stress(method=method))

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
    only its form and the factor scaling the reference load are physical. `reference`
    is the pre-buckling solve the modes were computed about (its stress is the
    prestress); it is not saved.
    '''
    load_factors: FloatArray   # (n_modes,) ascending λ
    modes: FloatArray          # (n_modes, n_dofs) mode-shape displacement vectors
    reference: 'ElasticSolution | None' = field(
        default=None, kw_only=True, repr=False, metadata={'persist': False})

    @property
    def critical_load_factor(self) -> float:
        '''The lowest buckling factor λ_1: the one a real structure reaches first.'''
        return float(self.load_factors[0])

    def mode_mesh(self, i: int, scale: float = 1.0) -> 'Mesh':
        '''The mesh displaced by `scale` times buckling mode `i`, for drawing it.

        The amplitude is arbitrary, so `scale` is a display choice. Only the leading
        vertex DOFs move the geometry (a P2 mode draws as its P1 restriction).
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

        The amplitude is arbitrary, so `scale` is a display choice. Only the leading
        vertex DOFs move the geometry (a P2 mode draws as its P1 restriction).
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
