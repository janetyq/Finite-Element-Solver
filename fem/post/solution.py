"""Typed solution containers: one dataclass per solve shape.

A `FieldSolution` is a `NodalField` (the unknown on its space) with the provenance a
solve adds: `DiffusionSolution` its recovered gradient, `ElasticSolution` the stress
state. The other shapes are collections of fields on one space and hand them out as
such: `TransientSolution` is a time series indexed by step (`history[i]` is the typed
steady solution, `WaveSolution` adds `velocity(i)`), `BucklingSolution` and
`ModalSolution` hold eigenpairs (`mode(i)` is a `NodalField`). `save`/`load`
round-trip any of them through `fem.post.io`, which reflects over the dataclass
fields.

`save` and `load` import `fem.post.io` lazily: I/O reads the solution types, so the edge
points up and stays function-local.
"""
from dataclasses import dataclass, field
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np

from fem.field import NodalField
from fem.post import invariants
from fem.post import recovery
from fem.post.recovery import RecoveryMethod
from fem.typing import DofVector, ElementValues, FloatArray

S = TypeVar('S', bound='FieldSolution')   # the steady solution a step packages
T = TypeVar('T', bound='Solution')

if TYPE_CHECKING:
    from fem.elements import Element
    from fem.physics.forms import ElasticPointState, Form, RecoversElasticState
    from fem.mesh.mesh import Mesh
    from fem.space import FunctionSpace


class Solution:
    '''What every solution shares: the `FunctionSpace` its values live on, and
    persistence.

    A mixin rather than a dataclass base, so `FieldSolution` can take its `space` from
    `NodalField`; the eigen and transient solutions declare `space` themselves. `save`
    stores the mesh and the space's parameters, and `load` rebuilds the space, whose
    numbering is deterministic.
    '''
    space: 'FunctionSpace'

    @property
    def mesh(self) -> 'Mesh':
        return self.space.mesh

    @property
    def n_components(self) -> int:
        return self.space.n_components

    @property
    def element_type(self) -> 'type[Element]':
        return self.space.element_type

    def save(self, path: str) -> None:
        from fem.post.io import save_solution
        save_solution(self, path)

    @classmethod
    def load(cls: type[T], path: str) -> T:
        '''The solution saved at `path`, checked to be a `cls`: `ElasticSolution.load`
        returns an `ElasticSolution` or raises; `Solution.load` takes any.'''
        from fem.post.io import load_solution
        loaded = load_solution(path)
        if not isinstance(loaded, cls):
            raise TypeError(f'{path} holds a {type(loaded).__name__}, not a {cls.__name__}')
        return loaded


@dataclass(frozen=True, eq=False)
class FieldSolution(NodalField, Solution):
    '''A single steady field: Projection, and the base of Poisson and elasticity.

    A `NodalField`, so it reads by node (`nodal_values`, `component`), integrates,
    evaluates at points, and warps the mesh; the subclasses add what their physics
    recovers from it.
    '''


@dataclass(frozen=True, eq=False)
class DiffusionSolution(FieldSolution):
    '''A scalar field plus its recovered per-element gradient: the solution of
    `DiffusionForm` (Poisson, heat, wave).

    `gradient` is one `grad u` per element (the element mean), the `NodalField`
    gradient held so a loaded solution carries it. The diffusive flux `kappa grad u`
    is the form's `GradientFlux`, which applies the coefficient. `nodal_gradient`
    gives the continuous per-node field a P2 plot or a nodal consumer wants,
    re-evaluated at the nodes so a P2 gradient's variation within the element is kept.
    '''
    gradient: ElementValues   # (n_elements, spatial_dim) per-element grad u

    @classmethod
    def from_solve(cls, space: 'FunctionSpace', dofs: DofVector) -> 'DiffusionSolution':
        '''Package a scalar solve, recovering its per-element gradient.'''
        return cls(space, dofs, gradient=NodalField(space, dofs).gradient())

    def nodal_gradient(self, method: RecoveryMethod = 'average') -> FloatArray:
        '''(n_nodes, spatial_dim) continuous gradient at the nodes.

        `method` is the recovery (`'average'` or `'l2'`); see `fem.post.recovery`.
        '''
        return recovery.nodal_gradient(self.space, self.dofs, method=method)


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
    compliance: ElementValues  # (n_elements,)
    form: 'RecoversElasticState | None' = field(
        default=None, kw_only=True, repr=False, metadata={'persist': False})

    def __post_init__(self) -> None:
        super().__post_init__()
        # `fem.post.io` rebuilds this from stored arrays without checking their rank.
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
        dofs: DofVector,
        form: 'RecoversElasticState',
    ) -> 'ElasticSolution':
        '''Recover the elastic fields for `dofs` and package them.'''
        # (n_elements, N, n_components): the layout RecoversElasticState takes, and
        # the same one FunctionSpace.assemble_residual gathers.
        u_elements = NodalField(space, dofs).element_values
        fields = form.recover(space.geometry, u_elements)
        return cls(space, dofs, fields.strain, fields.stress, fields.compliance, form=form)

    @property
    def von_mises(self) -> ElementValues:
        '''Von Mises equivalent stress per element: the usual scalar to plot.'''
        return invariants.von_mises(self.stress)

    def nodal_stress(self, method: RecoveryMethod = 'average') -> FloatArray:
        '''(n_nodes, 3, 3) continuous stress at the nodes.

        `'average'` evaluates each element's stress at its own nodes and volume-averages
        the elements sharing a node; `'l2'` projects the stress sampled at quadrature
        points onto the nodal space. Both keep a P2 stress's variation within the
        element, so a boundary node gets the boundary value. Without `form` (a loaded
        solution) they fall back to recovering the per-element tensor.
        '''
        return self._nodal_field(self.stress, lambda fields: fields.stress, method)

    def nodal_strain(self, method: RecoveryMethod = 'average') -> FloatArray:
        '''(n_nodes, 3, 3) continuous strain at the nodes; see `nodal_stress`.'''
        return self._nodal_field(self.strain, lambda fields: fields.strain, method)

    def _nodal_field(self, stored: FloatArray,
                     sampled: 'Callable[[ElasticPointState], FloatArray]',
                     method: RecoveryMethod) -> FloatArray:
        if self.form is None:
            return recovery.recover_nodal(self.space, stored, method=method)
        space = self.space
        u_elements = self.element_values
        if method == 'average':
            fields = self.form.sample(space.geometry_at_nodes, u_elements)
            return recovery.average_to_nodal(space, sampled(fields))
        if method == 'l2':
            # A degree-p field's gradient is degree p - 1; the rule that integrates its
            # product with a shape function exactly is 2p - 1, and 2p is the cached one.
            geometry = space.geometry_at(2 * space.element_type.SHAPE_DEGREE)
            fields = self.form.sample(geometry, u_elements)
            return recovery.project_to_nodal(space, sampled(fields), geometry)
        raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")

    def nodal_von_mises(self, method: RecoveryMethod = 'average') -> FloatArray:
        '''(n_nodes,) von Mises stress at the nodes, the smooth field to plot.

        Recover-then-reduce: the stress tensor is recovered to the nodes first, then the
        von Mises scalar formed there. Reducing first and averaging the per-element von
        Mises is a different, less faithful number, since the reduction is nonlinear.
        `method` is the tensor recovery (`'average'` or `'l2'`).
        '''
        return invariants.von_mises(self.nodal_stress(method=method))

    @property
    def pressure(self) -> ElementValues:
        '''Hydrostatic pressure per element, positive in compression.'''
        return invariants.pressure(self.stress)

    @property
    def principal_stress(self) -> FloatArray:
        '''(n_elements, 3) principal stresses, ascending.'''
        return invariants.principal(self.stress)

    @property
    def max_shear(self) -> ElementValues:
        '''Maximum shear stress per element.'''
        return invariants.max_shear(self.stress)


class ModeShapes:
    '''What an eigen-solution shares: `modes (n_modes, n_dofs)` on `space`, handed out
    as fields.

    A mode is a shape, not a displacement: its amplitude is arbitrary (the eigenproblem
    is homogeneous), so only its form is physical and the scale of a drawing is a
    display choice: `mode(i).deformed_mesh(scale)`.
    '''
    space: 'FunctionSpace'
    modes: FloatArray

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def mode(self, i: int) -> NodalField:
        '''Mode `i` as a field on the space.'''
        return NodalField(self.space, self.modes[i])


@dataclass(frozen=True, eq=False)
class BucklingSolution(ModeShapes, Solution):
    '''Linearised buckling result: critical load factors and their mode shapes.

    `load_factors[i]` is λ_i, the multiplier on the reference load at which the
    structure buckles into `mode(i)`, the eigenvalues of `K φ = -λ K_g φ`, in
    ascending order, so `load_factors[0]` is the critical (lowest) one and its mode
    is the shape the structure buckles into first. `reference` is the pre-buckling
    solve the modes were computed about (its stress is the prestress); it is not saved.
    '''
    space: 'FunctionSpace'
    load_factors: FloatArray   # (n_modes,) ascending λ
    modes: FloatArray          # (n_modes, n_dofs) mode-shape displacement vectors
    reference: 'ElasticSolution | None' = field(
        default=None, kw_only=True, repr=False, metadata={'persist': False})

    @property
    def critical_load_factor(self) -> float:
        '''The lowest buckling factor λ_1: the one a real structure reaches first.'''
        return float(self.load_factors[0])


@dataclass(frozen=True, eq=False)
class ModalSolution(ModeShapes, Solution):
    '''Free-vibration result: natural frequencies and their mode shapes.

    `angular_frequencies[i]` is omega_i (rad/s), ascending, and `mode(i)` the shape the
    structure oscillates in at that frequency: the eigenpairs of `K phi = omega^2 M phi`.
    Any real free vibration is a superposition of the modes, weighted by how the
    structure was set moving. `frequencies` (Hz) and `periods` (s) are the same data in
    engineering units.
    '''
    space: 'FunctionSpace'
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


@dataclass(frozen=True, eq=False)
class TransientSolution(Solution, Generic[S]):
    '''A time series: the times `t (n_steps,)` and the DOF vectors `dofs (n_steps,
    n_dofs)`, one row per step, step 0 the initial state.

    A sequence of steady solutions: `history[i]` (negative indices count from the end)
    packages step `i` as the typed solution the operator recovers (gradient for heat,
    stress for elasticity), `len` is the step count, and iterating yields every step.
    `operator` is the `Form` that packages; it is not saved, so a loaded series
    packages a bare `FieldSolution`.
    '''
    space: 'FunctionSpace'
    t: FloatArray
    dofs: FloatArray
    operator: 'Form[S] | None' = field(
        default=None, kw_only=True, repr=False, metadata={'persist': False})

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        dofs = np.asarray(self.dofs, dtype=float)
        if dofs.ndim != 2 or dofs.shape[1] != self.space.n_dofs or len(t) != len(dofs):
            raise ValueError(
                f'a series on {self.space!r} needs dofs of shape (n_steps, '
                f'{self.space.n_dofs}) and one time per step; got dofs {dofs.shape} '
                f'and {len(t)} times'
            )
        object.__setattr__(self, 't', t)
        object.__setattr__(self, 'dofs', dofs)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int) -> S:
        '''Step `i` as a steady solution, with the derived field the operator recovers.'''
        if not isinstance(i, (int, np.integer)):
            raise TypeError(f'a series is indexed by step, got {type(i).__name__}')
        dofs = self.dofs[i]
        if self.operator is None:
            return cast(S, FieldSolution(self.space, dofs))
        return self.operator.solution(self.space, dofs)

    def __iter__(self) -> Iterator[S]:
        return (self[i] for i in range(len(self)))


@dataclass(frozen=True, eq=False)
class WaveSolution(TransientSolution[S]):
    '''A time series that also carries the velocity `dudt (n_steps, n_dofs)`.'''
    dudt: FloatArray

    def __post_init__(self) -> None:
        super().__post_init__()
        dudt = np.asarray(self.dudt, dtype=float)
        if dudt.shape != self.dofs.shape:
            raise ValueError(f'dudt {dudt.shape} must match dofs {self.dofs.shape}')
        object.__setattr__(self, 'dudt', dudt)

    def velocity(self, i: int) -> NodalField:
        '''The velocity at step `i` as a field on the space.'''
        return NodalField(self.space, self.dudt[i])
