"""The `Problem`: the assembly-ready statement a solve strategy consumes.

A `Problem` is to a composition of physics what `ResolvedBC` is to a
`BoundaryConditions` -- the resolved, immutable view of a specification, built for
one mesh. It answers the four questions a solver needs and nothing more:
`constraints` (which DOFs are fixed), `load` (the right-hand side), `tangent(u)`,
and `residual(u)`. Above it the world is PDE-rich; below it, `DiscreteSystem` sees
only a matrix and a partition. The `Problem` is the narrow waist between them, so a
solve strategy never learns which PDE it is solving.

`LinearProblem` and `EnergyProblem` share that protocol, mirroring the `Form` /
`EnergyForm` split: the linear one is the special case whose tangent does not
depend on the state. Both own their constraints (resolved from the BC spec once,
here) -- which is what takes the re-resolve-after-remesh dance out of the solver:
a driver that remeshes just builds a new `Problem`.

Named PDEs survive as *factory functions* (`poisson`, `linear_elastic`, ...), not
dispatch classes: composing a typed operator with a typed load is what "solving
Poisson" means, so there is no PDE type to switch on.
"""
import copy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fem.boundary import BoundaryConditions
from fem.forms import (
    DiffusionForm, EnergyForm, Form, LaplacianForm, LinearElasticForm, LinearForm, MaskedMassForm,
    MassForm, ScaledForm,
)
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.regions import evaluate_field
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FieldValue, Operator


class Problem(Protocol):
    '''What a solve strategy consumes: constraints, a load, and residual/tangent.'''
    space: FunctionSpace

    @property
    def constraints(self) -> Constraints: ...

    @property
    def load(self) -> DofVector: ...

    def tangent(self, u: DofVector | None) -> Operator: ...

    def residual(self, u: DofVector) -> DofVector: ...


@runtime_checkable
class SupportsEnergy(Protocol):
    '''A `Problem` that can score a state by a scalar energy Π(u).

    The merit function a globalized `NewtonSolve` minimises: `EnergyProblem`
    supplies its potential energy, so the line search decreases the quantity being
    solved for. A problem without one (a `LinearProblem`) falls back to ½‖r‖².
    '''
    def energy(self, u: DofVector) -> float: ...


# -- load terms: the linear form L(v), assembled as a vector --------------------
#
# The volume source is a mass form over the domain used as a load: the mass matrix times
# the nodal source is the exact integral of its P1 interpolant. Boundary tractions are the
# same idea over the facets, but built per Neumann region (see `LinearProblem` and
# `boundary.NeumannContribution`) rather than one global boundary mass, so a traction
# stays on its own edge.


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f·v, with f a constant or a callable of position.'''
    field: FieldValue = None

    def vector(self, space: FunctionSpace) -> DofVector:
        # Sampled at the space's nodes, not the mesh vertices: a P2 space carries
        # edge-midpoint nodes the mesh does not, and the mass matrix is sized to them.
        values = evaluate_field(self.field, space.node_coords, space.n_components)
        return np.asarray(space.mass_matrix @ values.flatten()).flatten()


class LinearProblem:
    '''a(u, v) = L(v): a constant operator, a load, and Dirichlet constraints.'''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue | LinearForm = None,
        bc: BoundaryConditions | None = None,
    ) -> None:
        self.space = space
        self.operator = operator
        bc = bc if bc is not None else BoundaryConditions()
        self._resolved = bc.resolve(space.nodes, space.n_components)

        # A Robin condition contributes to both sides: κ∫_∂Ω_R u·v on the operator
        # and ∫_∂Ω_R g·v on the load, each the region-restricted boundary mass. The
        # operator half is kept apart from `operator`'s own so that a problem derived
        # under a new operator can re-apply it.
        self._robin_operator: Operator | None = None
        robin_load = np.zeros(space.n_dofs)
        for robin in self._resolved.robin:
            boundary_mass = space.assemble(MaskedMassForm(space.n_components, robin.facet_mask), boundary=True)
            term = robin.kappa * boundary_mass
            self._robin_operator = term if self._robin_operator is None else self._robin_operator + term
            robin_load = robin_load + np.asarray(boundary_mass @ robin.g.flatten()).flatten()

        # Each Neumann traction is integrated over its own region's facets (as the Robin
        # load is), so it stays on that edge instead of spreading onto a neighbour through
        # a shared corner -- which an unmasked global boundary mass would do.
        traction_load = np.zeros(space.n_dofs)
        for neumann in self._resolved.neumann:
            boundary_mass = space.assemble(
                MaskedMassForm(space.n_components, neumann.facet_mask), boundary=True)
            traction_load = traction_load + np.asarray(
                boundary_mass @ neumann.traction.flatten()).flatten()

        # Callers pass only the volume source; the BC resolution supplies the traction
        # terms above. The source is a field -- integrated as its P1 interpolant via the
        # cached mass matrix -- or a LinearForm sampled at the quadrature points, for a
        # source that varies within an element.
        if isinstance(source, LinearForm):
            volume_load = space.assemble_load(source)
        else:
            volume_load = Source(source).vector(space)
        self._b = volume_load + traction_load + robin_load
        # Assembled on first use, not here. Stating a problem is cheap; assembling
        # its operator is the expensive half, and a problem can be built without
        # ever being solved -- a topology optimization iteration derives its own
        # operator from a template whose own operator is never assembled.
        self._A: Operator | None = None

    def _assemble(self, operator: Form) -> Operator:
        A = self.space.assemble(operator)
        return A if self._robin_operator is None else A + self._robin_operator

    def with_operator(self, operator: Form) -> 'LinearProblem':
        '''The same problem stated with a different operator.

        Only the operator is reassembled. Which DOFs are constrained and what the
        load is follow from the boundary conditions and the source, neither of which
        the operator enters, so a driver re-solving under a rebuilt operator --
        a topology optimization iteration rescaling the stiffness -- keeps them
        rather than resolving the BCs and reassembling the load per solve.

        A new problem rather than a mutation: the two share the constraints and load
        they agree on, and nothing here writes to either.
        '''
        derived = copy.copy(self)
        derived.operator = operator
        # The copy carries this problem's assembled operator, which is precisely what
        # the derived one must not answer with. Dropping it is what makes the new
        # operator take effect; keeping it would hand back the old stiffness silently.
        derived._A = None
        return derived

    @property
    def constraints(self) -> Constraints:
        r = self._resolved
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def load(self) -> DofVector:
        return self._b

    def tangent(self, u: DofVector | None = None) -> Operator:
        # Assembled once, on the first call, and held: the operator is constant, so
        # a Newton loop or a time-stepper asking repeatedly pays for one assembly.
        if self._A is None:
            self._A = self._assemble(self.operator)
        return self._A

    def residual(self, u: DofVector) -> DofVector:
        return self.tangent() @ u - self._b


class EnergyProblem:
    '''∇Π(u) = 0: a nonlinear operator whose tangent depends on the state.

    The residual is the energy gradient and the tangent its Hessian, both from an
    `EnergyForm`. No external work term yet -- the load is zero, so a source is
    refused rather than silently dropped (as `EnergySolver` always has).
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: EnergyForm,
        bc: BoundaryConditions,
        source: FieldValue = None,
    ) -> None:
        if source is not None:
            raise NotImplementedError(
                'EnergyProblem has no external work term yet: a source would be '
                'silently dropped from the minimised energy. Use a LinearProblem '
                'for forced problems.'
            )
        self.space = space
        self.operator = operator
        self._resolved = bc.resolve(space.nodes, space.n_components)
        if self._resolved.robin:
            raise NotImplementedError(
                'EnergyProblem does not support Robin conditions: the energy path has '
                'no boundary term for them. Use a LinearProblem.'
            )

    @property
    def constraints(self) -> Constraints:
        r = self._resolved
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def load(self) -> DofVector:
        return np.zeros(self.space.n_dofs)

    def tangent(self, u: DofVector | None) -> Operator:
        # Unlike a LinearProblem's, this tangent genuinely depends on the state, so
        # the "state-independent" None a LinearSolve would pass is a category error.
        if u is None:
            raise ValueError('EnergyProblem has a state-dependent tangent; evaluate it at a u')
        return self.space.assemble_tangent(self.operator, u)

    def residual(self, u: DofVector) -> DofVector:
        return self.space.assemble_residual(self.operator, u)

    def energy(self, u: DofVector) -> float:
        return self.space.total_energy(self.operator, u)


# -- named PDE factories: compose a space, an operator, a load, and constraints --


def projection(mesh: Mesh, source: FieldValue, bc: BoundaryConditions | None = None) -> LinearProblem:
    '''L2 projection of `source` onto the P1 space (M u = M f).'''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, MassForm(space.n_components), source, bc)


def poisson(mesh: Mesh, source: FieldValue, bc: BoundaryConditions | None = None) -> LinearProblem:
    '''Poisson K u = b, the material-free Laplacian.'''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, LaplacianForm(), source, bc)


def diffusion(
    mesh: Mesh,
    coefficient: FieldValue,
    source: FieldValue | LinearForm = None,
    bc: BoundaryConditions | None = None,
) -> LinearProblem:
    '''Variable-coefficient diffusion -div(κ(x) grad u) = f.

    Poisson with a position-dependent coefficient κ, sampled at the quadrature
    points rather than assumed constant. Pass a `LinearForm` source to sample f at
    the quadrature points too; a plain field is integrated as its nodal interpolant.
    '''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, DiffusionForm(coefficient), source, bc)


def linear_elastic(
    mesh: Mesh,
    material: LinearElasticMaterial,
    bc: BoundaryConditions | None = None,
    source: FieldValue = None,
) -> LinearProblem:
    '''Small-strain linear elasticity; a vector field, one component per spatial dim.'''
    space = FunctionSpace(mesh, n_components=mesh.spatial_dim)
    return LinearProblem(space, LinearElasticForm(material), source, bc)


def heat(mesh: Mesh, source: FieldValue = None, bc: BoundaryConditions | None = None) -> LinearProblem:
    '''Transient heat: the same Laplacian operator Poisson uses, to be time-stepped.

    A heat problem is not a distinct operator -- it is Poisson's, integrated in
    time -- so this is `poisson` under another name, paired with a `ThetaMethod`.
    '''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, LaplacianForm(), source, bc)


def wave(mesh: Mesh, c: float, bc: BoundaryConditions | None = None, source: FieldValue = None) -> LinearProblem:
    '''Transient wave with speed `c`: the Laplacian scaled by c², to be Newmark-stepped.

    The wave speed lives in the operator (`ScaledForm(c², …)`), so the integrator sees
    only c²K and never learns `c`.
    '''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, ScaledForm(c**2, LaplacianForm()), source, bc)
