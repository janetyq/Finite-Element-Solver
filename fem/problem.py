"""The `Problem`: the assembly-ready statement a solve strategy consumes.

A `Problem` is the resolved, immutable view of a physics composition, built for one
mesh. It answers the four questions a solver needs: `constraints` (which DOFs are
fixed), `load` (the right-hand side), `tangent(u)`, and `residual(u)`. Below it,
`DiscreteSystem` sees only a matrix and a partition, so a solve strategy never learns
which PDE it is solving. After the solve, `solution(u)` packages the DOF vector as
the typed `Solution` its physics recovers (stress for elasticity, flux for Poisson).

`LinearProblem` and `EnergyProblem` share that protocol, mirroring the `Form` /
`EnergyForm` split: the linear one has a state-independent tangent. Both own their
constraints, resolved from the BC spec once; a driver that remeshes builds a new
`Problem`. Named PDEs are `Equation`s (`fem.equations`), whose `problem` builds one.
"""
import copy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fem.boundary import BoundaryConditions, ResolvedBC
from fem.forms import (
    EnergyForm, Form, HasNearNullSpace, LinearForm, MaskedMassForm, NamesDerivedField,
    RecoversElasticFields,
)
from fem.regions import evaluate_field
from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FieldValue, FloatArray, Operator


class Problem(Protocol):
    '''What a solve strategy consumes: constraints, a load, and residual/tangent.

    `bc` is the mesh-independent spec the constraints were resolved from, and
    `resolved` that resolution on this space (the estimators read its Neumann load).
    `operator` and `source` are the physics and the volume load the problem was
    composed from; the estimators read the recoverable flux off the operator.
    '''
    space: FunctionSpace
    bc: BoundaryConditions

    @property
    def operator(self) -> 'Form | EnergyForm': ...

    @property
    def source(self) -> 'FieldValue | LinearForm | Source': ...

    @property
    def resolved(self) -> ResolvedBC: ...

    @property
    def constraints(self) -> Constraints: ...

    @property
    def load(self) -> DofVector: ...

    def tangent(self, u: DofVector | None) -> Operator: ...

    def residual(self, u: DofVector) -> DofVector: ...

    def solution(self, u: DofVector) -> FieldSolution:
        '''Package a solved DOF vector as the typed `Solution` this physics recovers.'''
        ...

    def near_null_space(self) -> FloatArray | None:
        '''The operator's AMG near-kernel over all DOFs (see `HasNearNullSpace`), or None.

        `LinearSolve` hands it to an `IterativeBackend`, so an elastic solve composed
        by hand converges as well as one through the facade.
        '''
        ...


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
    '''Volume load L(v) = ∫ f·v integrated as f's nodal interpolant, f a constant or a
    callable of position. Pass one to `LinearProblem` to ask for that path explicitly;
    a bare callable is sampled at the quadrature points instead.'''
    field: FieldValue = None

    def vector(self, space: FunctionSpace) -> DofVector:
        # Sampled at the space's nodes, not the mesh vertices: a P2 space carries
        # edge-midpoint nodes the mesh does not, and the mass matrix is sized to them.
        values = evaluate_field(self.field, space.node_coords, space.n_components)
        return np.asarray(space.mass_matrix @ values.flatten()).flatten()


class LinearProblem:
    '''a(u, v) = L(v): a constant operator, a load, and Dirichlet constraints.

    `source` is kept as given (a field, a `LinearForm`, or a `Source`) beside the
    assembled load, so a residual estimator can read the pointwise source it needs.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue | LinearForm | Source = None,
        bc: BoundaryConditions | None = None,
    ) -> None:
        self.space = space
        self.operator = operator
        self.source = source
        self.bc = bc if bc is not None else BoundaryConditions()
        self._resolved = self.bc.resolve(space.nodes, space.n_components)

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
        # a shared corner, which an unmasked global boundary mass would do.
        traction_load = np.zeros(space.n_dofs)
        for neumann in self._resolved.neumann:
            boundary_mass = space.assemble(
                MaskedMassForm(space.n_components, neumann.facet_mask), boundary=True)
            traction_load = traction_load + np.asarray(
                boundary_mass @ neumann.traction.flatten()).flatten()

        # Callers pass only the volume source; the BC resolution supplies the traction
        # terms above. A callable source is sampled at the quadrature points (as a
        # LinearForm), which captures variation within an element; a constant or a
        # nodal array is integrated as its interpolant through the cached mass matrix.
        if callable(source) and not isinstance(source, (LinearForm, Source)):
            source = LinearForm(source, n_components=space.n_components)
            self.source = source
        if isinstance(source, LinearForm):
            volume_load = space.assemble_load(source)
        elif isinstance(source, Source):
            volume_load = source.vector(space)
        else:
            volume_load = Source(source).vector(space)
        self._b = volume_load + traction_load + robin_load
        # Assembled on first use, not here. Stating a problem is cheap; assembling
        # its operator is the expensive half, and a problem can be built without
        # ever being solved: a topology optimization iteration derives its own
        # operator from a template whose own operator is never assembled.
        self._A: Operator | None = None

    def _assemble(self, operator: Form) -> Operator:
        A = self.space.assemble(operator)
        return A if self._robin_operator is None else A + self._robin_operator

    def with_operator(self, operator: Form) -> 'LinearProblem':
        '''The same problem stated with a different operator.

        Only the operator is reassembled. Which DOFs are constrained and what the
        load is follow from the boundary conditions and the source, neither of which
        the operator enters, so a driver re-solving under a rebuilt operator
        (a topology optimization iteration rescaling the stiffness) keeps them
        rather than resolving the BCs and reassembling the load per solve.

        A new problem rather than a mutation: the two share the constraints and load
        they agree on, and nothing here writes to either.
        '''
        derived = copy.copy(self)
        derived.operator = operator
        # The copy carries this problem's assembled operator, which is precisely what
        # the derived one must not answer with.
        derived._A = None
        return derived

    @property
    def resolved(self) -> ResolvedBC:
        return self._resolved

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

    def solution(self, u: DofVector) -> FieldSolution:
        '''An `ElasticSolution` for an operator that recovers stress, a
        `ScalarFieldSolution` for one naming a flux (Poisson's gradient), else a bare
        `FieldSolution` (a projection).'''
        space = self.space
        if isinstance(self.operator, RecoversElasticFields):
            return ElasticSolution.from_solve(space, u, self.operator)
        if isinstance(self.operator, NamesDerivedField):
            return ScalarFieldSolution.from_solve(space, u)
        return FieldSolution(space, u)

    def near_null_space(self) -> FloatArray | None:
        if isinstance(self.operator, HasNearNullSpace):
            return self.operator.near_null_space(self.space)
        return None


class EnergyProblem:
    '''∇Π(u) = 0: a nonlinear operator whose tangent depends on the state.

    The residual is the energy gradient and the tangent its Hessian, both from an
    `EnergyForm`. No external work term yet: the load is zero, so a source is refused.
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
        self.source = None
        self.bc = bc
        self._resolved = bc.resolve(space.nodes, space.n_components)
        if self._resolved.robin:
            raise NotImplementedError(
                'EnergyProblem does not support Robin conditions: the energy path has '
                'no boundary term for them. Use a LinearProblem.'
            )

    @property
    def resolved(self) -> ResolvedBC:
        return self._resolved

    @property
    def constraints(self) -> Constraints:
        r = self._resolved
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def load(self) -> DofVector:
        return np.zeros(self.space.n_dofs)

    def tangent(self, u: DofVector | None) -> Operator:
        # Unlike a LinearProblem's, this tangent depends on the state, so
        # the "state-independent" None a LinearSolve would pass is a category error.
        if u is None:
            raise ValueError('EnergyProblem has a state-dependent tangent; evaluate it at a u')
        return self.space.assemble_tangent(self.operator, u)

    def residual(self, u: DofVector) -> DofVector:
        return self.space.assemble_residual(self.operator, u)

    def energy(self, u: DofVector) -> float:
        return self.space.total_energy(self.operator, u)

    def solution(self, u: DofVector) -> ElasticSolution:
        # The energy form recovers Cauchy stress from the same derivative chain Newton
        # used, so the nonlinear path reports the stress state the linear one does.
        return ElasticSolution.from_solve(self.space, u, self.operator)

    def near_null_space(self) -> None:
        # The tangent is indefinite away from a minimum, so its iterative solve is
        # MINRES, which takes no near-kernel.
        return None
