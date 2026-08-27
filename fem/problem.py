"""The `Problem`: the assembly-ready statement a solve strategy consumes.

A `Problem` is the resolved view of a physics composition, built for one mesh: a
space, an operator (`Form`), a volume source, and boundary conditions. It answers
the questions a solver needs: `constraints` (which DOFs are fixed), `load` (the
right-hand side), `residual(u)`, `tangent(u)`, and, where the operator has an energy,
`energy(u)`. Below it, `DiscreteSystem` sees only a matrix and a partition, so a
solve strategy never learns which PDE it is solving. After the solve, `solution(u)`
packages the DOF vector as the typed `Solution` its physics recovers (stress for
elasticity, flux for Poisson).

The residual is composed from three terms, each present in the energy, the residual,
and the tangent alike:

    term    energy         residual      tangent
    form    Π_form(u)      R_form(u)     ∂R_form/∂u
    Robin   ½ κ uᵀ R u     κ R u         κ R
    load    -fᵀ u          -f            0

`LinearProblem` is the case whose operator has a constant tangent (a `BilinearForm`):
the matrix is assembled once and held, and the residual is affine. Everything that
needs one fixed operator (a direct solve, the integrators, an eigenproblem, SIMP)
requires it. Both own their constraints, resolved from the BC spec once; a driver
that remeshes builds a new `Problem`. Named PDEs are `Equation`s (`fem.equations`),
whose `problem` builds one of these.
"""
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem.boundary import BoundaryConditions, ResolvedBC
from fem.forms import BilinearForm, Form, LinearForm, MaskedMassForm, RecoversElasticFields
from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FieldValue, FloatArray, Operator

if TYPE_CHECKING:
    from fem.backends import Backend
    from fem.solve import SolveStrategy


# -- load terms: the linear form L(v), assembled as a vector --------------------
#
# The volume source is a mass form over the domain used as a load: the mass matrix times
# the nodal source is the exact integral of its P1 interpolant. Boundary tractions are the
# same idea over the facets, but built per Neumann region (see `Problem` and
# `boundary.NeumannContribution`) rather than one global boundary mass, so a traction
# stays on its own edge.


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f·v integrated as f's nodal interpolant, f a constant or a
    callable of position. Pass one to `Problem` to ask for that path explicitly;
    a bare callable is sampled at the quadrature points instead.'''
    field: FieldValue = None

    def vector(self, space: FunctionSpace) -> DofVector:
        return np.asarray(space.mass_matrix @ space.interpolate(self.field)).flatten()


class Problem:
    '''R(u) = 0: an operator, a load, and boundary conditions on one space.

    `source` is kept as given (a field, a `LinearForm`, or a `Source`) beside the
    assembled load, so a residual estimator can read the pointwise source it needs.
    `bc` is the mesh-independent spec the constraints were resolved from, and
    `resolved` that resolution on this space.
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
        # A constant tangent is assembled on first use, not here. Stating a problem
        # is cheap; assembling its operator is the expensive half, and a problem can
        # be built without ever being solved: a topology optimization iteration
        # derives its own operator from a template whose own operator is never assembled.
        self._A: Operator | None = None

    @property
    def is_linear(self) -> bool:
        '''Whether the operator's tangent is constant.'''
        return self.operator.constant_tangent

    @property
    def has_energy(self) -> bool:
        '''Whether the residual is the gradient of an energy.

        Only the line search reads this: with an energy it scores a step by Π(u),
        without one by ½‖r‖². The Newton iteration itself is the same either way.
        '''
        return self.operator.has_energy

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

    def with_operator(self, operator: Form) -> 'Problem':
        '''The same problem stated with a different operator.

        Which DOFs are constrained and what the load is follow from the boundary
        conditions and the source, neither of which the operator enters, so a driver
        re-solving under a rebuilt operator (a topology optimization iteration
        rescaling the stiffness) keeps them rather than resolving the BCs and
        reassembling the load per solve.

        A new problem rather than a mutation: the two share the constraints and load
        they agree on, and nothing here writes to either.
        '''
        derived = copy.copy(self)
        derived.operator = operator
        # The copy carries this problem's assembled operator, which is precisely what
        # the derived one must not answer with.
        derived._A = None
        return derived

    def _with_robin(self, operator: Operator) -> Operator:
        return operator if self._robin_operator is None else operator + self._robin_operator

    def tangent(self, u: DofVector | None = None) -> Operator:
        '''∂R/∂u at `u`; with `u` omitted, the constant tangent of a linear problem.'''
        if self.is_linear:
            # Assembled once, on the first call, and held: the operator is constant,
            # so a Newton loop or a time-stepper asking repeatedly pays for one assembly.
            if self._A is None:
                assert isinstance(self.operator, BilinearForm)
                self._A = self._with_robin(self.space.assemble(self.operator))
            return self._A
        if u is None:
            raise ValueError(
                f'{type(self.operator).__name__} has a state-dependent tangent; evaluate '
                'it at a u. A solve that needs one fixed operator needs a LinearProblem.'
            )
        return self._with_robin(self.space.assemble_tangent(self.operator, u))

    def internal_residual(self, u: DofVector) -> DofVector:
        '''The internal force at `u`, the residual without the load: the operator's
        residual plus the Robin term. Kept apart from `load` so a strategy can scale
        the load against it.'''
        if self.is_linear:
            return self.tangent() @ u
        residual = self.space.assemble_residual(self.operator, u)
        if self._robin_operator is not None:
            residual = residual + self._robin_operator @ u
        return residual

    def residual(self, u: DofVector) -> DofVector:
        return self.internal_residual(u) - self._b

    def energy(self, u: DofVector) -> float:
        '''The potential Π(u) whose gradient is `residual(u)`, for an operator with an energy.'''
        if not self.has_energy:
            raise TypeError(f'{type(self.operator).__name__} has no energy to minimise')
        if self.is_linear:
            return float(0.5 * u @ (self.tangent() @ u) - self._b @ u)
        energy = self.space.total_energy(self.operator, u)
        if self._robin_operator is not None:
            energy += 0.5 * float(u @ (self._robin_operator @ u))
        return float(energy - self._b @ u)

    def solution(self, u: DofVector) -> FieldSolution:
        '''An `ElasticSolution` for an operator that recovers stress, a
        `ScalarFieldSolution` for one naming a flux (Poisson's gradient), else a bare
        `FieldSolution` (a projection).'''
        space = self.space
        if isinstance(self.operator, RecoversElasticFields):
            return ElasticSolution.from_solve(space, u, self.operator)
        if self.operator.derived_field() is not None:
            return ScalarFieldSolution.from_solve(space, u)
        return FieldSolution(space, u)

    def solve(
        self,
        strategy: 'SolveStrategy | None' = None,
        backend: 'Backend | None' = None,
        u0: DofVector | None = None,
    ) -> FieldSolution:
        '''Solve and package the result as the typed `Solution` for this operator.

        `strategy` None is `default_strategy`: `LinearSolve` for a constant tangent,
        line-searched `NewtonSolve` otherwise, over `backend`. A strategy carries its
        own backend, so the two are not given together. `u0` seeds an iterative
        strategy.
        '''
        if strategy is not None and backend is not None:
            raise ValueError('a strategy carries its own backend; pass one or the other')
        if strategy is None:
            from fem.solve import default_strategy
            strategy = default_strategy(self, backend)
        return self.solution(strategy.solve(self, u0))

    def near_null_space(self) -> FloatArray | None:
        '''The operator's AMG near-kernel over all DOFs, or None.

        `LinearSolve` hands it to an `IterativeBackend`, so an elastic solve composed
        by hand converges as well as one through the facade.
        '''
        return self.operator.near_null_space(self.space)


class LinearProblem(Problem):
    '''a(u, v) = L(v): a `Problem` whose operator has a constant tangent.

    The type every consumer that needs one fixed operator asks for. `tangent()` with
    no state is the assembled matrix, held after the first call.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue | LinearForm | Source = None,
        bc: BoundaryConditions | None = None,
    ) -> None:
        if not operator.constant_tangent:
            raise TypeError(
                f'{type(operator).__name__} has a state-dependent tangent; state it as a '
                'Problem and solve it with NewtonSolve.'
            )
        super().__init__(space, operator, source, bc)

    def with_operator(self, operator: Form) -> 'LinearProblem':
        if not operator.constant_tangent:
            raise TypeError(f'{type(operator).__name__} has a state-dependent tangent')
        derived = super().with_operator(operator)
        assert isinstance(derived, LinearProblem)
        return derived
