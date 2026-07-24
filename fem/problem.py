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
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fem.boundary import BoundaryConditions
from fem.forms import EnergyForm, Form, LaplacianForm, LinearElasticForm, MassForm, Scaled
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.regions import evaluate_field
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FieldValue, FloatArray, Operator


class Problem(Protocol):
    '''What a solve strategy consumes: constraints, a load, and residual/tangent.'''
    space: FunctionSpace

    @property
    def constraints(self) -> Constraints: ...

    @property
    def load(self) -> DofVector: ...

    def tangent(self, u: DofVector | None) -> Operator: ...

    def residual(self, u: DofVector) -> DofVector: ...


# -- load terms: the linear form L(v), assembled as a vector --------------------
#
# Both are "a mass form over a domain, used as a load operator": the volume mass
# matrix applied to the nodal source is the exact integral of the source's P1
# interpolant, and the boundary mass matrix does the same over the facets. They
# sum, which is the composition the old assemble_everything did inline as
# `M @ f + M_b @ neumann`.


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f·v, with f a constant or a callable of position.'''
    field: FieldValue = None

    def vector(self, space: FunctionSpace) -> DofVector:
        values = evaluate_field(self.field, space.mesh.vertices, space.n_components)
        return np.asarray(space.mass_matrix @ values.flatten()).flatten()


@dataclass(frozen=True)
class Traction:
    '''Boundary load L(v) = ∫ g·v over the facets, from nodal traction values.'''
    nodal: FloatArray

    def vector(self, space: FunctionSpace) -> DofVector:
        return np.asarray(space.boundary_mass_matrix @ np.asarray(self.nodal).flatten()).flatten()


class LinearProblem:
    '''a(u, v) = L(v): a constant operator, a load, and Dirichlet constraints.'''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue = None,
        bc: BoundaryConditions | None = None,
    ) -> None:
        self.space = space
        self.operator = operator
        bc = bc if bc is not None else BoundaryConditions()
        self._resolved = bc.resolve(space.mesh, space.n_components)
        self._A = space.assemble(operator)
        # The load folds the Neumann contribution in as a boundary traction term,
        # so callers pass only the volume source; the BC resolution supplies the rest.
        self._b = Source(source).vector(space) + Traction(self._resolved.neumann_load).vector(space)

    @property
    def constraints(self) -> Constraints:
        r = self._resolved
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def load(self) -> DofVector:
        return self._b

    def tangent(self, u: DofVector | None = None) -> Operator:
        return self._A

    def residual(self, u: DofVector) -> DofVector:
        return self._A @ u - self._b


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
        self._resolved = bc.resolve(space.mesh, space.n_components)

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

    The wave speed lives in the operator (`Scaled(c², …)`), so the integrator sees
    only c²K and never learns `c`.
    '''
    space = FunctionSpace(mesh, n_components=1)
    return LinearProblem(space, Scaled(c**2, LaplacianForm()), source, bc)
