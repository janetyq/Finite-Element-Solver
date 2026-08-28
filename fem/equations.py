"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its physical parameters.
`operator` gives the `Form` a solve assembles, constant-tangent or not. `space` and
`problem` resolve the equation against a mesh and a BC spec, the two steps every
facade shares.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fem.energies import SmallStrain, StVenantKirchhoff
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import (
    DiffusionForm, EnergyDensity, EnergyForm, Form, LaplacianForm, LinearElasticForm, LinearForm,
    MassForm, ScaledForm,
)
from fem.materials import LinearElasticMaterial
from fem.loads import Load
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.space import FunctionSpace
from fem.typing import ElementField, FieldValue

from fem.mesh.mesh import Mesh

if TYPE_CHECKING:
    from fem.boundary import BoundaryConditions
    from fem.elements import Element


class Equation:
    '''Base class for a PDE to solve.

    An Equation says what to solve and carries the physical parameters; a solver
    owns how. A transient problem is a steady operator paired with a time integrator,
    so `Poisson` under a `ThetaMethod` is the heat equation and `Wave` under a
    `NewmarkMethod` the wave equation.

    `field` says what kind of value the unknown takes; the DOFs per node follow from
    it and the mesh. `source` is the PDE's right-hand side f (a body force for
    elasticity): a constant, a callable of position, or a `LinearForm` sampled at the
    quadrature points. `density` is the coefficient on the time-derivative term (mass
    density for elasticity and the wave operator, volumetric heat capacity for
    diffusion), read by the integrators and modal analysis through `Problem.mass`.
    `damping` is the `RayleighDamping` a second-order integrator applies, and `loads`
    are load terms beyond the source (a `PointLoad`).
    '''
    field: FieldShape = Scalar()

    def __init__(
        self,
        source: FieldValue | LinearForm = None,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')
        self.source = source
        self.density = density
        self.damping = damping
        self.loads = tuple(loads)

    def space(self, mesh: Mesh, element_type: type[Element] | None = None) -> FunctionSpace:
        '''The discretization of this equation's unknown on `mesh`.

        The component count follows from the equation's field and the mesh, so a space
        that disagrees with the equation it solves is not constructible here.
        `element_type` None is the linear element for the mesh.
        '''
        n_components = self.field.components_for(mesh.spatial_dim)
        return FunctionSpace(mesh, element_type, n_components=n_components)

    def problem(
        self,
        domain: Mesh | FunctionSpace,
        bc: BoundaryConditions | None = None,
        element_type: type[Element] | None = None,
    ) -> Problem:
        '''The composition on `domain`: this operator, this source, `bc`. A
        `LinearProblem` when the operator has a constant tangent.

        `domain` is a `Mesh`, discretized through `space(mesh, element_type)`, or a
        `FunctionSpace` used as it is. `element_type` applies to a mesh only.
        '''
        if isinstance(domain, FunctionSpace):
            if element_type is not None:
                raise ValueError('element_type applies to a Mesh; the FunctionSpace already has one')
            space = domain
        else:
            space = self.space(domain, element_type)
        operator = self.operator(space)
        if operator.constant_tangent:
            return LinearProblem(space, operator, self.source, bc, density=self.density,
                                 loads=self.loads, damping=self.damping)
        return Problem(space, operator, self.source, bc, density=self.density,
                       loads=self.loads, damping=self.damping)

    def operator(self, space: FunctionSpace) -> Form:
        '''The form a solve assembles for this equation on `space`.'''
        raise NotImplementedError(f'{type(self).__name__} names no operator')


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b).'''

    def operator(self, space: FunctionSpace) -> Form:
        return MassForm(space.n_components)


class Poisson(Equation):
    '''Poisson equation (K u = b); under a `ThetaMethod`, the heat equation.'''

    def operator(self, space: FunctionSpace) -> Form:
        return LaplacianForm()


class Diffusion(Equation):
    '''Variable-coefficient diffusion -div(κ(x) grad u) = f.

    `coefficient` is κ, a constant or a callable of position, sampled at the
    quadrature points.
    '''

    def __init__(self, coefficient: FieldValue, source: FieldValue | LinearForm = None,
                 density: float = 1.0, damping: RayleighDamping | None = None,
                 loads: tuple[Load, ...] = ()) -> None:
        super().__init__(source, density, damping, loads)
        self.coefficient = coefficient

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.coefficient)


class Wave(Equation):
    '''The wave operator c²K, to be stepped by a `NewmarkMethod` as M u'' + c²K u = b.

    The wave speed lives in the operator, so the integrator sees only c²K.
    '''

    def __init__(self, c: float, source: FieldValue | LinearForm = None,
                 density: float = 1.0, damping: RayleighDamping | None = None,
                 loads: tuple[Load, ...] = ()) -> None:
        super().__init__(source, density, damping, loads)
        self.c = c

    def operator(self, space: FunctionSpace) -> Form:
        return ScaledForm(self.c**2, LaplacianForm())


class Elasticity(Equation):
    '''Base of the elastic equations: a vector unknown with Young's modulus `E`,
    Poisson's ratio `nu`, and a mass `density`. `LinearElastic` and
    `FiniteStrainElastic` name the two models.'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue | LinearForm = None,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(source, density, damping, loads)
        self.E = E
        self.nu = nu

    @property
    def material(self) -> LinearElasticMaterial:
        return LinearElasticMaterial(self.E, self.nu)

    def energy_density(self) -> EnergyDensity:
        '''The stored-energy density `W` of this model, for an `EnergyForm`.'''
        raise NotImplementedError

    def _scalar_modulus(self) -> float:
        # A density carries one pair of Lame parameters for the whole mesh; an array
        # lamb would broadcast wrongly against the constant d2W/dS2.
        if not isinstance(self.E, int | float):
            raise NotImplementedError(
                'an energy density needs a scalar Youngs modulus, got a per-element '
                'array; a density-scaled modulus solves through the linear operator.'
            )
        return float(self.E)


class LinearElastic(Elasticity):
    '''Small-strain linear elasticity: the infinitesimal strain `ε = ½(∇u + ∇uᵀ)`
    under Hooke's law, a constant stiffness solved in one linear solve. `E` may be a
    scalar or a per-element array (a SIMP density-scaled modulus).'''

    def operator(self, space: FunctionSpace) -> Form:
        return LinearElasticForm(self.material)

    def energy_density(self) -> SmallStrain:
        '''The quadratic energy the stiffness is the Hessian of.'''
        return SmallStrain(self._scalar_modulus(), self.nu)


class FiniteStrainElastic(Elasticity):
    '''Finite-strain elasticity on the Green-Lagrange strain `S = ½(FᵀF − I)`, a
    stored energy minimised by Newton. `law` builds the energy density from `(E, nu)`;
    St-Venant-Kirchhoff by default.'''

    def __init__(
        self,
        E: float,
        nu: float,
        source: FieldValue | LinearForm = None,
        density: float = 1.0,
        law: Callable[[float, float], EnergyDensity] = StVenantKirchhoff,
        damping: RayleighDamping | None = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(E, nu, source, density, damping, loads)
        self.law = law

    def operator(self, space: FunctionSpace) -> Form:
        return EnergyForm(self.energy_density())

    def energy_density(self) -> EnergyDensity:
        return self.law(self._scalar_modulus(), self.nu)
