"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its physical parameters.
`operator` gives the `Form` a solve assembles, constant-tangent or not. `space` and
`problem` resolve the equation against a mesh and a BC spec, the two steps every
facade shares.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from fem.energies import SmallStrain, StVenantKirchhoff
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import (
    DiffusionForm, EnergyForm, Form, LaplacianForm, LinearElasticForm, LinearForm, MassForm,
    ScaledForm,
)
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem, Problem
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
    '''
    field: FieldShape = Scalar()

    def __init__(self, source: FieldValue | LinearForm = None, density: float = 1.0) -> None:
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')
        self.source = source
        self.density = density

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
            return LinearProblem(space, operator, self.source, bc, density=self.density)
        return Problem(space, operator, self.source, bc, density=self.density)

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
                 density: float = 1.0) -> None:
        super().__init__(source, density)
        self.coefficient = coefficient

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.coefficient)


class Wave(Equation):
    '''The wave operator c²K, to be stepped by a `NewmarkMethod` as M u'' + c²K u = b.

    The wave speed lives in the operator, so the integrator sees only c²K.
    '''

    def __init__(self, c: float, source: FieldValue | LinearForm = None,
                 density: float = 1.0) -> None:
        super().__init__(source, density)
        self.c = c

    def operator(self, space: FunctionSpace) -> Form:
        return ScaledForm(self.c**2, LaplacianForm())


class StrainMeasure(Enum):
    '''Which strain the elastic energy is built on: the kinematics axis.

    The material `W` is one function; the two paths differ only in the strain fed
    to it (see `fem.energies`). SMALL is the infinitesimal `ε = ½(∇u + ∇uᵀ)`, whose
    energy is quadratic and so has a constant stiffness; GREEN_LAGRANGE is the
    geometrically exact `S = ½(FᵀF − I)` (St-Venant–Kirchhoff), whose energy is not
    and so is solved by minimising it.
    '''
    SMALL = 'small'
    GREEN_LAGRANGE = 'green_lagrange'


class Elasticity(Equation):
    '''Elasticity with a selectable strain measure. `kinematics` is SMALL by
    default (infinitesimal strain, a constant stiffness, linear elasticity);
    GREEN_LAGRANGE selects the St-Venant–Kirchhoff model (an energy minimised by
    Newton). E may be a scalar or a per-element array (a SIMP density-scaled modulus)
    on the small-strain path.'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue | LinearForm = None,
        kinematics: StrainMeasure = StrainMeasure.SMALL,
        density: float = 1.0,
    ) -> None:
        super().__init__(source, density)
        self.E = E
        self.nu = nu
        self.kinematics = kinematics

    @property
    def material(self) -> LinearElasticMaterial:
        return LinearElasticMaterial(self.E, self.nu)

    def operator(self, space: FunctionSpace) -> Form:
        '''The form for this equation's kinematics: the small-strain stiffness
        (constant tangent) for SMALL, the St-Venant-Kirchhoff `EnergyForm` for
        GREEN_LAGRANGE.'''
        if self.kinematics is StrainMeasure.SMALL:
            return LinearElasticForm(self.material)
        return EnergyForm(self.energy_density())

    def energy_density(self) -> StVenantKirchhoff:
        '''The stored-energy density for this equation's kinematics.

        Same `W`, different strain measure: SmallStrain subclasses
        StVenantKirchhoff and overrides only the strain, so both satisfy the
        return type.
        '''
        # E may be per-element (a SIMP density-scaled modulus),
        # but a density carries one pair of Lame parameters for the whole mesh,
        # and an array lamb broadcasts wrongly against the constant d2W/dS2.
        if not isinstance(self.E, int | float):
            raise NotImplementedError(
                'an energy density needs a scalar Youngs modulus, got a per-element '
                'array; a density-scaled modulus solves through the linear operator.'
            )
        density = {
            StrainMeasure.SMALL: SmallStrain,
            StrainMeasure.GREEN_LAGRANGE: StVenantKirchhoff,
        }[self.kinematics]
        return density(self.E, self.nu)


LinearElastic = Elasticity
'''Alias of `Elasticity`.'''
