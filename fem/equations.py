"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its physical parameters.
Each subclass answers for both assembly paths: `operator` gives the bilinear form the
linear path assembles and `energy_density` the strain-energy density the nonlinear path
differentiates. `space` and `problem` resolve the equation against a mesh and a BC
spec, the two steps every facade shares.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from fem.energies import SmallStrain, StVenantKirchhoff
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import (
    DiffusionForm, EnergyDensity, Form, LaplacianForm, LinearElasticForm, LinearForm, MassForm,
    ScaledForm,
)
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.space import FunctionSpace
from fem.typing import ElementField, FieldValue

if TYPE_CHECKING:
    from fem.boundary import BoundaryConditions
    from fem.elements import Element
    from fem.mesh.mesh import Mesh


class Equation:
    '''Base class for a PDE to solve.

    An Equation says what to solve and carries the physical parameters; a solver
    owns how. A transient problem is a steady operator paired with a time integrator,
    so `Poisson` under a `ThetaMethod` is the heat equation and `Wave` under a
    `NewmarkMethod` the wave equation.

    `field` says what kind of value the unknown takes; the DOFs per node follow from
    it and the mesh. `source` is the PDE's right-hand side f (a body force for
    elasticity): a constant, a callable of position, or a `LinearForm` sampled at the
    quadrature points.
    '''
    field: FieldShape = Scalar()

    def __init__(self, source: FieldValue | LinearForm = None) -> None:
        self.source = source

    def space(self, mesh: Mesh, element_type: type[Element] | None = None) -> FunctionSpace:
        '''The discretization of this equation's unknown on `mesh`.

        The component count follows from the equation's field and the mesh, so a space
        that disagrees with the equation it solves is not constructible here.
        `element_type` None is the linear element for the mesh.
        '''
        n_components = self.field.components_for(mesh.spatial_dim)
        return FunctionSpace(mesh, element_type, n_components=n_components)

    def problem(self, space: FunctionSpace, bc: BoundaryConditions | None = None) -> LinearProblem:
        '''The linear composition on `space`: this operator, this source, `bc`.'''
        return LinearProblem(space, self.operator(space), self.source, bc)

    def operator(self, space: FunctionSpace) -> Form:
        '''The bilinear form a linear solve assembles for this equation on `space`.

        The scalar diffusion family shares the material-free Laplacian, so it is the
        base answer; subclasses that mean something else override.
        '''
        return LaplacianForm()

    def energy_density(self) -> EnergyDensity:
        '''The strain-energy density a nonlinear solve differentiates.

        Only defined for equations with a stored-energy formulation; the scalar
        family has none, so the base raises rather than returning a stand-in.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} has no strain-energy density to minimise; '
            'solve it through its operator.'
        )


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b).'''

    def operator(self, space: FunctionSpace) -> Form:
        return MassForm(space.n_components)


class Poisson(Equation):
    '''Poisson equation (K u = b); under a `ThetaMethod`, the heat equation.'''


class Diffusion(Equation):
    '''Variable-coefficient diffusion -div(κ(x) grad u) = f.

    `coefficient` is κ, a constant or a callable of position, sampled at the
    quadrature points.
    '''

    def __init__(self, coefficient: FieldValue, source: FieldValue | LinearForm = None) -> None:
        super().__init__(source)
        self.coefficient = coefficient

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.coefficient)


class Wave(Equation):
    '''The wave operator c²K, to be stepped by a `NewmarkMethod` as M u'' + c²K u = b.

    The wave speed lives in the operator, so the integrator sees only c²K.
    '''

    def __init__(self, c: float, source: FieldValue | LinearForm = None) -> None:
        super().__init__(source)
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


class LinearElastic(Equation):
    '''Elasticity with a selectable strain measure. `kinematics` is SMALL by
    default (infinitesimal strain, a linear solve); GREEN_LAGRANGE selects the
    St-Venant–Kirchhoff model (an energy minimisation). E may be a scalar or a
    per-element array (a SIMP density-scaled modulus).'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue | LinearForm = None,
        kinematics: StrainMeasure = StrainMeasure.SMALL,
    ) -> None:
        super().__init__(source)
        self.E = E
        self.nu = nu
        self.kinematics = kinematics

    @property
    def material(self) -> LinearElasticMaterial:
        return LinearElasticMaterial(self.E, self.nu)

    def operator(self, space: FunctionSpace) -> Form:
        '''The small-strain stiffness form, built from this equation's material.

        The bilinear form exists only for the small-strain measure: a
        Green-Lagrange energy is not quadratic, so it has no constant stiffness.
        '''
        if self.kinematics is not StrainMeasure.SMALL:
            raise NotImplementedError(
                f'{self.kinematics.name} kinematics has no constant stiffness; '
                'solve it by minimising its energy (EnergyProblem).'
            )
        return LinearElasticForm(self.material)

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

