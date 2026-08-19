"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its genuinely physical
parameters. It sits in the physics layer next to `fem.forms` and `fem.energies`,
not with the solvers: `Solver` and `EnergySolver` both consume equations, so
neither module owns them.

Each subclass answers "what physics do I mean?" for both assembly paths:
`operator` gives the bilinear form the linear path assembles, `energy_density`
gives the strain-energy density the nonlinear path differentiates. A solver picks
a path; it does not decide what material an equation implies.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from fem.energies import SmallStrain, StVenantKirchhoff
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import EnergyDensity, Form, LaplacianForm, LinearElasticForm, MassForm
from fem.materials import LinearElasticMaterial
from fem.postprocess import GradientField, StressField
from fem.typing import ElementField, FieldValue

if TYPE_CHECKING:
    from fem.postprocess import DerivedField


class Equation:
    '''Base class for a PDE to solve.

    An Equation is typed data: it says what to solve and carries the physical
    parameters, while a solve strategy owns how (the same equation, e.g.
    LinearElastic, may be handled by several solvers). Transient problems are not
    equation types: heat and wave are a steady operator paired with a time
    integrator (see fem.problem.heat / .wave and fem.integrators).

    `field` says what kind of value the unknown takes; the DOFs per node follow
    from it and the mesh, so no subclass writes the count down. Not a ClassVar:
    a system of k equations would carry its count as constructor data.

    `source` is the PDE's right-hand side f (a body force for elasticity), given
    as a constant or a callable of position. It lives here rather than on
    BoundaryConditions because it is data of the equation, not of the boundary.
    '''
    field: FieldShape = Scalar()

    def __init__(self, source: FieldValue = None) -> None:
        self.source = source

    def operator(self, n_components: int) -> Form:
        '''The bilinear form a linear solve assembles for this equation.

        The scalar diffusion family (Poisson, and the Laplacian behind the heat
        and wave problems) shares the material-free Laplacian, so it is the base
        answer; subclasses that mean something else override.
        '''
        return LaplacianForm()

    def energy_density(self) -> EnergyDensity:
        '''The strain-energy density a nonlinear solve differentiates.

        Only defined for equations with a stored-energy formulation; the scalar
        family has none, so the base raises rather than returning a stand-in.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} has no strain-energy density, so it cannot be '
            'solved by minimising an energy. Use Solver.'
        )

    def derived_field(self) -> 'DerivedField | None':
        '''The derived field this equation recovers, or None if it has none.

        Names the physics (Poisson's gradient, elasticity's stress) that post-processing
        recovers to nodes and `fem.estimators` builds a per-element indicator from, the
        post-processing analogue of `operator`. None where no such field is defined (a
        pure projection), so a caller dispatches on it without catching.
        '''
        return None


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b).'''

    def operator(self, n_components: int) -> Form:
        return MassForm(n_components)


class Poisson(Equation):
    '''Poisson equation (K u = b).'''

    def derived_field(self) -> 'DerivedField':
        '''The diffusion flux to recover and estimate from: the field gradient ∇u.'''
        return GradientField()


class StrainMeasure(Enum):
    '''Which strain the elastic energy is built on: the kinematics axis.

    The material `W` is one function; the two paths differ only in the strain fed
    to it (see `fem.energies`). SMALL is the infinitesimal `ε = ½(∇u + ∇uᵀ)`,
    solved directly by `Solver`; GREEN_LAGRANGE is the geometrically exact
    `S = ½(FᵀF − I)` (St-Venant–Kirchhoff), which only `EnergySolver` can solve
    because its energy is not quadratic.
    '''
    SMALL = 'small'
    GREEN_LAGRANGE = 'green_lagrange'


class LinearElastic(Equation):
    '''Elasticity with a selectable strain measure. `kinematics` is SMALL by
    default (infinitesimal strain, the linear `Solver` path); GREEN_LAGRANGE
    selects the St-Venant–Kirchhoff model, which needs `EnergySolver`. E may be a
    scalar or a per-element array (TopologyOptimizer sets a density-scaled modulus).'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue = None,
        kinematics: StrainMeasure = StrainMeasure.SMALL,
    ) -> None:
        super().__init__(source)
        self.E = E
        self.nu = nu
        self.kinematics = kinematics

    def operator(self, n_components: int) -> Form:
        '''The small-strain stiffness form, built from this equation's material.

        The bilinear form exists only for the small-strain measure: a
        Green-Lagrange energy is not quadratic, so it has no constant stiffness.
        A finite-strain LinearElastic is rejected rather than silently linearised.
        '''
        if self.kinematics is not StrainMeasure.SMALL:
            raise NotImplementedError(
                f'a linear solve is small-strain only; {self.kinematics.name} kinematics '
                'has no constant stiffness. Use EnergySolver.'
            )
        return LinearElasticForm(LinearElasticMaterial(self.E, self.nu))

    def energy_density(self) -> StVenantKirchhoff:
        '''The stored-energy density for this equation's kinematics.

        Same `W`, different strain measure: SmallStrain subclasses
        StVenantKirchhoff and overrides only the strain, so both satisfy the
        return type.
        '''
        # E may be per-element (TopologyOptimizer sets a density-scaled modulus),
        # but a density carries one pair of Lame parameters for the whole mesh,
        # and an array lamb broadcasts wrongly against the constant d2W/dS2.
        if not isinstance(self.E, int | float):
            raise NotImplementedError(
                'an energy density needs a scalar Youngs modulus, got a per-element '
                'array. Use Solver for density-scaled moduli.'
            )
        density = {
            StrainMeasure.SMALL: SmallStrain,
            StrainMeasure.GREEN_LAGRANGE: StVenantKirchhoff,
        }[self.kinematics]
        return density(self.E, self.nu)

    def derived_field(self) -> 'DerivedField':
        '''The elastic flux to recover and estimate from: the in-plane Cauchy stress.

        Carries the masked Neumann/free boundary residual as well, the term that lets a
        traction-free stress concentration register (see `fem.postprocess`).
        '''
        return StressField()
