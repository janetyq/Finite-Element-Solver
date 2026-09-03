"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its physical parameters.
`operator` gives the `Form` a solve assembles, constant-tangent or not. `space` and
`problem` resolve the equation against a mesh and a `Conditions`, the two steps every
facade shares. `time_orders` says which time-derivative orders the PDE has a meaning
for, and the solves check it: a steady solve needs order 0, `ThetaMethod` order 1,
`NewmarkMethod` and `ModalAnalysis` order 2.

| PDE                          | class                                   | then                                   |
|------------------------------|-----------------------------------------|----------------------------------------|
| Laplace / Poisson, any κ(x)  | `Poisson(coefficient)`                  | `.problem(mesh, conditions).solve()`   |
| heat                         | `Heat(conductivity, capacity)`          | `ThetaMethod(dt, steps).solve(p, u0)`  |
| wave                         | `Wave(stiffness, density)`              | `NewmarkMethod(dt, steps).solve(p, u0, v0)` |
| linear elasticity, static    | `LinearElastic(E, nu)`                  | `.solve()`, `BucklingAnalysis`, `ModalAnalysis` |
| thermoelasticity             | `LinearElastic(E, nu, thermal=ThermalStrain(alpha, T))` | `.solve()`; `T` from a `Poisson` or `Heat` solve |
| elastodynamics               | `LinearElastic(E, nu, density, damping)`| `NewmarkMethod`                        |
| finite strain                | `FiniteStrainElastic(E, nu)`            | `.solve()` (Newton)                    |
| deformation plasticity       | `DeformationPlasticity(E, nu, sigma_y, n)` | `.solve()` (Newton)                 |
| L2 projection                | `Projection()`                          | `.solve()` with a `Source` to project  |

What is applied to the domain (the boundary conditions, the volume source, point loads)
is a `Conditions`, given to `problem`; the equation carries only the law and its
material.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from fem.physics.energies import SmallStrain, StVenantKirchhoff
from fem.physics.plasticity import RambergOsgood
from fem.physics.fields import FieldShape, Scalar, Vector
from fem.physics.forms import (
    DiffusionForm, EnergyDensity, EnergyForm, Form, LinearElasticForm, MassForm, ThermalStrain,
)
from fem.physics.materials import LinearElasticMaterial
from fem.post.solution import ElasticSolution, FieldSolution, DiffusionSolution
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.space import FunctionSpace
from fem.typing import ElementValues, FieldValue

from fem.mesh.mesh import Mesh

# The problem an equation builds, `LinearProblem[S]` for a constant tangent, else
# `Problem[S]`, with `S` the solution its operator packages.
P = TypeVar('P', bound=Problem[Any])

if TYPE_CHECKING:
    from fem.conditions import Conditions
    from fem.elements import Element


class Equation(ABC, Generic[P]):
    '''Base class for a PDE to solve.

    An Equation says what to solve and carries the physical parameters; a solver
    owns how. `field` says what kind of value the unknown takes; the DOFs per node
    follow from it and the mesh. `time_orders` is the set of time-derivative orders
    the PDE has a meaning for (0 steady, 1 first order, 2 second order), which
    `Problem.solve` and the integrators check.

    `density` is the coefficient on the time-derivative term (mass density for
    elasticity and the wave equation, volumetric heat capacity for heat), read by the
    integrators and modal analysis through `Problem.mass`. `damping` is the
    `RayleighDamping` a second-order integrator applies. The forcing (a source, a
    traction, a point load) is not the equation's: it is a `Conditions`.
    '''
    field: FieldShape = Scalar()
    time_orders: ClassVar[frozenset[int]] = frozenset({0})

    def __init__(self, density: float = 1.0, damping: RayleighDamping | None = None) -> None:
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')
        self.density = density
        self.damping = damping

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
        conditions: Conditions | None = None,
        element_type: type[Element] | None = None,
    ) -> P:
        '''The composition on `domain`: this operator under `conditions`. A
        `LinearProblem` when the operator has a constant tangent, which is what each
        equation's `P` states.

        `domain` is a `Mesh`, discretized through `space(mesh, element_type)`, or a
        `FunctionSpace` with the component count this equation's field implies.
        `element_type` applies to a mesh only.
        '''
        if isinstance(domain, FunctionSpace):
            if element_type is not None:
                raise ValueError('element_type applies to a Mesh; the FunctionSpace already has one')
            expected = self.field.components_for(domain.spatial_dim)
            if domain.n_components != expected:
                raise ValueError(
                    f'{type(self).__name__} has a {expected}-component unknown; the space has '
                    f'{domain.n_components} components per node'
                )
            space = domain
        else:
            space = self.space(domain, element_type)
        operator = self.operator(space)
        # The class follows the operator's tangent, which is what the subclass's `P`
        # declares; the one cast in the package, checked by the LinearProblem constructor.
        cls = LinearProblem if operator.constant_tangent else Problem
        return cast(P, cls(space, operator, conditions, density=self.density,
                           damping=self.damping, time_orders=self.time_orders))

    @abstractmethod
    def operator(self, space: FunctionSpace) -> Form[Any]:
        '''The form a solve assembles for this equation on `space`.'''


class Projection(Equation[LinearProblem[FieldSolution]]):
    '''L2 projection of the source field onto the FE space (M u = b): a utility for
    putting a field on the mesh, not a PDE.'''
    time_orders = frozenset({0})

    def __init__(self) -> None:
        super().__init__()

    def operator(self, space: FunctionSpace) -> MassForm:
        return MassForm(space.n_components)


class Poisson(Equation[LinearProblem[DiffusionSolution]]):
    '''Poisson's equation −∇·(κ∇u) = f, Laplace's at f = 0: the steady scalar
    conservation law. `coefficient` is κ, a constant or a callable of position (a
    conductivity, a permittivity, a permeability). Its transient relatives are `Heat`
    and `Wave`.'''
    time_orders = frozenset({0})

    def __init__(
        self,
        coefficient: FieldValue = 1.0,
    ) -> None:
        super().__init__()
        self.coefficient = coefficient

    def operator(self, space: FunctionSpace) -> DiffusionForm:
        return DiffusionForm(self.coefficient)


class Heat(Equation[LinearProblem[DiffusionSolution]]):
    '''The heat equation ρ ∂u/∂t − ∇·(κ∇u) = f, stepped by a `ThetaMethod`.
    `conductivity` is κ (a constant or a callable of position) and `capacity` the
    volumetric heat capacity ρ on the time derivative. Its steady state is
    `Poisson(coefficient=κ)`.'''
    time_orders = frozenset({1})

    def __init__(
        self,
        conductivity: FieldValue = 1.0,
        capacity: float = 1.0,
    ) -> None:
        super().__init__(density=capacity)
        self.conductivity = conductivity

    @property
    def capacity(self) -> float:
        return self.density

    def operator(self, space: FunctionSpace) -> DiffusionForm:
        return DiffusionForm(self.conductivity)


class Wave(Equation[LinearProblem[DiffusionSolution]]):
    '''The wave equation ρ ∂²u/∂t² − ∇·(T∇u) = f, stepped by a `NewmarkMethod`.
    `stiffness` is T (a membrane's tension, a constant or a callable of position) and
    `density` ρ, so the wave speed is √(T/ρ). Its steady state is
    `Poisson(coefficient=T)`.'''
    time_orders = frozenset({2})

    def __init__(
        self,
        stiffness: FieldValue = 1.0,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
    ) -> None:
        super().__init__(density, damping)
        self.stiffness = stiffness

    def operator(self, space: FunctionSpace) -> DiffusionForm:
        return DiffusionForm(self.stiffness)


class Elasticity(Equation[P]):
    '''Base of the elastic equations: a vector unknown with Young's modulus `E`,
    Poisson's ratio `nu`, and a mass `density`. Static (order 0) or, under a
    `NewmarkMethod`, elastodynamic (order 2). `LinearElastic` and
    `FiniteStrainElastic` name the two models. `thermal` is a `ThermalStrain` the
    law subtracts (`σ = C : (ε − α ΔT I)`), a parameter rather than a model: the
    stiffness, the solve, and the solution are the same.'''
    field: FieldShape = Vector()
    time_orders = frozenset({0, 2})

    def __init__(
        self,
        E: float | ElementValues,
        nu: float,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        thermal: ThermalStrain | None = None,
    ) -> None:
        super().__init__(density, damping)
        self.E = E
        self.nu = nu
        self.thermal = thermal

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


class LinearElastic(Elasticity[LinearProblem[ElasticSolution]]):
    '''Small-strain linear elasticity: the infinitesimal strain `ε = ½(∇u + ∇uᵀ)`
    under Hooke's law, a constant stiffness solved in one linear solve. `E` may be a
    scalar or a per-element array (a SIMP density-scaled modulus). With `thermal`, a
    `ThermalStrain`, it is thermoelasticity: the temperature's expansion enters as a
    load and the stress is `C : (ε − α ΔT I)`.'''

    @property
    def material(self) -> LinearElasticMaterial:
        return LinearElasticMaterial(self.E, self.nu)

    def operator(self, space: FunctionSpace) -> LinearElasticForm:
        return LinearElasticForm(self.material, eigenstrain=self.thermal)

    def energy_density(self) -> SmallStrain:
        '''The quadratic energy the stiffness is the Hessian of.'''
        if self.thermal is not None:
            raise NotImplementedError(
                'the energy densities take no thermal strain yet; the linear operator does'
            )
        return SmallStrain(self._scalar_modulus(), self.nu)


class DeformationPlasticity(Elasticity[Problem[ElasticSolution]]):
    '''Small-strain J2 deformation-theory plasticity under a Ramberg-Osgood curve:
    Hooke's law below `yield_stress`, power-law hardening of exponent
    `hardening_exponent` beyond it, minimised by Newton.

    History-free (stress is a function of the current strain), so it is valid for
    monotonic, proportional loading and has no transient meaning: steady only. See
    `fem.physics.plasticity.RambergOsgood` for the law and its limits.
    '''
    time_orders = frozenset({0})

    def __init__(
        self,
        E: float,
        nu: float,
        yield_stress: float,
        hardening_exponent: float,
        offset: float = 3.0 / 7.0,
    ) -> None:
        super().__init__(E, nu)
        self.yield_stress = yield_stress
        self.hardening_exponent = hardening_exponent
        self.offset = offset

    def operator(self, space: FunctionSpace) -> EnergyForm:
        return EnergyForm(self.energy_density())

    def energy_density(self) -> RambergOsgood:
        return RambergOsgood(self._scalar_modulus(), self.nu, self.yield_stress,
                             self.hardening_exponent, self.offset)


class FiniteStrainElastic(Elasticity[Problem[ElasticSolution]]):
    '''Finite-strain elasticity on the Green-Lagrange strain `S = ½(FᵀF − I)`, a
    stored energy minimised by Newton. `law` builds the energy density from `(E, nu)`;
    St-Venant-Kirchhoff by default.'''

    def __init__(
        self,
        E: float,
        nu: float,
        density: float = 1.0,
        law: Callable[[float, float], EnergyDensity] = StVenantKirchhoff,
        damping: RayleighDamping | None = None,
        thermal: ThermalStrain | None = None,
    ) -> None:
        if thermal is not None:
            raise NotImplementedError(
                'the energy densities take no thermal strain yet; use LinearElastic'
            )
        super().__init__(E, nu, density, damping)
        self.law = law

    def operator(self, space: FunctionSpace) -> EnergyForm:
        return EnergyForm(self.energy_density())

    def energy_density(self) -> EnergyDensity:
        return self.law(self._scalar_modulus(), self.nu)
