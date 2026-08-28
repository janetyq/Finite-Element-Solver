"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its physical parameters.
`operator` gives the `Form` a solve assembles, constant-tangent or not. `space` and
`problem` resolve the equation against a mesh and a BC spec, the two steps every
facade shares. `time_orders` says which time-derivative orders the PDE has a meaning
for, and the solves check it: a steady solve needs order 0, `ThetaMethod` order 1,
`NewmarkMethod` and `ModalAnalysis` order 2.

| PDE                          | class                                   | then                                   |
|------------------------------|-----------------------------------------|----------------------------------------|
| Laplace / Poisson, any κ(x)  | `Poisson(coefficient, source)`          | `.problem(mesh, bc).solve()`           |
| heat                         | `Heat(conductivity, capacity, source)`  | `ThetaMethod(dt, steps).solve(p, u0)`  |
| wave                         | `Wave(stiffness, density, source)`      | `NewmarkMethod(dt, steps).solve(p, u0, v0)` |
| linear elasticity, static    | `LinearElastic(E, nu)`                  | `.solve()`, `BucklingAnalysis`, `ModalAnalysis` |
| elastodynamics               | `LinearElastic(E, nu, density, damping)`| `NewmarkMethod`                        |
| finite strain                | `FiniteStrainElastic(E, nu)`            | `.solve()` (Newton)                    |
| L2 projection                | `Projection(source)`                    | `.solve()`                             |
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from fem.energies import SmallStrain, StVenantKirchhoff
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import DiffusionForm, EnergyDensity, EnergyForm, Form, LinearElasticForm, MassForm
from fem.materials import LinearElasticMaterial
from fem.loads import Load, VolumeSource
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
    owns how. `field` says what kind of value the unknown takes; the DOFs per node
    follow from it and the mesh. `time_orders` is the set of time-derivative orders
    the PDE has a meaning for (0 steady, 1 first order, 2 second order), which
    `Problem.solve` and the integrators check.

    `source` is the PDE's right-hand side f (a body force for elasticity): a constant,
    a callable of position, or a `Source`. `density` is the coefficient on the
    time-derivative term (mass density for elasticity and the wave equation, volumetric
    heat capacity for heat), read by the integrators and modal analysis through
    `Problem.mass`. `damping` is the `RayleighDamping` a second-order integrator
    applies, and `loads` are load terms beyond the source (a `PointLoad`).
    '''
    field: FieldShape = Scalar()
    time_orders: ClassVar[frozenset[int]] = frozenset({0})

    def __init__(
        self,
        source: FieldValue | VolumeSource = None,
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
        cls = LinearProblem if operator.constant_tangent else Problem
        return cls(space, operator, self.source, bc, density=self.density,
                   loads=self.loads, damping=self.damping, time_orders=self.time_orders)

    def operator(self, space: FunctionSpace) -> Form:
        '''The form a solve assembles for this equation on `space`.'''
        raise NotImplementedError(f'{type(self).__name__} names no operator')


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b): a utility for
    putting a field on the mesh, not a PDE.'''
    time_orders = frozenset({0})

    def __init__(self, source: FieldValue | VolumeSource = None, loads: tuple[Load, ...] = ()) -> None:
        super().__init__(source, loads=loads)

    def operator(self, space: FunctionSpace) -> Form:
        return MassForm(space.n_components)


class Poisson(Equation):
    '''Poisson's equation −∇·(κ∇u) = f, Laplace's at f = 0: the steady scalar
    conservation law. `coefficient` is κ, a constant or a callable of position (a
    conductivity, a permittivity, a permeability). Its transient relatives are `Heat`
    and `Wave`.'''
    time_orders = frozenset({0})

    def __init__(
        self,
        coefficient: FieldValue = 1.0,
        source: FieldValue | VolumeSource = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(source, loads=loads)
        self.coefficient = coefficient

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.coefficient)


class Heat(Equation):
    '''The heat equation ρ ∂u/∂t − ∇·(κ∇u) = f, stepped by a `ThetaMethod`.
    `conductivity` is κ (a constant or a callable of position) and `capacity` the
    volumetric heat capacity ρ on the time derivative. Its steady state is
    `Poisson(coefficient=κ)`.'''
    time_orders = frozenset({1})

    def __init__(
        self,
        conductivity: FieldValue = 1.0,
        capacity: float = 1.0,
        source: FieldValue | VolumeSource = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(source, density=capacity, loads=loads)
        self.conductivity = conductivity

    @property
    def capacity(self) -> float:
        return self.density

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.conductivity)


class Wave(Equation):
    '''The wave equation ρ ∂²u/∂t² − ∇·(T∇u) = f, stepped by a `NewmarkMethod`.
    `stiffness` is T (a membrane's tension, a constant or a callable of position) and
    `density` ρ, so the wave speed is √(T/ρ). Its steady state is
    `Poisson(coefficient=T)`.'''
    time_orders = frozenset({2})

    def __init__(
        self,
        stiffness: FieldValue = 1.0,
        density: float = 1.0,
        source: FieldValue | VolumeSource = None,
        damping: RayleighDamping | None = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(source, density, damping, loads)
        self.stiffness = stiffness

    def operator(self, space: FunctionSpace) -> Form:
        return DiffusionForm(self.stiffness)


class Elasticity(Equation):
    '''Base of the elastic equations: a vector unknown with Young's modulus `E`,
    Poisson's ratio `nu`, and a mass `density`. Static (order 0) or, under a
    `NewmarkMethod`, elastodynamic (order 2). `LinearElastic` and
    `FiniteStrainElastic` name the two models.'''
    field: FieldShape = Vector()
    time_orders = frozenset({0, 2})

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue | VolumeSource = None,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        loads: tuple[Load, ...] = (),
    ) -> None:
        super().__init__(source, density, damping, loads)
        self.E = E
        self.nu = nu

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

    @property
    def material(self) -> LinearElasticMaterial:
        return LinearElasticMaterial(self.E, self.nu)

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
        source: FieldValue | VolumeSource = None,
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
