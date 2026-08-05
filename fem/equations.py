"""What to solve: the PDE identity, its physical constants, and its physics.

An `Equation` is typed data naming a PDE and carrying its genuinely physical
parameters. It sits in the physics layer next to `fem.forms` and `fem.energies`,
not with the solvers: `Solver` and `EnergySolver` both consume equations, so
neither module owns them.

Each subclass answers "what physics do I mean?" for both assembly paths --
`operator` gives the bilinear form the linear path assembles, `energy_density`
gives the strain-energy density the nonlinear path differentiates. A solver picks
a path; it does not decide what material an equation implies.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from fem.energies import SmallStrain, StVenantKirchhoff

if TYPE_CHECKING:
    from fem.adaptivity import RefinableSolver
from fem.fields import FieldShape, Scalar, Vector
from fem.forms import EnergyDensity, Form, LaplacianForm, LinearElasticForm, MassForm
from fem.materials import LinearElasticMaterial
from fem.typing import ElementField, FieldValue


class Equation:
    '''Base class for a PDE to solve.

    An Equation is typed data: it says *what* to solve and carries the physical
    parameters, while a solve strategy owns *how* (the same equation, e.g.
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

        The scalar diffusion family -- Poisson, and the Laplacian behind the heat
        and wave problems -- shares the material-free Laplacian, so it is the base
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


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b).'''

    def operator(self, n_components: int) -> Form:
        return MassForm(n_components)


class Poisson(Equation):
    '''Poisson equation (K u = b).'''

    def error_estimate(self, solver: RefinableSolver) -> ElementField:
        '''Residual-based a posteriori error estimator for adaptive refinement.

        Computes η_K = √(h_K² ‖f‖²_K + (h_K/2) Σ_edges ‖[[∇u·n]]‖²_e).

        For P1 elements, the Laplacian vanishes inside each element (gradient is
        constant), so the interior residual is just the source f. Boundary edges
        contribute no jump term.
        '''
        mesh = solver.mesh
        space = solver.space

        if solver.solution is None:
            raise ValueError('error_estimate requires a solved system')
        u = solver.solution.u

        h_K = mesh.element_diameters
        n_elements = len(mesh.elements)

        # Interior residual: h_K² × ∫_K f² dx ≈ h_K² × f(centroid)² × area
        from fem.regions import evaluate_field
        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        f_values = evaluate_field(self.source, centroids, n_components=1).ravel()
        interior_term = h_K**2 * f_values**2 * space.element_volumes

        # Edge jump term: (h_K/2) × Σ_edges ‖[[∇u·n]]‖² × edge_length
        grad_u = space.gradient(u)
        jump_term = np.zeros(n_elements)

        for edge, adjacent in mesh.edge_to_elements.items():
            if len(adjacent) != 2:
                continue  # boundary edge

            e0, e1 = adjacent
            v0, v1 = edge
            edge_vec = mesh.vertices[v1] - mesh.vertices[v0]
            edge_len = float(np.linalg.norm(edge_vec))

            if mesh.spatial_dim == 2:
                normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            else:
                raise NotImplementedError('3D error estimator needs face normals')

            jump = float(np.dot(grad_u[e0] - grad_u[e1], normal))
            edge_contribution = edge_len * jump**2
            jump_term[e0] += (h_K[e0] / 2) * edge_contribution / 2
            jump_term[e1] += (h_K[e1] / 2) * edge_contribution / 2

        eta_squared = interior_term + jump_term
        return np.sqrt(np.maximum(eta_squared, 0.0))


class StrainMeasure(Enum):
    '''Which strain the elastic energy is built on -- the kinematics axis.

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
        # E may be per-element -- TopologyOptimizer sets a density-scaled modulus
        # -- but a density carries one pair of Lame parameters for the whole mesh,
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
