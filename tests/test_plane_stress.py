"""Plane stress: the other 2D reduction, owned by the material.

A thin plate is free in z, so `sigma_zz = 0` and the plate thins by `eps_zz` instead of
developing a stress. The in-plane law is the 3D one with `lambda` replaced by
`2 lambda mu / (lambda + 2 mu)`, the textbook `E / (1 - nu^2)` matrix. Uniaxial tension
is exact for P1 and pins the law, the out-of-plane strain, and von Mises at once; the
thermal cases pin the constrained stress, which loads the plane differently from plane strain
because the z growth costs nothing. The rate is in `test_convergence.py`.
"""
import numpy as np
import pytest

from fem.analysis.design import SIMPModel
from fem.analysis.sensitivity import DensityParameterization, MeanStress
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, LinearElastic
from fem.physics.forms import LinearElasticForm, ThermalStrain
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.post.solution import ElasticSolution
from fem.regions import everywhere, on_plane
from fem.space import FunctionSpace
from helpers import close, rollers

E, NU = 200.0, 0.3
MU, LAMB = Enu_to_Lame(E, NU)
ALPHA, DT = 1e-3, 50.0
SIGMA = 10.0


def plate(reduction='plane_stress', thermal=None):
    return LinearElastic(E, NU, thermal=thermal, reduction=reduction)


def uniaxial(reduction, **kwargs):
    """A unit plate on rollers pulled by `SIGMA` on its right edge."""
    bc = rollers(2) + Neumann(on_plane(0, 1.0), [SIGMA, 0.0])
    return plate(reduction).problem(box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5)), bc, **kwargs).solve()


# -- the law --------------------------------------------------------------------------


def test_plane_stress_hooke_matrix_is_the_textbook_one():
    D = LinearElasticMaterial(E, NU, 'plane_stress').constitutive_matrices(2, 1)[0]
    expected = E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    close(D, expected, rtol=1e-12)


def test_plane_strain_is_the_default_and_unchanged():
    material = LinearElasticMaterial(E, NU)
    assert material.reduction == 'plane_strain'
    close(material.constitutive_matrices(2, 1)[0],
          np.array([[2 * MU + LAMB, LAMB, 0], [LAMB, 2 * MU + LAMB, 0], [0, 0, MU]]), rtol=1e-12)


def test_a_3d_solve_has_no_reduction():
    with pytest.raises(ValueError, match='3D'):
        LinearElasticMaterial(E, NU, 'plane_stress').constitutive_matrices(3, 1)
    with pytest.raises(ValueError, match='reduction'):
        LinearElasticMaterial(E, NU, 'plane_stretch')  # type: ignore[arg-type]
    cube = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(2, 2, 2))
    with pytest.raises(ValueError, match='3D'):
        plate().problem(cube)


def test_the_energy_densities_are_plane_strain_only():
    with pytest.raises(NotImplementedError, match='plane strain only'):
        FiniteStrainElastic(E, NU, reduction='plane_stress')
    with pytest.raises(NotImplementedError, match='plane strain only'):
        plate().energy_density()


# -- uniaxial tension, exact for P1 ---------------------------------------------------


def test_uniaxial_tension_of_a_thin_plate():
    """`sigma_xx = SIGMA` and nothing else; the plate stretches by `SIGMA / E`, and
    narrows and thins by `nu SIGMA / E`. The whole one-dimensional Hooke's law, which a
    plane-strain solve does not reproduce."""
    solution = uniaxial('plane_stress')
    close(solution.stress, np.diag([SIGMA, 0.0, 0.0]), rtol=1e-10)
    close(solution.strain, np.diag([SIGMA / E, -NU * SIGMA / E, -NU * SIGMA / E]), rtol=1e-10)
    close(solution.von_mises, SIGMA, rtol=1e-10)


def test_uniaxial_tension_of_a_thick_section_for_contrast():
    """Plane strain holds z: the section stretches less, `(1 - nu^2) SIGMA / E`, and
    carries `sigma_zz = nu SIGMA`."""
    solution = uniaxial('plane_strain')
    close(solution.stress, np.diag([SIGMA, 0.0, NU * SIGMA]), rtol=1e-10)
    close(solution.strain[:, 0, 0], (1 - NU**2) * SIGMA / E, rtol=1e-10)
    close(solution.strain[:, 2, 2], 0.0)


def test_uniaxial_tension_holds_on_p2():
    solution = uniaxial('plane_stress', element_type=QuadraticTriangleElement)
    close(solution.stress, np.diag([SIGMA, 0.0, 0.0]), rtol=1e-10)


# -- thermal strain under plane stress -------------------------------------------------


def test_free_expansion_of_a_thin_plate_is_stress_free_in_every_direction():
    """Free in the plane and in z, the plate grows by `alpha dT` in all three
    directions with no stress anywhere, the 3D free expansion a plane-strain plate
    cannot have."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    solution = plate(thermal=ThermalStrain(ALPHA, DT)).problem(mesh, rollers(2)).solve()
    close(solution.stress, 0.0, atol=1e-11 * E * ALPHA * DT)
    close(solution.strain, ALPHA * DT * np.eye(3), rtol=1e-10)


def test_a_clamped_thin_plate_under_uniform_heating():
    """Held in the plane, free in z: `sigma_xx = sigma_yy = -E alpha dT / (1 - nu)`,
    `sigma_zz = 0`, and the plate thickens by `alpha dT (1 + nu) / (1 - nu)`, its own
    expansion plus the Poisson bulge from the in-plane compression."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0]))
    solution = plate(thermal=ThermalStrain(ALPHA, DT)).problem(mesh, bc).solve()
    close(solution.dofs, 0.0, atol=1e-12)
    close(solution.stress, np.diag([-E * ALPHA * DT / (1 - NU)] * 2 + [0.0]), rtol=1e-10)
    close(solution.strain, np.diag([0.0, 0.0, ALPHA * DT * (1 + NU) / (1 - NU)]), rtol=1e-10)
    # Twice the elastic energy: the in-plane compression works against -alpha dT each.
    close(solution.compliance.sum(), 2 * E * ALPHA**2 * DT**2 / (1 - NU), rtol=1e-10)


def test_plane_stress_constrained_stress_ignores_the_z_component():
    material = LinearElasticMaterial(E, NU, 'plane_stress')
    eigenstrain = ALPHA * DT * np.broadcast_to(np.eye(3), (2, 1, 3, 3))
    sigma = material.constrained_stress(eigenstrain)
    close(sigma, np.diag([E * ALPHA * DT / (1 - NU)] * 2 + [0.0]), rtol=1e-12)
    # Only the in-plane block of the eigenstrain enters.
    only_z = np.zeros((2, 1, 3, 3))
    only_z[..., 2, 2] = ALPHA * DT
    close(material.constrained_stress(only_z), 0.0)


# -- the consumers that read the reduction ------------------------------------------


def test_stress_quantity_of_interest_reads_the_reduction():
    """`MeanStress` completes the Voigt stress with `sigma_zz = 0` under plane stress,
    so its von Mises is the solution's."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    bc = rollers(2) + Neumann(on_plane(0, 1.0), [SIGMA, 0.0])
    problem = plate().problem(mesh, bc)
    solution = problem.solve()
    qoi = MeanStress(problem.space, problem.physics.material)
    close(qoi.value(problem, solution.dofs), solution.von_mises.mean(), rtol=1e-10)


def test_stress_divergence_uses_the_plane_stress_lambda():
    """div(sigma) for u = (x^2, 0) is (2 lambda* + 4 mu, 0) with the plane-stress lambda."""
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    u = np.zeros((space.n_nodes, 2))
    u[:, 0] = space.node_coords[:, 0]**2
    n_el = len(mesh.elements)
    solution = ElasticSolution(space, u.ravel(), strain=np.zeros((n_el, 3, 3)),
                               stress=np.zeros((n_el, 3, 3)), compliance=np.zeros(n_el))
    material = LinearElasticMaterial(E, NU, 'plane_stress')
    _, lamb_star = material.in_plane_lame(2)
    div = LinearElasticForm(material).flux().divergence(solution)
    close(div, [2 * lamb_star + 4 * MU, 0.0], rtol=1e-10)


def test_simp_keeps_the_reduction_in_its_stress():
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(4, 4))
    bc = rollers(2) + Neumann(on_plane(0, 1.0), [SIGMA, 0.0])
    problem = plate().problem(mesh, bc)
    model = SIMPModel(problem)
    rho = np.ones(len(mesh.elements))
    solution = model.solution(rho, model.problem(rho).solve().dofs)
    close(solution.stress[:, 2, 2], 0.0)
    close(solution.stress[:, 0, 0], SIGMA, rtol=1e-10)


def test_density_parameterization_takes_the_reduction():
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(3, 3))
    space = FunctionSpace(mesh, n_components=2)
    rho = np.ones(len(mesh.elements))
    material = LinearElasticMaterial(E, NU, 'plane_stress')
    K0 = DensityParameterization.create(space, rho, material)._K0
    expected = LinearElasticForm(material).element_matrices(space.geometry)
    close(K0, expected, rtol=1e-12)
