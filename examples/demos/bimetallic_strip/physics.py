"""A bimetallic strip, the thermostat's sensing element, against Timoshenko's formula.

Two bonded layers that expand differently turn temperature into motion: heating bends
the strip toward the layer that grows less, and the curvature per degree is the number
a thermostat designer buys the strip for. Timoshenko's bimetal formula (1925) gives
that curvature in closed form; the solve reproduces it, and adds the one thing the
formula integrates away: the internal stress that does the bending, largest at the
bond line where real bimetal elements delaminate. The operating temperatures are the
device's own, nothing here is exaggerated; only the drawings scale the displacement.

`bend` states and solves one strip; `run` solves the design strip and a sweep of layer
thickness ratios and returns a `BimetalStudy` of plain results. Nothing here draws:
`figures.py` does that from the study, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.physics.equations import LinearElastic
from fem.physics.forms import ThermalStrain
from fem.post.solution import ElasticSolution
from fem.regions import on_plane


@dataclass(frozen=True)
class Alloy:
    """One layer's material, in N-mm-K units (moduli in MPa)."""
    name: str
    E: float        # Young's modulus [MPa]
    alpha: float    # linear thermal expansion [1/K]


# The classic thermostat pair: brass expands sixteen times as much as invar, whose
# near-zero expansion below its Curie point is what it is smelted for. Poisson's ratio
# is taken as one value for both (each is near 0.3, and Timoshenko's narrow-strip
# formula carries no nu at all).
BRASS = Alloy('brass', 100e3, 19.0e-6)
INVAR = Alloy('invar', 140e3, 1.2e-6)
NU = 0.3


def strip_mesh(length: float, thickness: float, n_length: int, n_thickness: int) -> Mesh:
    """A structured mesh of the strip, length along x, layers stacked along y:
    `n_length` by `n_thickness` cells (`box_mesh` counts nodes), so a bond line at a
    multiple of `thickness / n_thickness` lands on a mesh line."""
    return box_mesh(corners=[[0.0, 0.0], [length, thickness]],
                    resolution=(n_length + 1, n_thickness + 1))


def layer_properties(mesh: Mesh, split: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-element `E` and `alpha`: brass below `split`, invar above.

    Elements are assigned by centroid, so `split` must sit on a mesh line
    (a multiple of thickness / n_thickness) for the layers to be the stated sizes.
    """
    lower = mesh.centroids[:, 1] < split
    E = np.where(lower, BRASS.E, INVAR.E)
    alpha = np.where(lower, BRASS.alpha, INVAR.alpha)
    return E, alpha


def clamped_root() -> Conditions:
    """The strip held like a thermostat finger: riveted at one end, free elsewhere."""
    return Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))


def bend(mesh: Mesh, split: float, dT: float, reference: float = 0.0) -> ElasticSolution:
    """The strip at `dT` above its stress-free temperature.

    Plane stress (a strip is thin and free in z) on P2 elements (a thin strip in
    bending is where the constant-strain triangle locks). The temperature is uniform:
    a bimetal element is deliberately small and conductive enough to be isothermal,
    so the bending is all material mismatch, none of it temperature gradient.
    """
    E, alpha = layer_properties(mesh, split)
    equation = LinearElastic(E, NU, thermal=ThermalStrain(alpha, reference + dT,
                                                          reference=reference),
                             reduction='plane_stress')
    problem = equation.problem(mesh, clamped_root(), element_type=QuadraticTriangleElement)
    return problem.solve()


def theory_bending(split: float, thickness: float, dT: float) -> tuple[float, float]:
    """Timoshenko's bimetal bending (1925), from force and moment balance.

    A free two-layer strip at uniform `dT` bends with axial strain `e(y) = a - kappa y`,
    `kappa` the curvature of the deflection (positive bending toward the top layer,
    whose fibres a positive `kappa` shortens); the stress `E(y) (e(y) - alpha(y) dT)`
    must carry no net force and no net moment, two linear equations for `(a, kappa)`.
    Equal layers reduce to the familiar
    `kappa = 24 (alpha_1 - alpha_2) dT / (h (14 + n + 1/n))` with `n = E_1 / E_2`.
    Returns `(a, kappa)`; the stress profile in `theory_stress` uses both.
    """
    moments = np.zeros((2, 2))
    misfit = np.zeros(2)
    for y0, y1, layer in [(0.0, split, BRASS), (split, thickness, INVAR)]:
        I0, I1, I2 = y1 - y0, (y1**2 - y0**2) / 2, (y1**3 - y0**3) / 3
        moments += layer.E * np.array([[I0, -I1], [I1, -I2]])
        misfit += layer.E * layer.alpha * dT * np.array([I0, I1])
    a, kappa = np.linalg.solve(moments, misfit)
    return float(a), float(kappa)


def theory_stress(y: np.ndarray, split: float, thickness: float, dT: float) -> np.ndarray:
    """The axial stress at heights `y`: what each fibre is held away from its free
    thermal length, `sigma = E (a - kappa y - alpha dT)`."""
    a, kappa = theory_bending(split, thickness, dT)
    lower = y < split
    E = np.where(lower, BRASS.E, INVAR.E)
    alpha = np.where(lower, BRASS.alpha, INVAR.alpha)
    return E * (a - kappa * y - alpha * dT)


def curvature_of(solution: ElasticSolution, mesh: Mesh) -> float:
    """The strip's curvature, read from the bottom edge's deflection.

    With the root clamped, `v(x) = kappa x^2 / 2` away from the clamp, so the
    quadratic coefficient of a fit is `kappa / 2`. The fit skips the root fifth,
    where the clamp's Saint-Venant disturbance lives.
    """
    bottom = np.flatnonzero(np.abs(mesh.vertices[:, 1]) < 1e-12)
    x = mesh.vertices[bottom, 0]
    v = solution.component(1)[bottom]
    away = x > 0.2 * float(x.max())
    return 2.0 * float(np.polyfit(x[away], v[away], 2)[0])


def midspan_profile(solution: ElasticSolution, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The axial stress through the thickness at mid-span, clear of both ends:
    element centroid heights and `sigma_xx` there, sorted bottom to top."""
    centroids = mesh.centroids
    x0 = float(np.max(mesh.vertices[:, 0])) / 2
    dx = np.diff(np.unique(np.round(centroids[:, 0], 12))).min()
    column = np.abs(centroids[:, 0] - x0) < 1.1 * dx
    order = np.argsort(centroids[column, 1])
    return centroids[column, 1][order], solution.stress[column, 0, 0][order]


@dataclass
class BimetalStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    length: float
    thickness: float
    split: float                    # the bond line: brass below, invar above
    dT_design: float                # the design temperature rise
    gap: float                      # the contact gap the switch panel closes
    mesh: Mesh
    bc: Conditions
    solution: ElasticSolution       # the design strip at dT_design
    kappa_fem: float
    kappa_theory: float
    profile_y: np.ndarray           # mid-span centroid heights
    profile_stress: np.ndarray      # sigma_xx there
    splits: np.ndarray              # the swept bond-line heights
    split_kappa_fem: np.ndarray
    split_kappa_theory: np.ndarray

    @property
    def tip_fem(self) -> float:
        """The tip's rise at the design temperature, `kappa L^2 / 2`."""
        return self.kappa_fem * self.length**2 / 2

    @property
    def tip_theory(self) -> float:
        return self.kappa_theory * self.length**2 / 2

    @property
    def kappa_error(self) -> float:
        """Computed curvature against Timoshenko's, as a fraction."""
        return abs(self.kappa_fem / self.kappa_theory - 1.0)

    @property
    def tip_per_kelvin(self) -> float:
        """The design strip's sensitivity: tip rise per degree [mm/K]."""
        return self.tip_fem / self.dT_design

    @property
    def dT_switch(self) -> float:
        """The temperature rise at which the tip crosses the contact gap."""
        return self.gap / self.tip_per_kelvin

    @property
    def best_split(self) -> float:
        """The bond height maximizing curvature, from the theory curve: the stiffer
        invar wants to be the thinner layer, `t_brass / t_invar = sqrt(E_invar / E_brass)`."""
        dense = np.linspace(0.05, 0.95, 400) * self.thickness
        kappas = [theory_bending(s, self.thickness, self.dT_design)[1] for s in dense]
        return float(dense[int(np.argmax(kappas))])


def run(length=20.0, thickness=1.0, n_length=160, n_thickness=8, dT_design=100.0,
        gap=0.3, splits=(0.25, 0.375, 0.5, 0.625, 0.75)) -> BimetalStudy:
    """Solve the equal-layer design strip and the bond-height sweep.

    The response is linear in `dT` (one solve fixes the whole deflection line), so the
    temperature axis needs no sweep; the bond height changes the operator, so each
    split is its own solve. Splits must sit on mesh lines (see `layer_properties`).
    """
    mesh = strip_mesh(length, thickness, n_length, n_thickness)
    split = thickness / 2
    bc = clamped_root()
    solution = bend(mesh, split, dT_design)
    kappa_fem = curvature_of(solution, mesh)
    _, kappa_theory = theory_bending(split, thickness, dT_design)
    profile_y, profile_stress = midspan_profile(solution, mesh)

    split_kappa_fem = [curvature_of(bend(mesh, s, dT_design), mesh) for s in splits]
    split_kappa_theory = [theory_bending(s, thickness, dT_design)[1] for s in splits]
    return BimetalStudy(length, thickness, split, dT_design, gap, mesh, bc, solution,
                        kappa_fem, kappa_theory, profile_y, profile_stress,
                        np.array(splits), np.array(split_kappa_fem),
                        np.array(split_kappa_theory))
