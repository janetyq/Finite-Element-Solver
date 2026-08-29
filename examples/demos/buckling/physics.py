"""Buckling loads and modes of a slender column, checked against Euler's column formula.

Buckling is an eigenproblem: a reference load puts the column under a prestress,
BucklingAnalysis assembles the geometric stiffness K_g from it and solves
K phi = -lambda K_g phi, and lambda multiplies the reference load. P2 elements
throughout: the constant-strain triangle locks in bending.

`solve_buckling` solves one column under one end condition; `run` calls it for the
pinned column's modes, for each of the four classic end conditions, and over a sweep of
lengths, and returns a `BucklingStudy` of plain results. Nothing here draws:
`figures.py` does that from the study, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.analysis.buckling import BucklingAnalysis
from fem.elements import QuadraticTriangleElement
from fem.physics.equations import LinearElastic
from fem.mesh.mesh import Mesh
from fem.regions import intersect, on_plane
from fem.post.solution import BucklingSolution
from fem.mesh.structured import box_mesh


def column(length: float = 24.0, height: float = 1.0,
           n_length: int = 48, n_across: int = 6) -> Mesh:
    """A slender column standing upright, meshed for a buckling solve.

    Length runs along y (ends at y = 0 and y = length) so the mode shapes draw as columns
    stand, with `height` the thin cross-dimension along x.

    The through-thickness count is set independently of the aspect
    ratio: a buckling mode is bending, which needs several elements across the thin
    dimension. `n_across` is forced odd so a vertex lands on the neutral axis for a
    pinned end to anchor.
    """
    n_across += 1 - n_across % 2
    return box_mesh(corners=[[0.0, 0.0], [height, length]],
                            resolution=(n_across, n_length))

E, NU = 200.0, 0.3
E_STAR = E / (1 - NU**2)     # plane-strain effective modulus, the one bending sees
equation = LinearElastic(E, NU)


def second_moment(height):
    """Second moment of area of the rectangular section."""
    return height**3 / 12


def euler_load(span, height, K=1.0):
    """Euler (1744): an ideal slender column buckles at P_cr = pi^2 E* I / (K L)^2."""
    return np.pi**2 * E_STAR * second_moment(height) / (K * span)**2


# The four classic end conditions. What sets an end's effective-length factor is
# whether it can rotate: a traction-loaded edge (u_y free) rotates (a pin or a free
# end), an imposed uniform axial displacement (u_y fixed) cannot (a clamp). u_x = 0
# along an edge holds it transversely without touching its rotation. The column
# stands along y, so the ends are at y = 0 and y = span and the load pushes in -y.
def cantilever(span, height):   # fixed-free, K = 2
    return BoundaryConditions(
        Dirichlet(on_plane(1, 0.0), [0, 0]),
        Neumann(on_plane(1, span), [0, -1.0]),
    )


def pinned(span, height):       # pinned-pinned, K = 1
    return BoundaryConditions(
        Dirichlet(on_plane(1, 0.0), [0, None]),
        Dirichlet(intersect(on_plane(1, 0.0), on_plane(0, height / 2)), [0, 0]),
        Dirichlet(on_plane(1, span), [0, None]),
        Neumann(on_plane(1, span), [0, -1.0]),
    )


def fixed(span, height):        # fixed-fixed, K = 1/2
    return BoundaryConditions(
        Dirichlet(on_plane(1, 0.0), [0, 0]),
        Dirichlet(on_plane(1, span), [0, -0.02 * span]),
    )


def fixed_pinned(span, height):  # fixed-pinned, K ~ 0.7
    return BoundaryConditions(
        Dirichlet(on_plane(1, 0.0), [0, 0]),
        Dirichlet(on_plane(1, span), [0, None]),
        Neumann(on_plane(1, span), [0, -1.0]),
    )


ENDS = [('Cantilever', cantilever, 2.0),
        ('Pinned-pinned', pinned, 1.0),
        ('Fixed-fixed', fixed, 0.5),
        ('Fixed-pinned', fixed_pinned, 0.699)]


def solve_buckling(mesh, bc, span, height, n_modes) -> tuple[BucklingSolution, np.ndarray]:
    """The first `n_modes` buckling modes of the column and their physical loads.

    The load factor multiplies the reference load; the physical buckling load is that
    factor times the actual axial force the column carries, read at mid-span where it
    is uniform and clear of the end disturbances.
    """
    problem = equation.problem(mesh, bc, element_type=QuadraticTriangleElement)
    solution = BucklingAnalysis(n_modes=n_modes).solve(problem)
    centroids = mesh.centroids
    dy = span / (len(np.unique(mesh.vertices[:, 1])) - 1)
    midspan = np.abs(centroids[:, 1] - span / 2) < dy
    assert solution.reference is not None
    axial = -float(np.mean(solution.reference.stress[midspan, 1, 1])) * height
    return solution, solution.load_factors * axial




@dataclass
class EndCondition:
    """One way of holding the column's ends, solved for its first buckling mode."""
    name: str
    bc: BoundaryConditions
    solution: BucklingSolution
    load: float                     # the first critical load
    K_ideal: float                  # Euler's effective-length factor
    K_measured: float               # the factor read back from the computed load


@dataclass
class BucklingStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    length: float
    height: float
    mesh: Mesh
    pinned_bc: BoundaryConditions
    pinned: BucklingSolution        # the pinned column's first modes
    pinned_loads: np.ndarray        # their critical loads
    ends: list[EndCondition]        # the same column held four ways
    sweep_lengths: np.ndarray       # pinned-column lengths swept for the slenderness law
    sweep_loads: np.ndarray         # the first critical load at each

    @property
    def n_modes(self) -> int:
        return len(self.pinned_loads)

    @property
    def slope(self) -> float:
        """The fitted exponent of P_cr ~ L^slope over the sweep (Euler: -2)."""
        return float(np.polyfit(np.log(self.sweep_lengths), np.log(self.sweep_loads), 1)[0])

    @property
    def load_ratios(self) -> dict[str, float]:
        """Each other end condition's critical load over the pinned column's."""
        pinned_load = next(e.load for e in self.ends if e.name == 'Pinned-pinned')
        return {e.name: e.load / pinned_load for e in self.ends if e.name != 'Pinned-pinned'}


def run(length=24.0, height=1.0, n_length=48, n_across=6, n_modes=3,
        sweep_lengths=(16.0, 20.0, 28.0, 40.0)) -> BucklingStudy:
    """Solve the pinned column's modes, the four end conditions, and the length sweep."""
    n_across += n_across % 2      # a vertex on the neutral axis, for the pinned anchor
    mesh = column(length, height, n_length, n_across)

    # 1. Mode shapes of a pinned column: the buckling analogue of vibration modes.
    pinned_bc = pinned(length, height)
    pinned_solution, pinned_loads = solve_buckling(mesh, pinned_bc, length, height, n_modes)

    # 2. Effective length: the same column, four ways to hold its ends. K is read back
    # from the computed load by inverting Euler's formula.
    ends = []
    for name, make_bc, K_ideal in ENDS:
        bc = make_bc(length, height)
        solution, loads = solve_buckling(mesh, bc, length, height, 1)
        K_measured = np.pi / length * np.sqrt(E_STAR * second_moment(height) / loads[0])
        ends.append(EndCondition(name, bc, solution, float(loads[0]), K_ideal, float(K_measured)))

    # 3. Slenderness: the pinned column's critical load over a sweep of lengths.
    sweep_loads = [solve_buckling(column(L, height, max(32, int(2 * L)), n_across),
                                  pinned(L, height), L, height, 1)[1][0] for L in sweep_lengths]
    return BucklingStudy(length, height, mesh, pinned_bc, pinned_solution, pinned_loads, ends,
                         np.array(sweep_lengths), np.array(sweep_loads))
