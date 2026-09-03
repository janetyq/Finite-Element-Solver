"""A thick-walled cylinder pressurized past first yield, against Hill's solution.

Below first yield the wall is Lame's elastic annulus, most stressed at the bore. Push
the pressure past that and the bore cannot carry more: it yields, and a plastic front
marches outward through the wall as the pressure rises, until the whole section flows
at the limit pressure. Hill's classical elastic-plastic solution gives the pressure
that holds the front at radius c,

    p(c) = k * (2 ln(c/a) + 1 - c^2/b^2),      k = sigma_y / sqrt(3),

so first yield is p(a) = k (1 - a^2/b^2) and the limit pressure p(b) = 2k ln(b/a):
the curve the measured front is judged against, the way the plate demo is judged
against Kirsch and Howland.

The material is Ramberg-Osgood deformation plasticity with a sharp hardening exponent
(near elastic-perfectly-plastic), which is valid here because the pressurization is
monotonic; each pressure is an independent Newton solve, seeded with the previous one.
Hill assumes a perfectly plastic, plastically incompressible wall, so the measured
front tracks his curve to within a few percent, not exactly. What this model cannot do
is unload: the residual stress an overpressurized vessel keeps (autofrettage, the
point of doing this to a real vessel) lives in the loading history, which is
flow-theory plasticity's territory.

One quarter of the annulus is modelled, rollers on the two cut edges standing in for
the symmetry, with curved quadratic elements on both arcs. `quarter_annulus` builds
the outline, `pressurize` solves the sweep, and `run` returns a `CylinderStudy` of
plain results; `figures.py` draws it.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions, Initial
from fem.elements import IsoparametricTriangleElement
from fem.mesh.curves import Arc, Line
from fem.mesh.mesh import Mesh
from fem.mesh.outline import Outline
from fem.physics.equations import DeformationPlasticity
from fem.post.solution import ElasticSolution
from fem.regions import on_plane
from fem.algebra.solve import BacktrackingLineSearch, NewtonSolve

E, NU = 1000.0, 0.3


def hill_pressure(c, inner, outer, k):
    """Hill's pressure holding the plastic front at radius `c` (see module docstring).

    At `c = inner` this is the first-yield pressure and at `c = outer` the limit
    pressure; elementwise in `c`.
    """
    c = np.asarray(c, dtype=float)
    return k * (2.0 * np.log(c / inner) + 1.0 - (c / outer) ** 2)


def quarter_annulus(inner: float, outer: float) -> Outline:
    """One quarter of the annulus, the two arcs joined by radial cuts on the axes.

    Both arcs are `Arc` pieces, so Ruppert's split points and the isoparametric edge
    nodes land on the true circles: the bore the pressure loads is round, not a
    chord polygon.
    """
    quarter = np.pi / 2.0
    return Outline([[
        Line([inner, 0.0], [outer, 0.0]),
        Arc([0.0, 0.0], outer, 0.0, quarter),
        Line([0.0, outer], [0.0, inner]),
        Arc([0.0, 0.0], inner, 0.0, quarter).reversed(),
    ]])


def cylinder_bc(inner: float, pressure: float) -> Conditions:
    """Rollers on the two cut edges (the symmetry planes), pressure on the bore.

    The pressure is the traction `p * r_hat` on the bore surface: it pushes the wall
    radially outward, written as a callable of position so it follows the arc. The
    outer surface carries nothing and is traction-free. On the axes the traction's
    tangential component is exactly zero, so it never loads the component the roller
    there pins.
    """
    def outward(points):
        radial = points / np.linalg.norm(points, axis=1, keepdims=True)
        # The arc endpoint on the y axis sits at x = cos(pi/2) ~ 6e-17, and a traction
        # that is nonzero there in x would (rightly) be refused as driving the rolled
        # component; roundoff of the true zero is snapped back to it.
        radial[np.abs(radial) < 1e-12] = 0.0
        return pressure * radial

    def bore(points):
        return np.hypot(points[:, 0], points[:, 1]) <= inner * (1.0 + 1e-6)

    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0, None]),
        Dirichlet(on_plane(1, 0.0), [None, 0]),
        Neumann(bore, outward),
    )


def plastic_front(solution: ElasticSolution, yield_stress: float, inner: float,
                  threshold: float = 0.98) -> float:
    """The radius the yielding has reached: the outermost element at flow stress.

    With near-perfectly-plastic hardening the whole plastic zone sits at von Mises
    stress ~ yield, approached from *below* as the exponent sharpens (in the perfectly
    plastic limit nothing exceeds it at all), so the front is read at a hair under
    yield: `threshold` times it. Read per element (the centroid radius of every
    element past the threshold), so its resolution is the local element size. `inner`
    when nothing has yielded.
    """
    radii = np.hypot(*solution.mesh.centroids.T)
    yielded = solution.von_mises > threshold * yield_stress
    return float(radii[yielded].max()) if bool(yielded.any()) else inner


def pressurize(mesh: Mesh, inner: float, metal: DeformationPlasticity,
               pressures: np.ndarray) -> list[ElasticSolution]:
    """Solve the sweep, each pressure seeded with the last solution.

    Deformation theory is history-free, so every pressure is its own equilibrium and
    the seed is only a head start for Newton; the answers do not depend on the order.
    """
    space = metal.space(mesh, element_type=IsoparametricTriangleElement)
    newton = NewtonSolve(line_search=BacktrackingLineSearch())
    solutions: list[ElasticSolution] = []
    previous: ElasticSolution | None = None
    for pressure in pressures:
        problem = metal.problem(space, cylinder_bc(inner, float(pressure)))
        seed = None if previous is None else Initial(previous)
        previous = problem.solve(strategy=newton, initial=seed)
        solutions.append(previous)
    return solutions


@dataclass
class CylinderStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    inner: float
    outer: float
    yield_stress: float
    k: float                     # sigma_y / sqrt(3), Hill's shear-yield scale
    hardening_exponent: float
    mesh: Mesh
    pressures: np.ndarray        # the sweep, ascending
    fronts: np.ndarray           # measured plastic front radius at each pressure
    showcase: list[tuple[float, float, ElasticSolution, np.ndarray]]   # (p, front, solution, nodal vm)

    @property
    def first_yield(self) -> float:
        """The pressure at which the bore first yields: Hill's p(a)."""
        return float(hill_pressure(self.inner, self.inner, self.outer, self.k))

    @property
    def limit_pressure(self) -> float:
        """The pressure at which the whole wall flows: Hill's p(b)."""
        return float(hill_pressure(self.outer, self.inner, self.outer, self.k))

    @property
    def reserve(self) -> float:
        """Limit pressure over first-yield pressure: the post-yield capacity the
        redistribution buys. Pure geometry, `2 ln(b/a) / (1 - a^2/b^2)`: large for a
        thick wall with under-stressed material to recruit, 1 in the thin-wall limit,
        where first yield and collapse coincide."""
        return self.limit_pressure / self.first_yield

    @property
    def hill_fronts(self) -> np.ndarray:
        """Hill's front radius at each swept pressure, `inner` below first yield.

        `p(c)` is monotone in `c`, so the front is read by bisection; vectorized over
        the sweep by looping, which at a handful of pressures costs nothing.
        """
        from scipy.optimize import brentq
        fronts = []
        for p in self.pressures:
            if p <= self.first_yield:
                fronts.append(self.inner)
                continue
            fronts.append(brentq(
                lambda c: float(hill_pressure(c, self.inner, self.outer, self.k)) - p,
                self.inner, self.outer))
        return np.asarray(fronts)


def run(inner=1.0, outer=2.0, yield_stress=1.0, hardening_exponent=100.0,
        n_pressures=8, min_angle=28, max_area_fraction=0.001,
        resolution=0.02) -> CylinderStudy:
    """Mesh the quarter annulus, sweep the pressure past first yield toward the limit,
    and measure the plastic front at every level.

    The sweep starts below first yield (the front sits at the bore) and stops just
    short of the limit pressure, where the equilibrium exists only because the
    hardening curve still rises.
    """
    outline = quarter_annulus(inner, outer)
    mesh = outline.sample(resolution).mesh(min_angle=min_angle,
                                           max_area_fraction=max_area_fraction)
    metal = DeformationPlasticity(E, NU, yield_stress, hardening_exponent)
    k = yield_stress / np.sqrt(3.0)
    first_yield = float(hill_pressure(inner, inner, outer, k))
    limit = float(hill_pressure(outer, inner, outer, k))
    pressures = np.linspace(0.85 * first_yield, 0.97 * limit, n_pressures)

    solutions = pressurize(mesh, inner, metal, pressures)
    fronts = np.array([plastic_front(s, yield_stress, inner) for s in solutions])
    showcase = [(float(pressures[i]), float(fronts[i]), solutions[i],
                 solutions[i].nodal_von_mises())
                for i in (0, (len(solutions) - 1) // 2, len(solutions) - 1)]
    return CylinderStudy(inner, outer, yield_stress, k, hardening_exponent,
                         mesh, pressures, fronts, showcase)
