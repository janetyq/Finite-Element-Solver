"""A finned heatsink warmed from a cold start, then compared with a solid block and with
beam theory.

`warm_up`, `compare_with_block`, and `fin_efficiency` each state and solve one problem;
`run` calls them and returns a `HeatsinkStudy` of plain results. Nothing here draws:
`figures.py` does that from the `HeatsinkStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Neumann, Robin
from fem.physics.equations import Heat, Poisson
from fem.physics.forms import BoundaryMassForm
from fem.algebra.integrators import ThetaMethod
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.regions import TimeDependent, in_box, on_plane, union
from fem.post.solution import TransientSolution
from fem.mesh.outline import Outline


def heatsink_outline(width: float = 3.0, base_height: float = 0.5, fin_height: float = 1.4,
                     fin_width: float = 0.22, n_fins: int = 7, margin: float = 0.18) -> Outline:
    """A finned heatsink cross-section (a comb) as a single loop.

    A `width` x `base_height` base slab carries `n_fins` fins of `fin_width` x
    `fin_height` standing on top, evenly spaced and kept `margin` clear of the ends. The
    bottom edge is the heated face (a chip beneath it); every other edge is a surface
    that sheds heat, so a solver reads the whole top and sides as a convective film.
    """
    span = width - 2 * margin
    pitch = (span - fin_width) / (n_fins - 1) if n_fins > 1 else 0.0
    lefts = margin + pitch * np.arange(n_fins)

    # Traced counter-clockwise: the bottom edge, up the right side, then the top from
    # right to left, going up and over each fin, and finally down the left side.
    outline = [(0.0, 0.0), (width, 0.0), (width, base_height)]
    for x_l in lefts[::-1]:
        x_r = x_l + fin_width
        outline += [(x_r, base_height), (x_r, base_height + fin_height),
                    (x_l, base_height + fin_height), (x_l, base_height)]
    outline.append((0.0, base_height))
    return Outline.from_polygons([np.array(outline)])

FIN_THICKNESS = 0.22    # matches the sink's own fins (heatsink_outline's fin_width default)
FIN_LENGTH = 1.4        # the length of this sink's fins (heatsink_outline's fin_height default)


def heatsink_film(mesh):
    """The convective film: every boundary but the heated bottom edge (the surfaces above
    the base, plus the base's two sides down to the corners)."""
    w = float(np.max(mesh.vertices[:, 0]))
    return union(in_box([None, 1e-6], [None, None]), on_plane(0, 0.0), on_plane(0, w))


def heatsink_bc(mesh, base, kappa, u_ambient):
    """The boundary spec: `base` on the bottom edge, a Robin film everywhere else."""
    return BoundaryConditions(base, Robin(heatsink_film(mesh), kappa=kappa, g=kappa * u_ambient))


def steady_heatsink(mesh, bc, kappa, u_ambient):
    """Steady heat field for `bc` (a base condition plus a Robin film).

    Returns (u, heat_shed), where heat_shed is the convective loss through the film,
    kappa * integral_film (u - u_ambient). At steady state that equals the heat entering
    the base, so it is the sink's dissipation.
    """
    problem = Poisson(source=0).problem(mesh, bc)
    u = problem.solve().u
    # The convective loss, read off the same region-restricted boundary mass a Robin
    # condition assembles, so it is the exact discrete integral of (u - u_ambient).
    resolved = bc.resolve(problem.space.nodes, 1)
    film_mass = problem.space.assemble(
        BoundaryMassForm(1, resolved.robin[0].facet_mask), boundary=True)
    heat_shed = kappa * float(np.asarray(film_mass @ (u - u_ambient)).sum())
    return u, heat_shed


def solid_block(width, height, target_area):
    """A structured mesh of a solid `width` x `height` block, at roughly `target_area` per
    element so it matches a Ruppert's mesh built to the same cap."""
    nx = max(2, round(width / np.sqrt(target_area)))
    ny = max(2, round(height / np.sqrt(target_area)))
    return box_mesh(corners=[[0.0, 0.0], [width, height]], resolution=(nx, ny))


def fin_efficiency(kappa, u_ambient, u_hot, thickness, lengths):
    """Fin efficiency for a single straight fin at each length, computed and from theory.

    Efficiency is the heat a fin sheds over what it would shed with all of it at the base
    temperature: eta = shed / (kappa * A_fin * (u_hot - u_ambient)), A_fin = 2L + t the
    convecting surface (two sides and the tip, per unit depth). Beam theory gives
    eta = tanh(m*Lc)/(m*Lc), with m = sqrt(2*kappa/(k*t)) and the corrected length
    Lc = L + t/2 standing in for the convecting tip.
    """
    hot = Dirichlet(on_plane(1, 0.0), u_hot)

    eta_fem = []
    for length in lengths:
        ny = max(10, round(10 * length / thickness))    # ~10 elements across the thickness
        fin = box_mesh(corners=[[0.0, 0.0], [thickness, length]], resolution=(10, ny))
        _, shed = steady_heatsink(fin, heatsink_bc(fin, hot, kappa, u_ambient),
                                  kappa, u_ambient)
        area = 2 * length + thickness
        eta_fem.append(shed / (kappa * area * (u_hot - u_ambient)))
    return np.array(lengths), np.array(eta_fem), theory_efficiency(kappa, thickness, lengths)


def theory_efficiency(kappa, thickness, lengths):
    """Beam theory's fin efficiency tanh(m*Lc)/(m*Lc) at each length (see `fin_efficiency`)."""
    m = np.sqrt(2 * kappa / thickness)      # conductivity k = 1
    lc = np.asarray(lengths, dtype=float) + thickness / 2
    return np.tanh(m * lc) / (m * lc)


def warm_up(mesh, dt, steps, kappa, u_ambient, u_hot, ramp):
    """Warm the sink from a cold start.

    The bottom face is a chip switching on: its temperature ramps from ambient to hot
    over `ramp` seconds, then holds. Every other surface is a convective film,
    du/dn + kappa*(u - u_ambient) = 0. A cold start at ambient makes the run a
    warm-up, the front climbing the fins to a steady gradient. Returns the conditions,
    the series, the heat flux magnitude at each step, and the heat shed at each step.
    """
    def base_temperature(p, t):
        return u_ambient + (u_hot - u_ambient) * min(t / ramp, 1.0)

    bc = BoundaryConditions(
        Dirichlet(on_plane(1, 0.0), TimeDependent(base_temperature)),
        Robin(heatsink_film(mesh), kappa=kappa, g=kappa * u_ambient),
    )
    # The heat equation is Poisson's operator integrated in time (see fem.problem.heat).
    heat = Heat().problem(mesh, bc)
    u_initial = heat.space.interpolate(u_ambient)
    solution = ThetaMethod(dt=dt, steps=steps).solve(heat, u_initial)
    # Each step as a steady solution carries the recovered heat flux -grad u. The heat
    # shed through the film at each step is the same convective integral the steady
    # comparison reads, kappa * integral_film (u - u_ambient).
    flux_values = [np.linalg.norm(solution.at(i).nodal_flux(), axis=1)
                   for i in range(len(solution.u))]
    film_mass = heat.space.assemble(
        BoundaryMassForm(1, heat.resolved.robin[0].facet_mask), boundary=True)
    shed_values = [kappa * float(np.asarray(film_mass @ (u - u_ambient)).sum())
                   for u in solution.u]
    return bc, solution, flux_values, shed_values


def compare_with_block(mesh, block, kappa, u_ambient, u_hot, flux):
    """The block and the finned sink at steady state, posed two ways.

    Fixed power: the same heat flux into each base (a chip of fixed wattage); compare the
    base temperature. Fixed temperature: each base held hot; compare the heat shed. The
    thermal resistance R = (base rise)/power is the shape's property either way. Returns
    the four fields and the two heats shed with the base held hot.
    """
    flux_in = Neumann(on_plane(1, 0.0), [flux])
    hot = Dirichlet(on_plane(1, 0.0), u_hot)

    u_block_p, _ = steady_heatsink(block, heatsink_bc(block, flux_in, kappa, u_ambient),
                                   kappa, u_ambient)
    u_fin_p, _ = steady_heatsink(mesh, heatsink_bc(mesh, flux_in, kappa, u_ambient),
                                 kappa, u_ambient)
    u_block_t, q_block = steady_heatsink(block, heatsink_bc(block, hot, kappa, u_ambient),
                                         kappa, u_ambient)
    u_fin_t, q_fin = steady_heatsink(mesh, heatsink_bc(mesh, hot, kappa, u_ambient),
                                     kappa, u_ambient)
    return u_block_p, u_fin_p, u_block_t, q_block, u_fin_t, q_fin


@dataclass
class HeatsinkStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    kappa: float
    u_ambient: float
    u_hot: float
    ramp: float
    flux: float
    mesh: Mesh                      # the finned sink
    block: Mesh                     # the solid block of the same bounding box
    bc: BoundaryConditions          # the transient's conditions
    solution: TransientSolution     # the warm-up
    flux_values: list[np.ndarray]   # |grad u| at each step
    shed_values: list[float]        # heat shed through the film at each step
    u_block_p: np.ndarray           # fixed power: block
    u_fin_p: np.ndarray             # fixed power: finned
    u_block_t: np.ndarray           # base held hot: block
    u_fin_t: np.ndarray             # base held hot: finned
    q_block: float                  # heat shed with the base held hot
    q_fin: float
    fin_lengths: np.ndarray         # the single fins swept for efficiency
    eta_fem: np.ndarray
    eta_theory: np.ndarray

    @property
    def u_values(self) -> list[np.ndarray]:
        return self.solution.u

    @property
    def t_values(self) -> np.ndarray:
        return self.solution.t

    @property
    def width(self) -> float:
        return float(np.max(self.mesh.vertices[:, 0]))

    def base_temperature(self, t) -> float:
        """The chip's temperature schedule: ambient to hot over `ramp` seconds, then held."""
        return self.u_ambient + (self.u_hot - self.u_ambient) * min(t / self.ramp, 1.0)

    @property
    def metal_ratio(self) -> float:
        """The finned sink's material over the block's: the fins carve channels out of it."""
        return self.mesh.area / self.block.area

    @property
    def power(self) -> float:
        return self.flux * self.width

    @property
    def block_rise(self) -> float:
        """The block's base temperature above ambient at fixed power."""
        return float(self.u_block_p.max()) - self.u_ambient

    @property
    def fin_rise(self) -> float:
        """The finned sink's base temperature above ambient at fixed power."""
        return float(self.u_fin_p.max()) - self.u_ambient

    @property
    def r_block(self) -> float:
        """Thermal resistance, base rise per unit power."""
        return self.block_rise / self.power

    @property
    def r_fin(self) -> float:
        return self.fin_rise / self.power

    @property
    def effectiveness(self) -> float:
        """Heat shed by the finned sink over the block's, both bases held hot."""
        return self.q_fin / self.q_block

    @property
    def tip(self) -> float:
        """The coldest point at the end of the warm-up: the fin tips."""
        return float(self.u_values[-1].min())

    @property
    def eta_here(self) -> float:
        """The computed efficiency of a fin the length of this sink's own."""
        return float(self.eta_fem[np.argmin(np.abs(self.fin_lengths - FIN_LENGTH))])


def run(dt=0.05, steps=30, kappa=0.3, u_ambient=300.0, u_hot=400.0, ramp=0.6, flux=40.0,
        fin_lengths=(0.4, 0.8, 1.4, 2.0, 2.8), min_angle=28,
        max_area_fraction=0.0004) -> HeatsinkStudy:
    """Mesh the sink, warm it up, compare it with a block, and validate its fins."""
    # A heatsink conducts heat up its fins and sheds it, so the shape is worth measuring;
    # the mesh is built here because it is part of what the demo says.
    outline = heatsink_outline()
    target_area = max_area_fraction * outline.area()
    mesh = outline.mesh(min_angle=min_angle, max_area=target_area)
    width = float(np.max(mesh.vertices[:, 0]))
    height = float(np.max(mesh.vertices[:, 1]))
    # The naive baseline: a solid block of the same bounding box. The fins carve channels
    # out of it, trading metal for surface area.
    block = solid_block(width, height, target_area)

    bc, solution, flux_values, shed_values = warm_up(
        mesh, dt, steps, kappa, u_ambient, u_hot, ramp)
    u_block_p, u_fin_p, u_block_t, q_block, u_fin_t, q_fin = compare_with_block(
        mesh, block, kappa, u_ambient, u_hot, flux)
    lengths, eta_fem, eta_theory = fin_efficiency(
        kappa, u_ambient, u_hot, thickness=FIN_THICKNESS, lengths=fin_lengths)
    return HeatsinkStudy(kappa, u_ambient, u_hot, ramp, flux, mesh, block, bc, solution,
                         flux_values, shed_values, u_block_p, u_fin_p, u_block_t, u_fin_t,
                         q_block, q_fin, lengths, eta_fem, eta_theory)
