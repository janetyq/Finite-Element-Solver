"""A steel tuning fork meshed from its outline: its modes, its tuning law, and its
ring-down when struck.

`fork_modes`, `voice_index`, and `strike` each state and solve one problem; `run` calls
them and returns a `ForkStudy` of plain results. Nothing here draws: `figures.py` does
that from the `ForkStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet
from fem.elements import QuadraticTriangleElement
from fem.physics.equations import LinearElastic
from fem.algebra.integrators import NewmarkMethod
from fem.loads import PointLoad
from fem.mesh.mesh import Mesh
from fem.analysis.modal import ModalAnalysis
from fem.problem import RayleighDamping
from fem.regions import TimeDependent, at_indices, on_plane
from fem.post.solution import ModalSolution, TransientSolution

from domains import tuning_fork_pslg

# Real SI steel, so the frequencies come out in Hz a musician would recognise.
E, NU, RHO = 2.0e11, 0.3, 7850.0             # Young's (Pa), Poisson, density (kg/m^3)
E_STAR = E / (1 - NU**2)                      # the plane-strain modulus a 2D solve sees
BETA1_SQ = 1.875104**2                        # first fixed-free beam root, squared

# Grounded at the stem base: the fork's node, held without damping the voice. A free
# body has rigid-body modes the shift-invert eigensolve cannot factor through.
clamp = BoundaryConditions(Dirichlet(on_plane(1, 0.0), [0, 0]))


def cantilever_hz(length, thickness):
    """The ideal clamped-free tine's fundamental (Hz): a bare beam, with no base."""
    return BETA1_SQ / (2*np.pi) * (thickness / length**2) * np.sqrt(E_STAR / (12*RHO))


def fork_modes(tine_length, tine_thickness, n_modes, across, min_angle=27) -> ModalSolution:
    """Mesh a fork from its outline and solve its first `n_modes` on P2 elements.

    The element size is set by resolving the thin tine, `across` elements through its
    thickness, since bending curves across it.
    """
    pslg = tuning_fork_pslg(tine_length=tine_length, tine_thickness=tine_thickness)
    mesh = pslg.mesh(min_angle=min_angle, max_area=0.5*(tine_thickness/across)**2)
    problem = LinearElastic(E, NU, density=RHO).problem(
        mesh, clamp, element_type=QuadraticTriangleElement)
    return ModalAnalysis(n_modes=n_modes).solve(problem)


def voice_index(fork: ModalSolution, tine_length) -> int:
    """The acoustic mode: the lowest whose two tine tips swing in opposite directions.

    A clamped fork's low modes come in pairs: the tips moving together (a rocking that
    shakes the stem, damped the moment the fork is held there) or oppositely. The
    oppositely-moving one keeps the stem still and rings.
    """
    verts = fork.mesh.vertices
    tips = verts[:, 1] > verts[:, 1].max() - 0.2*tine_length
    left, right = tips & (verts[:, 0] < 0), tips & (verts[:, 0] > 0)
    for i in range(len(fork.frequencies)):
        u_x = transverse_motion(fork, i)
        if u_x[left].mean() * u_x[right].mean() < 0:
            return i
    return 0


def transverse_motion(fork: ModalSolution, i) -> np.ndarray:
    """The sideways (x) displacement of mode `i` at the mesh vertices."""
    return fork.modes[i].reshape(-1, 2)[:len(fork.mesh.vertices), 0]


def strike(fork: ModalSolution, voice, ring_periods, steps_per_period,
           ring_down_periods) -> tuple[TransientSolution, int, float, RayleighDamping]:
    """Pinch the tine tips together and release, then let the fork ring down.

    A short half-sine impulse at each tip, equal and opposite, then free vibration
    stepped by Newmark. The opposite pair excites only the modes the tips swing
    oppositely in, the voice and its overtones; a strike on one tine alone would also
    excite the rocking mode (tips together, stem shaking), which is lower than the
    voice and, with nothing here modelling the hand that damps it at the stem, would
    outlast it. The damping is Rayleigh's, C = alpha M + beta K. Mode i then decays as exp(-sigma_i t) with
    sigma_i = alpha/2 + beta omega_i^2/2: the stiffness term damps the high modes
    fastest, so the tap's clatter dies and the voice is what is left ringing. The two
    are set to contribute equally at the voice, which reaches 1/e after
    `ring_down_periods` of its own period. Returns the series, the tip's vertex index,
    the tap length, and the damping.
    """
    omega = 2 * np.pi * float(fork.frequencies[voice])
    period = 2 * np.pi / omega
    sigma = 1.0 / (ring_down_periods * period)
    damping = RayleighDamping(alpha=sigma, beta=sigma / omega**2)
    verts = fork.mesh.vertices
    left_tip = int(np.argmax(np.where(verts[:, 0] < 0, verts[:, 1], -np.inf)))
    right_tip = int(np.argmax(np.where(verts[:, 0] > 0, verts[:, 1], -np.inf)))
    tap_length = 0.1 * period

    def pinch(p, t):
        inward = -np.sign(p[0])      # each tip pushed toward the other
        return [inward * np.sin(np.pi * t / tap_length) if t < tap_length else 0.0, 0.0]

    equation = LinearElastic(E, NU, density=RHO, damping=damping,
                             loads=(PointLoad(at_indices([left_tip, right_tip]),
                                              TimeDependent(pinch)),))
    problem = equation.problem(fork.mesh, clamp, element_type=QuadraticTriangleElement)
    rest = np.zeros(problem.space.n_dofs)
    ringing = NewmarkMethod(dt=period / steps_per_period,
                            steps=int(ring_periods * steps_per_period)).solve(problem, rest, rest)
    return ringing, right_tip, tap_length, damping


@dataclass
class ForkStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    tine_length: float
    tine_thickness: float
    fork: ModalSolution
    voice: int                      # index of the acoustic mode
    sweep_lengths: np.ndarray       # tine lengths swept for the tuning law
    sweep_freqs: np.ndarray         # the voice frequency at each
    ringing: TransientSolution      # the struck fork
    tip: int                        # vertex index of the struck tip
    tap_length: float
    damping: RayleighDamping
    ring_down_periods: float

    @property
    def mesh(self) -> Mesh:
        return self.fork.mesh

    @property
    def freqs(self) -> np.ndarray:
        return self.fork.frequencies

    @property
    def ideal_hz(self) -> float:
        """Beam theory's pitch for this fork's tine as a bare clamped-free beam."""
        return float(cantilever_hz(self.tine_length, self.tine_thickness))

    @property
    def tuning_slope(self) -> float:
        """The fitted exponent of f ~ L^slope over the sweep (beam theory: -2)."""
        return float(np.polyfit(np.log(self.sweep_lengths), np.log(self.sweep_freqs), 1)[0])

    def decay_rate(self, i) -> float:
        """sigma_i: mode `i` rings down as exp(-sigma_i t) under the Rayleigh damping."""
        omega = 2 * np.pi * float(self.freqs[i])
        return self.damping.alpha / 2 + self.damping.beta * omega**2 / 2

    @property
    def tip_trace(self) -> np.ndarray:
        """The struck tip's sideways displacement at every step."""
        return np.array([u.reshape(-1, 2)[self.tip, 0] for u in self.ringing.u])


def run(tine_length=0.088, tine_thickness=0.004, n_across_tine=5, min_angle=27, n_modes=6,
        sweep_lengths=(0.075, 0.088, 0.105, 0.125), ring_periods=40, steps_per_period=40,
        ring_down_periods=15.0) -> ForkStudy:
    """Solve the fork's modes, sweep its tine length, and strike it."""
    fork = fork_modes(tine_length, tine_thickness, n_modes, n_across_tine, min_angle)
    voice = voice_index(fork, tine_length)

    # The tuning law: the voice frequency of a fork at each tine length, solved a
    # little coarser since only one frequency is read from each.
    sweep_freqs = []
    for length in sweep_lengths:
        swept = fork_modes(length, tine_thickness, max(voice + 2, 3), max(3, n_across_tine - 1),
                           min_angle)
        sweep_freqs.append(swept.frequencies[voice_index(swept, length)])

    ringing, tip, tap_length, damping = strike(
        fork, voice, ring_periods, steps_per_period, ring_down_periods)
    return ForkStudy(tine_length, tine_thickness, fork, voice, np.array(sweep_lengths),
                     np.array(sweep_freqs), ringing, tip, tap_length, damping,
                     ring_down_periods)
