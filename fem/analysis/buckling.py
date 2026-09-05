"""Linearised buckling: a `Problem` -> critical load factors and modes, and the
post-buckling path that carries on past them.

The finite-element analogue of Euler's column formula: how far can a load be scaled
before the structure snaps sideways into a different shape? The result is the load
factor λ and the mode it buckles into.

1. Reference solve. Solve the linear-elastic problem under its reference load,
   recovering the membrane prestress σ₀ in every element.
2. Geometric stiffness. Assemble `K_g(σ₀)` (`GeometricStiffnessForm`), the
   initial-stress matrix that softens the structure under compression.
3. Eigenproblem. Solve `K φ = -λ K_g φ` for the lowest few λ through `EigenSolve`.
   `λ_1` is the critical load factor and `φ_1` the shape it buckles into.

That answers where the structure buckles, not what it does afterwards, which is a
question only a finite-strain solve on a slightly imperfect structure can answer.
`PostBucklingAnalysis` is that pipeline: the eigenproblem above for `(λ_cr, φ)`, a
geometric imperfection seeded in `φ`, the same loading restated on the imperfect mesh
under a finite-strain law, and `ArcLengthStepping` along the equilibrium path.
"""
import logging
from dataclasses import dataclass, replace

import numpy as np

from fem.algebra.solve import EigenSolve, LinearSolve
from fem.algebra.stepping import ArcLengthStepping
from fem.boundary import Condition
from fem.conditions import Conditions, Initial
from fem.elements import Element
from fem.loads import Load, PointLoad
from fem.mesh.mesh import Mesh
from fem.physics.equations import FiniteStrainElastic, LinearElastic
from fem.physics.forms import GeometricStiffnessForm
from fem.post.solution import BucklingSolution, ElasticSolution, PathSolution
from fem.problem import LinearProblem
from fem.regions import at_indices
from fem.space import FunctionSpace
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucklingAnalysis:
    '''Linearised buckling of a `LinearProblem`: load factors and mode shapes.

    The problem's boundary conditions encode the reference load whose buckling
    multiplier is sought (a compressive traction on an end, say). The load factors are
    dimensionless multipliers of that load, so the caller multiplies by the reference
    load's magnitude to get the buckling load in physical units. The operator must be
    a small-strain elastic stiffness: the prestress is read from the problem's
    `ElasticSolution`.

    Bending dominates a buckling mode and the linear (constant-strain) triangle locks in
    bending; a P2 space reaches Euler's loads on a far coarser mesh.
    '''
    n_modes: int = 4

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {self.n_modes}')

    def solve(self, problem: LinearProblem[ElasticSolution]) -> BucklingSolution:
        '''The buckling factors and modes about the problem's reference solve.'''
        if not problem.is_linear:
            raise TypeError(
                'linearised buckling needs a constant tangent (the small-strain stiffness); '
                f'{type(problem.operator).__name__} depends on the state'
            )
        logger.info('Buckling: reference solve for the prestress state...')
        space = problem.space
        reference = problem.solve(LinearSolve())
        if not isinstance(reference, ElasticSolution):
            raise TypeError('buckling needs the recovered stress; got a bare FieldSolution')

        d = space.spatial_dim
        # The stiffness the reference solve already assembled, including any Robin
        # (elastic support) term, which stiffens the structure in the eigenproblem too.
        K = problem.tangent()
        # The in-plane prestress drives the geometric stiffness; σ_zz (the plane-strain
        # out-of-plane component) has no in-plane displacement gradient to couple to.
        prestress = np.ascontiguousarray(reference.stress[:, :d, :d])

        # Buckling needs compression somewhere: if every principal stress of the prestress
        # is non-negative, K_g is positive-semidefinite and K + λ K_g stays SPD for all
        # λ > 0, so nothing buckles. A rigorous, cheap guard for the clean cases (an
        # unstressed structure or one in pure tension) that also spares `eigsh` a
        # trivial eigenproblem (`K_g = 0` forces every μ to 0, so no finite buckling
        # factor). It does not catch a member in overall tension whose clamped end
        # develops local corner compression: that discretely does have a (huge, spurious)
        # mode, and there is no threshold-free way to rule it out here.
        principal = np.linalg.eigvalsh(prestress)     # (n_el, d), ascending per element
        scale = float(np.abs(prestress).max())
        if scale == 0.0 or float(principal.min()) > -1e-9 * scale:
            raise ValueError(
                'no compressive prestress: the reference load leaves the structure in '
                'tension everywhere (or unstressed), which stiffens rather than buckles, '
                'so there is no buckling mode. Reverse the load.'
            )

        K_g = space.assemble(GeometricStiffnessForm(prestress))

        logger.info('Buckling: solving the eigenproblem K phi = -lambda K_g phi...')
        # -K_g φ = μ K φ with K the PD side; μ = 1/λ, so the largest μ ('LA') are the
        # smallest load factors, reached directly without shift-invert.
        eigensolve = EigenSolve(self.n_modes, which='LA')
        mu, modes = eigensolve.solve(-K_g, K, problem.partition)
        factors, modes = _buckling_factors(mu, modes, self.n_modes)
        return BucklingSolution(space, factors, modes, reference=reference)


def _buckling_factors(mu: FloatArray, modes: FloatArray,
                      n_requested: int) -> tuple[FloatArray, FloatArray]:
    '''Raw eigenvalues `μ = 1/λ` to ascending, positive-only load factors.

    Only positive μ buckle: a negative μ is a direction the load stiffens, not
    softens. Modes ride along with their factors to stay aligned, and the result is
    shorter than `n_requested` when some of the eigenpairs found are stiffening ones,
    which is worth saying out loud rather than handing back a short array.
    '''
    tol = 1e-8 * float(np.max(np.abs(mu)))
    positive = mu > tol
    if not positive.any():
        raise ValueError(
            'no positive buckling factor found: the reference load puts the structure '
            'in tension, which stiffens rather than buckles it. Reverse the load.'
        )

    factors = 1.0 / mu[positive]
    modes = modes[positive]
    if len(factors) < n_requested:
        logger.warning(
            'buckling: %d of the %d requested modes buckle; the others stiffen under '
            'the reference load (negative μ) and have no load factor. Ask for fewer '
            'modes, or reverse the load if none of them is the mode you meant.',
            len(factors), n_requested,
        )
    order = np.argsort(factors)
    return factors[order], modes[order]


@dataclass(frozen=True)
class PostBucklingResult:
    '''What `PostBucklingAnalysis` computes: the linearised prediction and the path
    that walks past it.

    `buckling` is the eigenproblem's answer on the pristine structure, whose
    `critical_load_factor` is the yardstick the path is read against, and `path` the
    equilibrium path of the imperfect one, its `lambdas` multiplying the same reference
    load. `imperfection` is the amplitude η the mode was seeded at, in length units.

    Two results of one analysis rather than one enriched result: `PathSolution` is a
    persisted series whose fields `fem.post.io` reflects over, and a reference to a
    different solution on a different mesh is not part of that series.
    '''
    buckling: BucklingSolution
    path: PathSolution[ElasticSolution]
    imperfection: float

    @property
    def critical_load_factor(self) -> float:
        '''λ_cr from the linearised analysis: the load the path's knee sits at.'''
        return self.buckling.critical_load_factor

    @property
    def mesh(self) -> Mesh:
        '''The imperfect mesh the path was traced on (the pristine one displaced by the
        seeded mode); `buckling.mesh` is the pristine one.'''
        return self.path.mesh


@dataclass(frozen=True)
class PostBucklingAnalysis:
    '''The equilibrium path of a structure loaded past its buckling load.

    Linearised buckling gives a load and a shape and stops there: at λ_cr the perfect
    structure has a second equilibrium, and nothing about how much load it carries once
    it is in it. A real structure is never perfect, and its path is a rounded version of
    that corner, so the answer is a finite-strain solve on a structure seeded with a
    small imperfection in the shape it is about to buckle into:

    1. `BucklingAnalysis` on the small-strain problem for `(λ_cr, φ)`.
    2. A geometric imperfection: the mesh displaced by `imperfection` times `φ`
       normalised to unit peak displacement (its P1 restriction; `Mesh.displaced`).
    3. The same conditions restated on the imperfect mesh, under `equation`.
    4. `ArcLengthStepping` along the path, which is what carries it past the knee and
       through any limit point.

    The parameters are the imperfection amplitude (default `1e-3` times the mesh's
    bounding-box diagonal, small enough to shadow the perfect structure's bifurcation
    and large enough to pick a branch), which `mode` to seed (the critical one), and how
    far to walk: `lambda_factor` times λ_cr, or `stepping.lambda_max` when the caller
    states its own. A `stepping` of None is one scaled to the structure: an arc length
    that puts about six increments below λ_cr, so the parameter means the same thing
    whatever the load's units (`steps_to_critical` and `max_steps` shape that default
    and say nothing once a `stepping` is given). A path that flattens (a column's does, rising as the
    elastica's `1 + Θ²/8` and reaching 1.5 λ_cr only when hugely deflected) ends at
    `stepping.max_steps` instead, which is the honest stop for it.

    `solve` takes the pieces of the problem rather than the problem itself: the path is
    traced on a mesh that does not exist until the mode is known, so the facade builds
    both problems. The reference eigenproblem is `LinearElastic` at the same `E` and
    `nu`, the small-strain law `equation` linearises to at zero strain.
    '''

    imperfection: float | None = None
    mode: int = 0
    n_modes: int = 2
    lambda_factor: float = 1.5
    stepping: ArcLengthStepping | None = None
    steps_to_critical: int = 6
    max_steps: int = 60

    def __post_init__(self) -> None:
        if self.imperfection is not None and self.imperfection <= 0:
            raise ValueError(
                f'imperfection must be a positive amplitude, got {self.imperfection}; a '
                'perfect structure has no post-buckling path to trace, it has a '
                'bifurcation. Leave it None for 1e-3 of the bounding-box diagonal.'
            )
        if self.mode < 0:
            raise ValueError(f'mode is an index into the buckling modes, got {self.mode}')
        if self.n_modes <= self.mode:
            raise ValueError(
                f'n_modes must exceed mode to compute it, got n_modes={self.n_modes} and '
                f'mode={self.mode}')
        if self.lambda_factor <= 1.0:
            raise ValueError(
                f'lambda_factor multiplies λ_cr and must exceed 1 to reach past it, got '
                f'{self.lambda_factor}')
        if self.steps_to_critical < 1:
            raise ValueError(
                f'steps_to_critical must be at least 1, got {self.steps_to_critical}')

    def solve(
        self,
        equation: FiniteStrainElastic,
        mesh: Mesh,
        conditions: Conditions,
        element_type: type[Element] | None = None,
    ) -> PostBucklingResult:
        '''Trace `equation` on `mesh` under `conditions` past its buckling load.

        `element_type` is the element both problems are discretized with; a buckling
        mode is bending, which the linear triangle locks in, so a P2 element earns its
        cost here.
        '''
        if not isinstance(equation, FiniteStrainElastic):
            raise TypeError(
                'the post-buckling path needs a finite-strain law, whose stiffness '
                f'changes as the structure bows; got {type(equation).__name__}. Pass '
                'FiniteStrainElastic(E, nu); LinearElastic is the reference '
                'eigenproblem, which BucklingAnalysis solves on its own.'
            )
        if not isinstance(mesh, Mesh):
            raise TypeError(
                'the imperfection displaces the mesh, so this takes a Mesh rather than '
                f'a ready-made space or problem; got {type(mesh).__name__}'
            )
        reference = LinearElastic(equation.E, equation.nu, reduction=equation.reduction).problem(
            mesh, conditions, element_type)
        buckling = BucklingAnalysis(n_modes=self.n_modes).solve(reference)
        if self.mode >= buckling.n_modes:
            raise ValueError(
                f'mode {self.mode} was asked for but only {buckling.n_modes} of the '
                f'{self.n_modes} eigenpairs buckle under this load; seed a lower mode.'
            )
        critical = buckling.critical_load_factor

        amplitude = self.imperfection if self.imperfection is not None else _default_amplitude(mesh)
        logger.info('Post-buckling: seeding mode %d at an amplitude of %g...',
                    self.mode, amplitude)
        imperfect = mesh.displaced(_unit_mode(buckling, self.mode, mesh.n_vertices), amplitude)

        # The regions are restated against the nodes they selected on the pristine mesh:
        # a predicate written in coordinates (an end plane, say) is resolved on the
        # perturbed ones otherwise, where the mode has moved a loaded or held node off
        # the plane it was named by, silently changing the problem.
        problem = equation.problem(imperfect, _restated(conditions, reference.space), element_type)
        if problem.partition != reference.partition:
            raise ValueError(
                'the conditions bind to different DOFs on the imperfect mesh than on the '
                'pristine one, so the path would not be the stated problem. State the '
                'regions so they survive the perturbation (fem.regions.at_indices pins '
                'them to the nodes they select here).'
            )

        # Past the knee the path is nearly flat in λ and long in displacement, so the
        # adaptive rule is given more room to grow the arc length than its own default:
        # the increment that crossed the knee is far shorter than the ones after it.
        stepping = self.stepping if self.stepping is not None else ArcLengthStepping(
            initial_step=critical / self.steps_to_critical, max_steps=self.max_steps,
            max_step_factor=20.0)
        if stepping.lambda_max is None:
            stepping = replace(stepping, lambda_max=self.lambda_factor * critical)
        logger.info('Post-buckling: tracing the path to λ = %g (λ_cr = %g)...',
                    stepping.lambda_max, critical)
        return PostBucklingResult(buckling, stepping.solve(problem), amplitude)


def _default_amplitude(mesh: Mesh) -> float:
    '''`1e-3` of the mesh's bounding-box diagonal: an imperfection in the ratio a
    structure's own tolerances come in, and one number on any mesh in any units.'''
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    return 1e-3 * float(np.linalg.norm(extent))


def _unit_mode(buckling: BucklingSolution, mode: int, n_vertices: int) -> FloatArray:
    '''`(n_vertices, d)` mode `mode` scaled to unit peak displacement magnitude.

    The P1 restriction of the mode: a `Mesh` has vertices, and on a P2 space they are
    the leading nodes, so the mode's vertex block is the warp `Mesh.displaced` takes.
    An eigenvector's amplitude is arbitrary, so it is the normalisation that makes the
    imperfection amplitude mean a length.
    '''
    shape = buckling.mode(mode).nodal_values[:n_vertices]
    return shape / float(np.linalg.norm(shape, axis=1).max())


def _restated(conditions: Conditions, space: FunctionSpace) -> Conditions:
    '''`conditions` with every region pinned to the nodes it selects on `space`.

    A geometric region means "wherever this place is on the mesh at hand", which is
    what makes a specification survive refinement; on a perturbed copy of one mesh it
    is the wrong reading, since the perturbation is the answer moving, not the domain
    changing. `at_indices` states the selection instead, which is a topological fact
    and identical on both meshes: the perturbation moves vertices without renumbering
    them, and the facet masks a Neumann or Robin integral is built from follow the
    node indices, not the coordinates.

    The values are untouched, so a value that is a callable of position is still read
    at the perturbed coordinates, where it is the same field evaluated at the point the
    node has moved to.
    '''
    nodes = space.nodes
    items: list[Condition | Load | Initial] = []
    for item in conditions.items:
        if isinstance(item, Condition):
            items.append(replace(item, region=at_indices(item.select(nodes))))
        elif isinstance(item, PointLoad):
            selected = np.flatnonzero(item.region(space.node_coords))
            items.append(replace(item, region=at_indices(selected)))
        else:
            items.append(item)
    return Conditions(*items)
