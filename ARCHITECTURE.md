# Architecture

An overview of the object model: which concepts exist, which object owns each job, and how they
fit together. Anchored on symbol names rather than line numbers. Open work lives in `BACKLOG.md`.

This file describes the relationships between modules; each module's docstring describes the
module. A change updates this file only when it adds or moves a seam, a layer, or a pattern. A new
member of an existing family (a material, a strategy, an estimator, a named equation) updates its
module's docstring — and the table in `fem/physics/equations.py` when it names a PDE — not this
file; in the ownership grid it joins its family's row.

## The idea

A solve is a composition assembled from parts, not a method looked up by PDE. The package has the
parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedConditions`), the object that holds
a composition (`Problem`), the strategies that consume one (`LinearSolve`, `NewtonSolve`, the time
integrators), and the drivers that wrap a strategy to re-solve (`AdaptiveRefinement`,
`DesignOptimizer`). A transient problem is a steady operator paired with a time integrator, not a
PDE type at the `Problem` level. An `Equation` names the PDE, its physical constants, and the
time-derivative orders it has a meaning for (`time_orders`), which the solves check. "What to
solve" is the `Problem`; "how" is the strategy.

A choice is a parameter when it changes numbers inside one computation (a modulus, a density,
plane stress against plane strain, a thermal strain) and a class when it changes what the object
is: which solve runs, what it returns, what it composes with. `LinearElastic` and
`FiniteStrainElastic` are two classes for that reason, as are the backends, the estimators, and
`Problem` / `LinearProblem`; thermoelasticity is `LinearElastic(thermal=ThermalStrain(...))`, the
same problem with one more load and a corrected stress.

## Building a solve

The chain is the same for every problem. Each step has one required object and a few options with
defaults; `Equation.problem` is this chain with the defaults filled in from the equation.
The README's "What you choose at each step" table is the menu of options; this is the chain.

| Step | Required | Options (default) |
|---|---|---|
| Geometry | `Mesh` | |
| Discretization | `FunctionSpace(mesh, n_components)`, or `Equation.space(mesh)` | `element_type` (linear) |
| Physics | a `Form`, or `Equation.operator` | a `Material` for elasticity; an `EnergyForm` over a strain-energy density for the nonlinear laws; forms compose: `a + b`, `c * a`, each term naming its `domain` (volume or boundary) |
| Conditions | `Conditions(Dirichlet(region, value), ...)` (default none) | `Neumann`, `Robin`; `Source`, `PointLoad`; `Initial`; a region geometric (`on_plane`, `in_box`, a callable of points) or `on_tag(k)`; a value constant, per-component (`None` = free), a callable of points, or `TimeDependent` |
| Statement | `Problem(space, form, conditions)`, or `Equation.problem(mesh, conditions)` (which takes the Discretization step too) | `element_type`; a `LinearProblem` when the form has a constant tangent |
| Solve | `problem.solve()`: `LinearSolve` for a constant tangent, else `NewtonSolve`; or `.solve(problem, ...)` on an integrator, the load stepper, or an analysis | `strategy`, `Backend` (direct); Newton: `line_search`, `regularization`; integrator: `dt`, `steps`, `theta` / `beta`; `initial=`, an `Initial` to start from instead of the conditions' own |
| Result | a typed `Solution`, returned by every `solve`; a steady one is a `NodalField` | |
| Outer loop | | `AdaptiveRefinement` over a `problem_for(mesh)` builder; `DesignOptimizer` over a `SIMPModel` |

Composed by hand:

```python
space = FunctionSpace(mesh, n_components=1)
conditions = Conditions(Dirichlet(on_plane(0, 0), 0.0), Source(1.0))
problem = LinearProblem(space, DiffusionForm(), conditions, backend=IterativeBackend())
solution = problem.solve()
```

The same solve from the named equation, which builds the space and the problem:

```python
solution = Poisson().problem(mesh, conditions).with_backend(IterativeBackend()).solve()
```

The two agree exactly: `Equation.problem` and `Problem.solve` hold no policy of their own, so
anything they can do can be composed, and anything composed (a different form, a hand-built load,
a custom strategy) needs neither. There is no third way: the equation builds, the problem solves.
Named PDEs are `Equation`s, not dispatch keys: `Poisson(k).problem(mesh, conditions)` returns
`LinearProblem(space, DiffusionForm(k), conditions)`, `FiniteStrainElastic(E, nu).problem(mesh,
conditions)` a `Problem` over an `EnergyForm`, and a PDE with no name is just a different
composition.

The equation carries only the law and its material; the forcing and the starting state are the
conditions', a frozen, mesh-independent specification that means the same thing on any mesh.
`conditions.resolve(space)` is the `ResolvedConditions` a solver indexes into — the Dirichlet DOF
partition, the Robin operator terms, the load terms, the starting state — with `at(t)`
re-evaluating the `TimeDependent` values. The value and region forms and the merging and conflict
rules are the docstrings of `fem/conditions.py`, `fem/boundary.py`, and `fem/regions.py`.

## Where the classes sit

Two views of one stack. The layers are the concerns that vary independently:

| # | Layer | Question it answers | Varies with |
|---|---|---|---|
| 1 | Geometry / topology | Where are the nodes, what connects to what? | meshing, refinement |
| 2 | Discretization (function space) | What functions can I represent? How are DOFs numbered? | element order, components per node |
| 3 | Physics (forms + materials) | What equation, what constitutive law? | the PDE being solved |
| 4 | Assembly | How do forms become matrices? | quadrature, element type |
| 5 | Constraints | Which DOFs are fixed, to what? | boundary conditions |
| 6 | Algebra | How is `Ax = b` (or `F(x) = 0`) solved? | direct/iterative |
| 7 | Time integration | How does a semi-discrete system advance in `t`? | scheme, step size |
| 8 | Drivers | Outer loops that re-solve: adaptivity, optimization | the study being run |
| 9 | Post-processing | Derived quantities, I/O, plotting | what you want to see |

Each layer can be swapped without touching its neighbours: direct for iterative algebra (6) without
touching physics (3), Crank-Nicolson for backward Euler (7) without touching the operator, a
refined mesh (1) without re-resolving constraints by hand (5). Post-processing (9) is the one
one-way layer: everything in it depends downward and nothing depends on it, so it grows freely
and never needs a swap.

The grid maps the classes onto those layers: `█` = owns the layer, `▒` = shares it. Each row is a
family: a new class joins the row it belongs to (a new density joins the materials row, a new
strategy the strategies row), and a new row means a new kind of object.

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh`, `Outline` and its pieces (`Curve`), `PSLG` | █ | | | | | | | | |
| `RuppertsAlgorithm`, `RedGreenRefiner` | █ | | | | | | | | |
| `Element` / `ElementGeometry`, `QuadratureRule` | | ▒ | | ▒ | | | | | |
| `FunctionSpace` | | ▒ | | █ | | | | | |
| `NodalField` | | ▒ | | | | | | | ▒ |
| `FieldShape` (`Scalar` / `Vector`) | | | ▒ | | | | | | |
| `Form` (`BilinearForm`s, `EnergyForm`) | | | █ | ▒ | | | | | ▒ |
| `Material` / energy densities | | | █ | | | | | | |
| `Equation` | | | █ | | | | | | |
| `Conditions` / `ResolvedConditions` | | | | | █ | | | | |
| `Problem` / `LinearProblem` | | | ▒ | ▒ | █ | | | | ▒ |
| `DiscreteSystem` | | | | | ▒ | █ | | | |
| `Backend` (`Direct`, `Iterative`, `Minres`) | | | | | | █ | | | |
| `LinearSolve` / `NewtonSolve` / `EigenSolve` / `QuasiStaticStepping` | | | | | | ▒ | | | |
| `ThetaMethod` / `NewmarkMethod` | | | | | | ▒ | █ | | |
| `BucklingAnalysis`, `ModalAnalysis` | | | | | | ▒ | | | ▒ |
| `AdaptiveRefinement`, `SIMPModel` / `DesignOptimizer` | | | | | | | | █ | ▒ |
| Error estimators, `SensitivityAnalysis` | | | | | | | | ▒ | █ |
| `Solution` (typed) | | | | | | | ▒ | | █ |
| `recovery`, `invariants`, `io` (`fem/post`) | | | | | | | | | █ |
| `Plotter`, `PanelView` / `fem/plot` | | | | | | | | | █ |
| `numerics` | | | | | | | | | ▒ |

A class spans columns for one of three reasons. It is a composition waist (`Problem`, `Solution`):
references across layers, logic in one, which is what lets everything else reference nothing. It
is a handoff between a specification and its mechanism: `Problem` resolves the constraints that
`DiscreteSystem` eliminates. Or it is data crossing layers (`NodalField`): data always does, and
only behaviour layers. The one cut the mathematics refuses is assembly (4): the element matrices
are the physics contracted against the discretization, and the `Gᵀ C G` split cuts as finely as
that allows. A new `▒` should be one of these three; one that is not marks logic in the wrong
place.

The shared cells are the splits worth knowing. Constraints (5) have one owner, the `Problem`,
whose constructor resolves the boundary conditions against the space and turns the Dirichlet
partition into its `constraints`, each Neumann and Robin value into a `BoundaryLoad` load term,
and each Robin coefficient into a boundary term of the operator. Algebra (6) is split:
`DiscreteSystem` owns the Dirichlet elimination and the `Backend` owns how the free-free block is
solved. Physics (3) is owned by the physics layer alone: an `Equation` answers `operator` itself,
so no facade holds a mapping from equation to material.

Post-processing (9) is distributed under one rule: a derived quantity lives on the object that
owns the data it needs. A `NodalField` owns what a field on a space can answer by itself
(integrals, components, point evaluation, the mesh warp); `fem/post/recovery.py` owns the nodal
recoveries, each a function of a space; a `Form` owns `sample` / `recover` and `flux` (which flux
is recoverable, read by the estimators off `problem.operator`); `Problem.solve` picks the typed
`Solution` for its operator; `Solution` owns the packaging; `invariants` owns the
frame-independent reductions. A `▒` in column 9 is a convenience method bound by that rule: a
one-line delegation into `fem/post`, never logic. The surface is closed: a new derived scalar is
a function in `fem/post/invariants.py`, and a `Solution` wraps only the canonical plotting set.

## How the pieces meet

What each piece is for and how it meets its neighbours; the pieces themselves are their modules'
docstrings.

### Geometry and discretization

`Mesh` is geometry and topology, its arrays read-only and its derived tables cached, so every
space, solution, and refiner built on it shares it; a changed mesh is a new one (`displaced`,
`refined`, `with_topology`). `FunctionSpace` has a mesh and owns the discretization: element
geometry, DOF numbering, cached operators. Two spaces (P1 and P2, scalar and vector) can share one
mesh, and DOFs are numbered by the space, never read off the mesh. `NodalField` (`fem/field.py`)
is a frozen DOF vector paired with the space that numbers it — what `space.interpolate` returns,
what every steady solution is, and what an integrator or a plot takes; `np.asarray(field)` is the
bare vector, so the numerics never see the wrapper. Per-element data (a stress, a density, an
error estimate) is not a field on the space and stays an `ElementValues` array.

Meshes come from `box_mesh`, from an `Outline`, or from `Mesh(vertices, elements)` directly. An
`Outline` (closed loops of curve pieces; `fem/mesh/outline.py`, `fem/mesh/curves.py`) is the
mesh-independent description; `outline.mesh(...)` samples it into a `PSLG` and runs
`RuppertsAlgorithm` on that. Every chord of a curved piece carries the piece as its `Curve`, so
Ruppert's split points, red-green midpoints, and an isoparametric element's edge nodes all project
onto the true shape, and `boundary_tags` name the loop each facet came from through refinement, so
an `on_tag(k)` condition survives it.

### Assembly

The integration stack is `QuadratureRule -> Element -> ElementGeometry -> Form`. A `QuadratureRule`
is reference-simplex data; an `Element` type is stateless and maps node coordinates to a batched
`ElementGeometry` for the whole mesh; a `Form` contracts that into element blocks in one
vectorized pass, at the larger of the element's default rule and the one the form asks for. P1 is
the `n_qp == 1` special case, so the same forms serve every order. `FunctionSpace.assemble` is
where the two meet:

```
1. degree = form.quadrature_degree (or the element's default)
2. ElementGeometry = element.geometry(node_coords[element_nodes], rule(degree))
3. blocks = form.element_matrices(ElementGeometry)
4. _ScatterPlan(dof_indices(element_nodes)).scatter(blocks)
```

Assembly lives on the space because its working state — the DOF numbering, the cached
geometries, the scatter plans — is the space's own. A second way to assemble (matrix-free, say)
is the trigger to extract it into an object of its own.

### `Form` / `Material`: the physics

Every assembly path goes through a form, and a form is one base class. Every `Form` writes
`element_residuals` and `element_tangents` at a state; what else it can answer is a hook with a
default of "no": `constant_tangent` (every `BilinearForm`, whose residual is `K u`), `has_energy`
(a bilinear form's `½ uᵀ K u`, an `EnergyForm`'s density), `flux` (the recoverable flux),
`near_null_space` (the AMG near-kernel), `element_loads` (a load from the form's own physics,
such as the thermal load of a heated body, integrated at the rule `load_quadrature_degree` asks
for). A consumer reads the answer it needs: `LinearSolve`, the integrators, the analyses, and SIMP
need a constant tangent; `NewtonSolve` needs nothing more and uses the energy as its line-search
merit when there is one; the `Problem` adds the form's load to those from its conditions.

There are two ways to write a form. A `BilinearForm` writes the constant element matrix (the
`Gᵀ C G` pattern: the element supplies the geometry `G`, the form the material `C`) and gets
residual, tangent, and energy from it. An `EnergyForm` delegates to an `EnergyDensity`, which
answers the energy `W`, the stress `P = dW/dF`, and the tangent `A = d²W/dF²` at a batch of
displacement gradients, in three tiers (`energy`, `stress`, `evaluate`) so a line search or a
residual never pays for the tangent it does not read; how a density gets there (a strain
measure, invariants, an inverted hardening curve) is its own business, so one contraction
serves every law. The densities live in
`fem/physics/energies.py` and `fem/physics/plasticity.py`; `Material` owns the constitutive matrix
`D` of the linear law, beside the strain-displacement `B` in `fem/physics/forms.py`. A 2D solve
reduces a 3D body, and the material owns the `reduction`: plane strain by default (the body held
in z, so a stress `σ_zz` develops) or plane stress (a thin plate free in z, so a strain `ε_zz`
develops instead). The in-plane law, the out-of-plane component, the constrained stress of an
eigenstrain, and the Navier operator of the residual estimator all read it from the material; the
finite-strain path is plane strain only. An `Eigenstrain` (`ThermalStrain`) is a strain the
material takes on with no stress; the elastic form subtracts it, so its constrained stress is the
form's load and is subtracted again in stress recovery. The material computes that stress on a
full 3x3 tensor, since under plane strain the expansion denied in z pushes on the plane and a 2D
shortcut misses it.
Stress recovery is on the form (`RecoversElasticState`, the protocol
the elastic forms share with `ElasticSolution` and `StressFlux`): full `(n_elements, 3, 3)`
tensors cross the boundary, never Voigt vectors, and `fem/post/invariants.py` reduces them to
frame-independent scalars.

### `Problem`: the narrow waist

A `Problem` is the assembly-ready composition for one mesh: space, operator (`physics` plus the
Robin boundary terms), load terms (plus `operator_load`, the operator's own, which
`with_operator` rebuilds), constraints. Its residual has two terms, each present in
`energy`, `residual`, and `tangent` alike: the operator's and the load's; `internal_residual` is
kept apart from `load` so a strategy can scale one against the other, and `with_load_factor`
scales the whole loading for continuation. `mass` and `damping_matrix` are the transient side the
integrators and modal analysis read. `LinearProblem` is the case whose operator has a constant
tangent, assembled once and held: the type every consumer that needs one fixed operator asks for.
The `Problem` owns its constraints, so nothing index-keyed is carried across a mesh change: a
driver that remeshes builds a new one. It also answers the two questions that depend on its
physics, by delegating to the form: `solution(u)` packages a solved vector as the typed `Solution`
its operator recovers, and `near_null_space()` is the operator's AMG near-kernel. The solution
type is carried statically: a form is a `Form[S]`, a problem over it a `Problem[S]` whose
`solve()` returns `S`, an equation an `Equation[P]`, so nothing downstream narrows.

### Solves: strategies, backends, integrators, the stepper

The strategy is how a problem is iterated; the `Backend` is how each linear system on the way is
solved; the two compose without either knowing the other. The strategy is given at the call; the
backend is the problem's (`backend=` at construction, `with_backend` for a copy), because the right
one is a function of the problem (definiteness, size, near-kernel) and so is the factorization it
produces: a `LinearProblem` holds its factored `system`, shared by its snapshots, so every solve of
it after the first is a back-substitution. `default_strategy(problem)` and `default_backend(problem)`
are the one place each choice is made for a caller who names none: `LinearSolve` for a constant
tangent, line-searched `NewtonSolve` otherwise; the direct backend. Underneath, `DiscreteSystem`
eliminates the Dirichlet DOFs and hands the free-free block to the backend, which prepares it into a
`Factorization` solved against many right-hand sides. `EigenSolve` covers the solves that are not `Ax = b`
(buckling, modal), sharing the elimination and lifting each mode back to a full DOF vector.

On top of the steady solves sit the two walks. The integrators (`fem/algebra/integrators.py`, one
family per time order) form a constant effective operator from the problem's mass and stiffness,
factor it once, and step the right-hand side, re-evaluating `TimeDependent` values per step
through `problem.load_at(t)` / `constraints_at(t)`; a steady solve or an estimator works on the
snapshot `problem.at(t)`. `QuasiStaticStepping` (`fem/algebra/stepping.py`) walks the *load* path
instead: steady equilibria from rest to full load, each Newton solve seeded with the last, a
diverging step bisected; `t` is a dial on the loading, not physical time. Integrators and stepper
alike return a `TransientSolution`, `history[i]` the typed steady solution at step `i`.

### `Equation`

`Equation` is typed data naming a PDE: its physical constants, the `density` on its
time-derivative term, and `time_orders`, which `Problem.solve`, the integrators, and
`ModalAnalysis` check, refusing an order the equation lacks rather than approximating.
`operator(space)` answers the form of its physics, `space(mesh)` the discretization its field
implies, and `problem(mesh_or_space, conditions)` the composition; every driver goes through these.
The menu of named PDEs and the solve that steps each is the table in `fem/physics/equations.py`.

### Drivers

Two drivers, each over one spec, and the only stateful algorithm objects. `AdaptiveRefinement`
owns a mesh and a `problem_for(mesh)` builder, solves each round's problem, hands
`(problem, solution)` to the estimator, and refines. `DesignOptimizer` owns a `SIMPModel` and each
iteration derives the diluted `LinearProblem` from the current density (`with_operator`), solves
it through `SensitivityAnalysis`, and moves the density by the optimality-criteria update; its
`DesignHistory` is a series like `TransientSolution`.

### Error estimation and sensitivity

An estimator has no physics of its own: each takes `(problem, solution)` and reads the `Flux` off
the problem's operator and the source off the problem; a custom estimate is any callable of the
same two arguments. `fem/analysis/sensitivity.py` computes `dJ/dp` for a `QuantityOfInterest`
through one adjoint solve on the forward factorization; `fem/analysis/design.py` drives the
optimality-criteria update from that gradient.

### `Solution`

One frozen dataclass per solve shape, each holding the `FunctionSpace` it was solved on: the
steady solutions (a `NodalField` plus what the physics recovers: a gradient, the stress state),
the series (`TransientSolution` / `WaveSolution`, `history[i]` a typed steady step), and the
eigen-solutions (buckling, modal; `mode(i)` a `NodalField`). Derived scalars (`von_mises`,
`pressure`) are computed on demand from the stored tensors; the nodal recoveries re-evaluate the
form so a P2 field keeps its within-element variation. `save` / `load` round-trip any of them
through `fem/post/io.py`. A solution is also what a plot takes: `Plotter.plot(solution, values)`
builds a `PanelView` from its space, so a P2 or curved field draws on its true geometry and the
drawing helpers never see a `FunctionSpace`.

### Extension seams

Each seam is a protocol, exported from `fem`, and the classes beside it are the implementations
to copy: `Form` (`BilinearForm` by `element_matrices`, `EnergyForm` by an `EnergyDensity`);
`SolveStrategy` (`LinearSolve`, `NewtonSolve`); `Backend` and `Factorization` (`DirectBackend`,
`IterativeBackend`, `MinresBackend`); `ErrorEstimator` (the three estimators, or any callable of
`(problem, solution)`); `QuantityOfInterest` and `Parameterization` (`Compliance`, `PointValue`,
`DensityParameterization`, `ModulusParameterization`); `Flux` (`GradientFlux`, `StressFlux`); `FieldShape`
(`Scalar`, `Vector`); `Eigenstrain` (`ThermalStrain`).

## Conventions

### Module order

`tests/test_layering.py` holds the package's module order, bottom to top, and checks that every
module's top-level imports name only modules at or below it. The list is the package inventory in
reading order: what a module may assume exists. `TYPE_CHECKING` imports are exempt outright: they
are erased at runtime, so types flow upward freely. A function-local import that points upward is
a back-edge, named in the importing module's docstring and checked against the `BACK_EDGES` list
in the test, so a new one is a deliberate decision, not drift:

- `problem -> algebra.solve`: `Problem.solve` picks `default_strategy`.
- `field -> physics.forms`: `boundary_integral` assembles a boundary mass form.
- `physics.derived -> physics.forms`: a stress divergence builds the elastic form.
- `analysis.estimators -> analysis.sensitivity`: the goal-oriented estimator solves the dual.
- `mesh.mesh -> mesh.refinement, post.io` and `post.solution -> post.io`: `refined`, `save`, `load`
  as methods on the object they act on.
- `mesh.pslg -> mesh.ruppert`: `PSLG.mesh` runs the mesher.
- `mesh.outline -> mesh.svg`: `Outline.from_svg` runs the reader.

A function-local import that points downward (`fem`'s `__getattr__` serving the plot layer)
defers cost, not layering, and needs no entry.

Everything a user needs is re-exported from `fem`; `Plotter` / `PlotMode` are served lazily so
`import fem` never imports matplotlib. The MMS convergence studies are demo and test support, not
library, and live in `examples/mms.py`.

### Vocabulary

Construction: `from_*` builds from another representation (`ElasticSolution.from_solve`,
`Outline.from_polygons`); `with_*` returns a copy with one thing changed (`LinearProblem.with_operator`,
`Mesh.with_topology`); `at(t)` fixes a time-dependent object at one instant, `history[i]` picks a step;
`sample(geometry)` evaluates a field at a rule's points; `*_for(x)` resolves a choice against `x`
(`element_type_for`, `problem_for`).

### Algorithm objects

The strategies, the integrators, and the stepper are frozen dataclasses of
their parameters with one `solve`. What varies per call (the problem, an `initial` to continue
from) is an argument, never a field, so one configured object serves many solves.
Only the drivers (`AdaptiveRefinement`, `DesignOptimizer`) hold state.

### Exceptions

`NotImplementedError` for a capability an object does not have, naming the
alternative (`'Use NewtonSolve.'`); `TypeError` for the wrong kind of object (a state-dependent
form handed to `LinearSolve` or `SIMPModel`, an abstract base instantiated); `ValueError` for bad
data (a field of the wrong length, a negative volume fraction); `RuntimeError` for a solve that
ran and failed (a singular factorization, a backend that rejected every shift). A failure with a
usable partial result carries it on the exception: `NewtonDivergence.u` is the last iterate,
`SteppingDivergence.history` the steps that had converged.

### The recurring pattern

`fem/regions.py` + `fem/conditions.py` is the model: a mesh-independent specification
(`Conditions`, a frozen tuple of conditions and loads) separated from its resolution against one
space (`ResolvedConditions`, frozen), with the time-dependent values a second, cheaper step on the
resolution (`ResolvedConditions.at(t)`). The same shape recurs: `FunctionSpace` is the resolved
discretization, `Form` the resolved view of an `Equation`'s physics, `Problem` the resolved
composition, and `Factorization` a `Backend` resolved against one matrix.
