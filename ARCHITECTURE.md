# Architecture — the object model

Companion to `ARCHITECTURE_REVIEW.md`. That document lists defects; this one is about the
object model: which concepts exist and which object owns each job. Anchored on symbol names
rather than line numbers, which drift with every refactor.

This document describes the model *as built*. An earlier revision framed it as "current model,
target model, and the gap"; the migration that closed the gap has landed, so the target and the
current model are now the same thing, and the framing below is the achieved design rather than a
plan. The numeric roadmap that remains — quadrature, higher-order elements, iterative solvers,
an error estimator — lives in `BACKLOG.md`.

---

## The thesis in one paragraph

A solve is not a method you look up by PDE — it is a **composition** you assemble from parts.
The package has the parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedBC`),
the object that *holds* a composition (`Problem`), the strategies that *consume* one
(`LinearSolve`, `NewtonSolve`, the time integrators), and the drivers that *wrap* a strategy to
re-solve (`AdaptiveRefinement`, `TopologyOptimizer`). The god `Solver` that used to hide the
composition — `solve_heat` writing `M + dt·K` by hand, `assemble_everything` writing
`M @ f + M_b @ neumann` by hand — is dissolved: `Heat` and `Wave` were never PDEs, just a steady
operator paired with a time integrator, so they are gone as types; `Equation` is back to the
*identity* of a PDE plus its physical constants. The line the `Equation` docstring always drew —
"*what* to solve" vs "*how*" — is now the `Problem` / strategy boundary, made structural.

---

## 1. The natural layering of an FEM code

These are the concepts the domain actually has. Most mature FEM libraries converge on some
version of this, not by fashion but because each layer varies independently of the others.

| # | Layer | Question it answers | Varies with |
|---|---|---|---|
| 1 | **Geometry / topology** | Where are the nodes, what connects to what? | meshing, refinement |
| 2 | **Discretization (function space)** | What functions can I represent? How are DOFs numbered? | element order, components per node |
| 3 | **Physics (forms + materials)** | What equation, what constitutive law? | the PDE being solved |
| 4 | **Assembly** | How do forms become matrices? | quadrature, element type |
| 5 | **Constraints** | Which DOFs are fixed, to what? | boundary conditions |
| 6 | **Algebra** | How is `Ax = b` (or `F(x) = 0`) actually solved? | dense/sparse, direct/iterative |
| 7 | **Time integration** | How does a semi-discrete system advance in `t`? | scheme, step size |
| 8 | **Drivers** | Outer loops that re-solve: adaptivity, optimization | the study being run |
| 9 | **Post-processing** | Derived quantities, I/O, plotting | what you want to see |

The test of a layering is substitution: you should be able to swap a layer without touching its
neighbours. Swap dense→sparse (6) without touching physics (3); swap heat's θ-method for
backward Euler (7) without touching the operator (3); remesh (1) without re-resolving constraints
by hand (5). Each of these is now a local change — the point of the object model that follows.

## 2. The four tiers of a solve

The layers above are the *concepts*; the tiers below are how the objects stack. Each tier
consumes the one beneath it and varies independently.

| Tier | Role | Objects |
|---|---|---|
| **1 · Primitives** | the parts a composition is built from | `Form` / `EnergyForm` (+ `ScaledForm`, `MaskedMassForm`), `Material`, `FunctionSpace`, `BoundaryConditions` / `ResolvedBC`, `DiscreteSystem` |
| **2 · `Problem`** | a composition: space + operator + load + constraints ("what to solve") | `LinearProblem`, `EnergyProblem`; named factories `poisson`, `linear_elastic`, `heat`, `wave`, `projection` |
| **3 · Solve strategy** | consumes a `Problem`, returns the solution ("how") | `LinearSolve`, `NewtonSolve`; time integrators `ThetaMethod`, `NewmarkMethod` |
| **4 · Driver** | wraps a strategy, re-solving | `AdaptiveRefinement`, `TopologyOptimizer` |

Named PDEs survive as **factory functions**, not dispatch keys: `poisson(mesh, f, bc)` builds the
space and returns `LinearProblem(space, LaplacianForm(), f, bc)`. You do not *dispatch* Poisson; you *are* Poisson
when your operator is a Laplacian and your load is a source. A PDE with no name in any taxonomy
(advection–diffusion, a Robin-loaded plate) is just a different composition, not a new registry
entry — and the composition is still fully typed, so nothing is given up against the repo's
"typed over stringly-typed" rule.

## 3. Where the classes sit

`█` = owns the layer · `▒` = shares it cleanly with another owner · `◧` = holds a piece it
should not.

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh` | █ | | | | | | | | |
| `Element` / `ElementGeometry` | | ▒ | | | | | | | |
| `FunctionSpace` | | ▒ | | █ | | | | | ▒ |
| `FieldShape` (`Scalar` / `Vector`) | | | ▒ | | | | | | |
| `Form` / `EnergyForm` (+ `ScaledForm`, `MaskedMassForm`) | | | █ | ▒ | | | | | |
| `Material` / energy densities | | | █ | | | | | | |
| `Equation` | | | █ | | | | | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `Problem` (`LinearProblem` / `EnergyProblem`) | | | ▒ | ▒ | █ | | | | |
| `DiscreteSystem` | | | | | ▒ | █ | | | |
| `LinearSolve` / `NewtonSolve` | | | | | | ▒ | | | |
| `ThetaMethod` / `NewmarkMethod` | | | | | | ▒ | █ | | |
| `Solver` (steady facade) | | | ◧ | | ▒ | ▒ | | | |
| `EnergySolver` | | | ◧ | | ▒ | ▒ | | | |
| `AdaptiveRefinement` / `TopologyOptimizer` | | | | | | | | █ | ▒ |
| `Solution` (typed) | | | | | | | ▒ | | █ |
| `RedGreenRefiner` | █ | | | | | | | | |
| `Plotter` / `io` | | | | | | | | | █ |

Read the columns. Constraints (5) has one clear owner now: the `Problem`, whose constructor
resolves the boundary conditions against the space and folds the Dirichlet partition into its
`constraints`, the Neumann load into its load vector, and any Robin contribution into *both*
sides. Algebra (6) belongs to `DiscreteSystem`, and every solve strategy sits on it rather than
re-deriving elimination. Time (7) is owned by the integrators, not smeared across `Equation` and
`Solver` as it once was. Drivers (8) are uniform — both own a solver rather than one being a
method on the thing it drives.

Two `◧` remain, both small and both in the physics column. `stiffness_form` (a module function
`Solver` calls) still constructs the `LinearElasticMaterial` when it builds the elastic form, and
`EnergySolver._select_energy` still maps `LinearElastic` to a `StVenantKirchhoff` density. Each is
the last thread of "which material does this equation mean?" left in a solver rather than in the
physics layer. Neither is a conflation of *jobs* — both solvers have otherwise dropped out of
physics, assembly, and time entirely.

---

## 4. Role-by-role

### `Mesh` / `FunctionSpace` — geometry vs discretization

`Mesh` is geometry: vertices, elements, boundary, topology queries. `FunctionSpace` has a mesh
and owns the discretization — element geometry, DOF numbering, cached operators. Two spaces can
share one domain, which is the property that made the split necessary. `assemble` takes a `Form`
rather than an untyped material bag, so the space forwards nothing it cannot interpret. `fem/mesh`
imports no plot code, so the geometry layer is clean of the core → plot dependency.

### `Form` / `Material` — the physics, and its one open axis

The constitutive law is off the element, and **every assembly path goes through a form**:

- **Bilinear forms** — `MassForm` (`∫u·v`), `LaplacianForm` and `LinearElasticForm` (the
  `Gᵀ C G · volume` stiffness family) — scatter through `FunctionSpace.assemble`, one loop that
  no longer knows what it is scattering. `ScaledForm(c², form)` and `MaskedMassForm(mask)` are the
  two combinators that exist because a term needed them (the wave operator's `c²K`, the Robin
  boundary integral); no speculative `OperatorSum` waits ahead of a second use.
- **The nonlinear energy path** is `EnergyForm`, the sibling that maps an element *and a state*
  to an energy, residual, and tangent; the energy path scatters it through
  `FunctionSpace.assemble_residual`/`assemble_tangent`, which `EnergyProblem` calls. A quadratic energy has a constant tangent, so the
  bilinear `Form` is `EnergyForm`'s state-independent special case.
- **Stress recovery is on the form.** `LinearElasticForm.derived_fields(geometry, u_elements)`
  returns strain, stress, and compliance from the same `B` and `D` it assembles from — the mirror
  of `element_matrices`, contracting against the solved displacement instead of assembling a
  stiffness. It is the reason a driver never rebuilds `B` and `D` to recover stresses.

`Material` owns `D`, and the strain-displacement matrix `B` sits in `fem/forms.py` next to the
form that contracts it against `D`. That split is what let `Element` drop to pure geometry.

The two constitutive representations are the same material, *pinned* rather than asserted:
`energies.py`'s `calculate_W_from_S` and the `½εᵀDε` implied by `Material` are one energy
`W(ε) = ½λ(tr ε)² + μ tr(εᵀε)`, and a test checks `D = ∂²W/∂ε²` in 2D. `D` is left in its
Lamé-parameter closed form rather than derived from `W` on purpose: the closed form is
dimension-general, whereas `energies.py` is fixed-rank-2, so deriving `D` from the density would
forfeit the 3D path. The other axis is **kinematics**: the two solver paths differ only in the
strain measure fed to that one `W` — Green–Lagrange `S = ½(FᵀF − I)` (St-VK) versus the
small-strain `ε`. Both are named (`SmallStrain`, `StVenantKirchhoff`) and pinned in
`tests/test_elasticity_models.py`. So the physics layer decomposes as **material** (the energy
`W`) × **kinematics** (the strain measure) — and choosing the kinematics point is the one axis
still made by test-only injection rather than an equation-level choice. It is the last thing the
physics layer wants; see `BACKLOG.md`.

### `Element` — stateless types, batched geometry

Element types are stateless: `LinearTetrahedralElement` describes a shape and holds no
per-element data, so there is one of them in a program rather than one per tet. The per-element
data lives in `ElementGeometry`, which holds it for the whole mesh at once — one
`(n_elements, N, spatial_dim)` array of `grad_phi`, one `(n_elements,)` array of measures — and
`Form.element_matrices` computes every element matrix in a single vectorized pass. The
type/instance split *is* the batching. `EnergyForm` is batched too: the densities evaluate the
full derivative chain over all elements at once, and are dimension-general (parameterized on
`d = grad_u.shape[-1]`), so `EnergySolver` accepts 3D meshes.

### `Problem` — the narrow waist between physics and algebra

A `Problem` is the assembly-ready composition for **one mesh**: a space, an operator, a load, and
constraints. It is to a composition what `ResolvedBC` is to a `BoundaryConditions` — the resolved,
immutable view of a declarative spec. Two shapes share one protocol, mirroring the `Form` /
`EnergyForm` split:

- `LinearProblem` — `tangent(u) = A` (constant), `residual(u) = A·u − b`. Its constructor
  assembles the operator, folds the Neumann load into `b`, and folds each Robin contribution into
  *both* `A` (`κ·` boundary mass) and `b` (the `∫g·v` term) — the two-sided operator/load algebra,
  used for real by Robin.
- `EnergyProblem` — `tangent(u) = ∇²Π(u)`, `residual(u) = ∇Π(u)`, the state-dependent sibling.

The `Problem` **owns its constraints**: nothing index-keyed is carried across a mesh change,
because a driver that remeshes just builds a new `Problem`. That is what took the "re-resolve BCs
after every remesh" dance out of the solver — the class of bug the old `adaptive_refinement`
carried. `LinearProblem` is the special case of `EnergyProblem` where the tangent does not depend
on `u`, exactly the relationship `Form` has to `EnergyForm`.

### Solve strategies — `LinearSolve`, `NewtonSolve`, one engine

Every strategy sits on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet partition +
factor-once solve), and knows nothing about which PDE produced the `Problem`. `LinearSolve`
assembles once and solves once; `NewtonSolve` iterates, re-factoring the tangent each step and
checking convergence before applying the increment. A `LinearProblem` has a constant tangent and
an affine residual, so `NewtonSolve` reaches its solution in a single applied step from any seed —
`LinearSolve` is that step done directly. The two are one engine: `EnergySolver` delegates its
Newton loop to `NewtonSolve(EnergyProblem(...))` rather than carrying a second copy. `SolveStrategy`
is the protocol both satisfy, and `TopologyOptimizer` takes one as an injectable parameter — a
driver that accepts any strategy, which is the protocol earning its place.

### Time integration — a strategy per ODE order

The domain has **two ODE orders** — heat is first (`M u̇ + K u = b`), wave is second
(`M ü + c²K u = b`) — and that split is the real structure, so there is one integrator family per
order rather than one first-order interface for both. `dt` and the step count are constructor data
of the integrator, not fields on an equation; initial conditions arrive through `run(...)`. Each
forms a *constant* effective operator from the problem's mass and stiffness, factors it once
through `DiscreteSystem`, and steps by updating only the right-hand side.

- `ThetaMethod` (θ=½ Crank–Nicolson default, θ=1 backward Euler) solves `(M + θ dt K) u_{n+1} = …`.
- `NewmarkMethod` (β=¼, γ=½ average-acceleration) solves for the acceleration against the SPD,
  N-sized `M + β dt² K` — **not** a 2N first-order block. The wave speed lives in the operator as
  `ScaledForm(c², …)`, so the integrator sees only `c²K` and never learns `c`; constant Dirichlet
  displacement means zero velocity and acceleration at fixed nodes, so those DOFs are the ordinary
  constraint, with no lifting into a block DOF space. That the operator is SPD (where the old block
  system was not) is what keeps it inside the CG/preconditioning story that is the top backlog
  item. `wave_energy` is the invariant this scheme conserves for a linear system, kept as a
  diagnostic.

### `Equation` — identity plus physical constants

`Equation` is typed data: it says *what* to solve and carries the genuinely physical parameters
(`E`/`nu` on `LinearElastic`), while a strategy owns *how*. `Projection`, `Poisson`, and
`LinearElastic` are the members; `Heat` and `Wave` are **not** among them, because a transient
problem is a steady operator paired with an integrator (`problem.heat(...)`, `problem.wave(...)`),
not a distinct PDE type. The `dt`/`iters` that used to sit on `Heat`/`Wave` are gone from the
equation, and `TopologyOptimizer` no longer mutates `equation.E` — it builds a fresh material each
iteration — so the equation is immutable specification again, which is what its docstring always
claimed.

### `Solver` / `EnergySolver` — thin facades over the core

`Solver` is now a steady-solve facade: it holds a mesh, an equation, and boundary conditions;
`solve()` builds a `LinearProblem` and hands it to `LinearSolve`, returning a typed `Solution`.
Its one remaining job beyond composition is `remesh(mesh)`, which rebuilds the space and
re-resolves the BC spec — the seam an `AdaptiveRefinement` driver uses to advance it across meshes
without reaching into its state. `EnergySolver` is the analogous facade for the nonlinear energy
path, delegating to `NewtonSolve(EnergyProblem(...))`. Both keep the one physics `◧` noted in §3
(`stiffness_form` / `_select_energy`), and nothing else.

### Drivers — uniform outer loops

`AdaptiveRefinement` and `TopologyOptimizer` are the two studies, and they now have the *same*
shape: each owns a solver (or strategy) and re-solves. `AdaptiveRefinement` owns a `Solver` and
advances it across meshes via `remesh`; `TopologyOptimizer` owns a `SolveStrategy` and rebuilds a
fresh `LinearProblem` from the current density each iteration. `TopologyOptimizer`'s objective is
an injected object (`MinCompliance`, `TargetCompliance`), not a string resolved through a
`_select_*` dispatch, and its result is a typed `TopologyHistory` rather than a `_list`-suffixed
dict. The old `adaptive_refinement` — a driver living *inside* the `Solver` it drove, mutating
`self.mesh` and re-resolving BCs by hand — is gone.

### `Solution` — typed, one dataclass per shape

The result is a typed dataclass, not a dict of named arrays: `FieldSolution` (the field `u`),
`ElasticSolution` (adds recovered strain/stress/compliance), `TransientSolution` (a time series),
`WaveSolution` (adds the velocity series). A steady field and a time series are now different
*types* rather than both being `values[...]` told apart by guessing at a length. `save`/`load`
round-trip any of them through `fem/io`, which reflects over the dataclass fields and stores the
class name — so the I/O follows the type rather than a naming convention. `TopologyHistory` is
deliberately *not* in this hierarchy: it is a driver-layer trajectory of designs (its axis is
optimization iteration, and `rho` is a design variable, not a solved field), so it stays a
standalone record that aggregates the per-iteration `ElasticSolution`s rather than being one.

---

## 5. Flexibility, and where it now lives

The old model had generality built for extensions that never happened and rigidity where
extension was actually wanted. The speculative machinery is gone: the `TopologyOptimizer._select_*`
plugin system (one method, an ignored args bag, an objective value never evaluated) and
`Solution.get_values(name, iter_idx, mode)` (three-axis generality with one real axis) are both
replaced by typed objects. `quadrature.py` remains the one piece of unused generality — five rules,
zero callers, and shaped wrong for the layer that would replace them; its fate is a `BACKLOG.md`
decision.

What the composition model bought is that the roadmap items are now *additive* rather than
blocked:

| Wanted (from `BACKLOG.md`) | Status |
|---|---|
| Robin BCs | **done** — the two-sided operator/load algebra, folded by `LinearProblem` |
| Time-integrator abstraction | **done** — `ThetaMethod` / `NewmarkMethod`, `dt` off the equation |
| Uniform drivers | **done** — `AdaptiveRefinement` is a class, `TopologyOptimizer` mutates nothing |
| Selectable kinematics | `SmallStrain` / `StVenantKirchhoff` exist and are tested; choosing between them is still a test-only injection, not an equation-level axis |
| Quadratic / higher-order elements | DOFs assumed one-per-vertex; needs a real quadrature layer |
| Variable coefficients / a `LinearForm` | assembly uses closed-form linear-simplex integrals; needs the quadrature hook |
| Time-varying loads / BCs | loads are built once; the field callables take position only, no `t` |
| Iterative solvers + preconditioning | the SPD operators are in place; the direct sparse factorization is now the scaling limit |

The pattern still holds: the flexibility that was added speculatively was all *lateral* (more
string options on existing operations), while the extension that remains is *vertical* (new layers
— quadrature, higher-order — between existing ones). Speculative generality widens; real extension
deepens.

---

## The pattern, found repeatedly

`fem/regions.py` + `fem/boundary.py` is the original model: a **mesh-independent specification**
(`BoundaryConditions`, a list of `(type, region, value)`) cleanly separated from its **resolution
against one discretization** (`ResolvedBC`, frozen, keyed by mesh *and* component count). It
detects conflicts rather than letting last-write-win, refuses what it cannot honour, and its
docstring explains *why* the split exists.

The same shape — a derived, immutable object keyed by the discretization, replacing mutable state
that used to drift — has since been arrived at independently four more times, which is the argument
that it is the right shape here rather than a stylistic preference:

- `FunctionSpace` is `ResolvedBC` for the discretization.
- `Form` is the resolved, assembly-ready view of an `Equation`'s physics — `LinearElasticForm(material)`
  derived from a `LinearElastic`, holding no mutable state.
- `Problem` is the resolved view of a whole composition, and `Solution` is the typed, immutable
  result the strategy hands back.

You found the right shape once, and the composition model is that shape applied all the way up the
stack.
