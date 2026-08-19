# Architecture: the object model

The package's object model: which concepts exist and which object owns each job. Anchored on symbol
names rather than line numbers, which drift with every refactor. Open work (numerics, performance,
small refactors) lives in `BACKLOG.md`.

---

## The thesis in one paragraph

A solve is not a method you look up by PDE, it is a **composition** you assemble from parts. The
package has the parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedBC`), the
object that *holds* a composition (`Problem`), the strategies that *consume* one (`LinearSolve`,
`NewtonSolve`, the time integrators), and the drivers that *wrap* a strategy to re-solve
(`AdaptiveRefinement`, `TopologyOptimizer`). A transient problem is a steady operator paired with a
time integrator (`problem.heat(...)`, `problem.wave(...)`), not a PDE type, so `Equation` carries
only the *identity* of a PDE plus its physical constants. The line the `Equation` docstring draws,
"*what* to solve" vs "*how*", is the `Problem` / strategy boundary, made structural.

---

## 1. The natural layering of an FEM code

These are the concepts the domain actually has. Most mature FEM libraries converge on some version
of this, not by fashion but because each layer varies independently of the others.

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
neighbours. Swap dense→sparse (6) without touching physics (3); swap heat's θ-method for backward
Euler (7) without touching the operator (3); remesh (1) without re-resolving constraints by hand
(5). Each is a local change, which is the point of the object model that follows.

## 2. The four tiers of a solve

The layers above are the *concepts*; the tiers below are how the objects stack. Each tier consumes
the one beneath it and varies independently.

| Tier | Role | Objects |
|---|---|---|
| **1 · Primitives** | the parts a composition is built from | `Form` / `EnergyForm` (+ `ScaledForm`, `MaskedMassForm`), `Material`, `FunctionSpace`, `BoundaryConditions` / `ResolvedBC`, `DiscreteSystem` + `Backend` |
| **2 · `Problem`** | a composition: space + operator + load + constraints ("what to solve") | `LinearProblem`, `EnergyProblem`; named factories `poisson`, `linear_elastic`, `heat`, `wave`, `projection` |
| **3 · Solve strategy** | consumes a `Problem`, returns the solution ("how") | `LinearSolve`, `NewtonSolve`; time integrators `ThetaMethod`, `NewmarkMethod` |
| **4 · Driver** | wraps a strategy, re-solving | `AdaptiveRefinement`, `TopologyOptimizer` |

Tier 3 has a second, orthogonal axis: the strategy picks linear vs. Newton, a `Backend` picks direct
vs. iterative (§4).

Named PDEs survive as **factory functions**, not dispatch keys: `poisson(mesh, f, bc)` builds the
space and returns `LinearProblem(space, LaplacianForm(), f, bc)`. You do not *dispatch* Poisson; you
*are* Poisson when your operator is a Laplacian and your load is a source. A PDE with no name in any
taxonomy (advection-diffusion, a Robin-loaded plate) is just a different composition, still fully
typed, so nothing is given up against the repo's "typed over stringly-typed" rule.

## 3. Where the classes sit

`█` = owns the layer · `▒` = shares it cleanly with another owner.

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh` | █ | | | | | | | | |
| `Element` / `ElementGeometry` | | ▒ | | | | | | | |
| `FunctionSpace` | | ▒ | | █ | | | | | ▒ |
| `FieldShape` (`Scalar` / `Vector`) | | | ▒ | | | | | | |
| `Form` / `EnergyForm` (+ `ScaledForm`, `MaskedMassForm`) | | | █ | ▒ | | | | | ▒ |
| `invariants` | | | | | | | | | █ |
| `Material` / energy densities | | | █ | | | | | | |
| `Equation` | | | █ | | | | | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `Problem` (`LinearProblem` / `EnergyProblem`) | | | ▒ | ▒ | █ | | | | |
| `DiscreteSystem` | | | | | ▒ | █ | | | |
| `Backend` (`DirectBackend` / `IterativeBackend`) | | | | | | █ | | | |
| `LinearSolve` / `NewtonSolve` | | | | | | ▒ | | | |
| `ThetaMethod` / `NewmarkMethod` | | | | | | ▒ | █ | | |
| `Solver` (steady facade) | | | | | ▒ | ▒ | | | |
| `EnergySolver` | | | | | ▒ | ▒ | | | |
| `AdaptiveRefinement` / `TopologyOptimizer` | | | | | | | | █ | ▒ |
| `Solution` (typed) | | | | | | | ▒ | | █ |
| `RedGreenRefiner` | █ | | | | | | | | |
| `Plotter` / `io` | | | | | | | | | █ |

Read the columns. Constraints (5) has one owner: the `Problem`, whose constructor resolves the
boundary conditions against the space and folds the Dirichlet partition, the Neumann load, and any
Robin contribution into the operator and load. Algebra (6) is split: `DiscreteSystem` owns the
Dirichlet elimination (every solve strategy sits on it rather than re-deriving it), and the
`Backend` owns how the remaining free-free block is solved. Time (7) is owned by the integrators.
Drivers (8) each own a solver.

Physics (3) is owned by the physics layer alone: an `Equation` answers "which form do I assemble?"
(`operator`) and "which strain energy do I minimise?" (`energy_density`) itself, so no facade holds a
mapping from equation to material. The one place a solver reads its equation is `Solver._backend_for`,
which hands an elastic AMG solve its rigid-body near-kernel, a `▒` share because that near-kernel is a
property of *which* operator is being solved.

Post-processing (9) is **distributed on purpose**, under one rule:

> A derived quantity lives on the object that owns the data it needs.

So `FunctionSpace` owns `integrate` / `mean_value` (they need the mass matrix) and `recover_nodal`
(it needs the element measures to average a per-element field onto the nodes); a `Form` owns
`derived_fields` (it needs `B` and `D`, or the energy's derivative chain); `Equation` owns
`derived_field` (which physics is the recoverable flux, `fem.postprocess.DerivedField`); `Solution`
owns the packaging (`ScalarFieldSolution.flux`, `ElasticSolution.stress`, and the `nodal_*` recoveries
built on `recover_nodal`) and `deformed_mesh`; `invariants` owns the frame-independent reductions,
which need only a tensor. The rule is what keeps this from being a junk drawer, and what made a
misplacement visible when the nodal recovery was found sitting on `Mesh`, a geometry object with no
measures to weight by.

---

## 4. Role-by-role

### `Mesh` / `FunctionSpace`: geometry vs discretization

`Mesh` is geometry: vertices, elements, boundary, topology queries. `FunctionSpace` has a mesh and
owns the discretization: element geometry, DOF numbering, cached operators. Two spaces can share one
domain, the property that made the split necessary. `assemble` takes a `Form` rather than an untyped
material bag, and `fem/mesh` imports no plot code, so the geometry layer stays clean of the core →
plot dependency.

### `Form` / `Material`: the physics, and its one open axis

The constitutive law lives on the form rather than the element, and **every assembly path goes
through a form**:

- **Bilinear forms** (`MassForm`, `LaplacianForm`, `LinearElasticForm`, the `Gᵀ C G · volume`
  stiffness family) scatter through `FunctionSpace.assemble`, one loop that does not know what it is
  scattering. `ScaledForm(c², form)` and `MaskedMassForm(mask)` are combinators that exist because a
  term needed them (the wave operator's `c²K`, the Robin boundary integral); `PrecomputedForm` is the
  escape hatch for a driver that can derive its element matrices more cheaply than by re-integrating
  (SIMP rescales one set by `rho^p`).
- **The nonlinear energy path** is `EnergyForm`, the sibling mapping an element *and a state* to an
  energy, residual, and tangent; it scatters through `assemble_residual` / `assemble_tangent`, which
  `EnergyProblem` calls. A quadratic energy has a constant tangent, so the bilinear `Form` is
  `EnergyForm`'s state-independent special case.
- **Stress recovery is on the form**, as the `RecoversElasticFields` capability.
  `derived_fields(geometry, u_elements)` is the mirror of `element_matrices`: the same physics
  contracted against the solved displacement instead of assembled into a stiffness. Both elastic paths
  implement it, so a finite-strain solve reports the stress state a small-strain one does, and two
  implementations are what earn the protocol its place.

**Tensors, not Voigt vectors, cross the assembly boundary.** Voigt packing is valid only under the
`Bᵀ D B` contraction it was designed for; every other operation on the packed form is wrong (reducing
a Voigt vector with a plain norm changes a stress scalar by √2 under a frame rotation). So
`derived_fields` unpacks to full `(n_elements, 3, 3)` tensors before returning, `fem/invariants.py`
operates on tensors alone, and `tests/test_invariants.py` pins invariance by rotating the input.

`Material` owns `D`, and the strain-displacement matrix `B` sits in `fem/forms.py` next to the form
that contracts it against `D`, which keeps `Element` pure geometry. The physics layer decomposes as
**material** (the energy `W`) × **kinematics** (the strain measure): the two solver paths feed one `W`
either Green-Lagrange `S` (St-VK) or the small-strain `ε`. Both are named (`SmallStrain`,
`StVenantKirchhoff`) and pinned in `tests/test_elasticity_models.py`, and choosing the kinematics is an
equation-level choice (`LinearElastic(kinematics=...)`). The linear path assembles a constant stiffness,
so it accepts only `SMALL` and rejects finite strain rather than silently linearising it.

### `Element`: stateless types, batched geometry

Element types are stateless: `LinearTetrahedralElement` describes a shape and holds no per-element
data, so there is one of them in a program rather than one per tet. The per-element data lives in
`ElementGeometry`, which holds it for the whole mesh at once, and `Form.element_matrices` computes
every element matrix in a single vectorized pass. The type/instance split *is* the batching.
`EnergyForm` is batched and dimension-general too, so `EnergySolver` accepts 3D meshes.

### `Problem`: the narrow waist between physics and algebra

A `Problem` is the assembly-ready composition for **one mesh**: a space, an operator, a load, and
constraints. Two shapes share one protocol, mirroring the `Form` / `EnergyForm` split:

- `LinearProblem`: `tangent(u) = A` (constant), `residual(u) = A·u − b`. Its constructor assembles
  the operator, folds the Neumann load into `b`, and folds each Robin contribution into *both* `A` and
  `b`, the term Robin needs.
- `EnergyProblem`: `tangent(u) = ∇²Π(u)`, `residual(u) = ∇Π(u)`, the state-dependent sibling.

The `Problem` **owns its constraints**: nothing index-keyed is carried across a mesh change, because a
driver that remeshes just builds a new `Problem`. No solver re-resolves boundary conditions by hand,
and no stale DOF partition can outlive the mesh it was built for.

### Solve strategies: `LinearSolve`, `NewtonSolve`, one engine

Every strategy sits on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet partition +
factor-once solve), and knows nothing about which PDE produced the `Problem`. `LinearSolve` assembles
once and solves once; `NewtonSolve` iterates, re-factoring the tangent each step. A `LinearProblem`
has a constant tangent and an affine residual, so `NewtonSolve` reaches its solution in a single step
from any seed, and `LinearSolve` is that step done directly. An optional `BacktrackingLineSearch`
globalizes the step length: it scales each increment to decrease a merit (the problem's energy Π(u)
via `SupportsEnergy`, else ½‖r‖²), so a non-convex St-VK solve converges from a seed a full step would
send diverging, at no cost near the solution where α = 1. `SolveStrategy` is the protocol both satisfy,
and `TopologyOptimizer` takes one as an injectable parameter.

### `Backend`: the second, orthogonal axis of the solve

Which strategy runs and which linear algebra it uses vary independently, so they are two injections
rather than a class per combination: `LinearSolve(IterativeBackend())` is a composition, not an
`IterativeLinearSolve`. `DiscreteSystem` eliminates the Dirichlet DOFs and hands the free-free block to
a `Backend`, which `prepare`s it into a `LinearSolver`: one matrix, factored or preconditioned once,
solved against many right-hand sides (what a time-stepper or constant-tangent Newton loop reuses).

- `DirectBackend`: sparse LU (`splu`), the default, robust for any nonsingular operator. Its fill-in
  on a 3D mesh grows super-linearly, which is the resolution ceiling.
- `IterativeBackend`: CG with a smoothed-aggregation AMG preconditioner (`pyamg`). CG is **SPD-only**,
  so it is opt-in: Poisson, small-strain elasticity, the mass matrix and the time-stepping operators
  qualify; Newton tangents away from a minimum do not. For vector elasticity, `rigid_body_modes`
  supplies the near-kernel that keeps CG's iteration count flat under refinement.

`LinearSolve`, `ThetaMethod`, and `NewmarkMethod` take a `backend`; `NewtonSolve` does not, because a
Newton tangent is not guaranteed SPD. The choice is offered only where the operator is SPD by
construction. `pyamg` is a base dependency because the iterative path is the core scaling story.

### `EigenSolve`: the solves that are not `Ax = b`

Linearised buckling and modal (free-vibration) analysis step outside the `DiscreteSystem` engine,
because an eigenproblem is a different question. `K φ = -λ K_g φ` (buckling) and `K φ = ω² M φ` (modal)
have no right-hand side to back-substitute, so the factor-once-solve-many atom does not apply. What
they *do* share with a linear solve is the Dirichlet elimination, so that reduction, the
`scipy.sparse.linalg.eigsh` call, and the lift of each eigenvector back to a full DOF vector live in one
place: `EigenSolve` in `fem/solve.py`. `BucklingSolver` and `ModalSolver` are thin facades over it, each
assembling its operator pair and reading the eigenvalues in its own physics (`μ = 1/λ` load factors,
`μ = ω²` frequencies).

Everything upstream is reused unchanged: the buckling reference state comes from an ordinary `Solver`,
the geometric stiffness is a `Form` (`GeometricStiffnessForm`) scattered by the same
`FunctionSpace.assemble`, and it is exactly `term1` of the St-VK tangent, so the nonlinear machinery
already carried its kernel. Modal analysis reuses the same seam for a different pencil: no reference
solve (the modes are load-free), the consistent mass matrix as the second operator, and shift-invert
about `σ = 0` to pull the lowest frequencies. Its supports must ground every rigid-body mode, since the
shift-invert factors `K` on the free block.

### Time integration: a strategy per ODE order

The domain has **two ODE orders** (heat is first, `M u̇ + K u = b`; wave is second, `M ü + c²K u = b`),
and that split is the real structure, so there is one integrator family per order rather than one
first-order interface for both. `dt` and the step count are constructor data of the integrator, not
fields on an equation; initial conditions arrive through `run(...)`. Each forms a *constant* effective
operator from the problem's mass and stiffness, factors it once through `DiscreteSystem`, and steps by
updating only the right-hand side. Both take a `Backend`, since those effective operators are SPD.

- `ThetaMethod` (θ=½ Crank-Nicolson default, θ=1 backward Euler) solves `(M + θ dt K) u_{n+1} = …`.
- `NewmarkMethod` (β=¼, γ=½) solves for the acceleration against the SPD, N-sized `M + β dt² K`, **not**
  a 2N first-order block. The wave speed lives in the operator as `ScaledForm(c², …)`, so the integrator
  sees only `c²K`; keeping the operator SPD is what lets a wave step run on `IterativeBackend`, which a
  first-order block reformulation would not. `wave_energy` is the invariant this scheme conserves for a
  linear system, kept as a diagnostic.

### `Equation`: identity, physical constants, and the physics they imply

`Equation` is typed data: it says *what* to solve and carries the genuinely physical parameters
(`E`/`nu` on `LinearElastic`), while a strategy owns *how*. `Projection`, `Poisson`, and `LinearElastic`
are the members; a transient problem is a steady operator paired with an integrator, not a distinct PDE
type, so there is no `Heat` or `Wave` class. It also answers what its constants *mean*, one method per
consumer: `operator(...)` returns the bilinear `Form` the linear path assembles, `energy_density()` the
density the nonlinear path differentiates, and `flux()` names the field an error estimator jumps or
recovers. The first two refuse rather than approximate (a Green-Lagrange `LinearElastic` has no constant
stiffness, so no `operator`; a scalar equation has no stored energy, so no `energy_density`), which is
why neither facade holds a `_select_*` mapping. The estimator *algorithm* lives apart in
`fem/estimators.py`: the equation only names its flux, and `fem/equations.py` is its own module because
both facades consume equations, so neither owns them.

### `Solver` / `EnergySolver`: thin facades over the core

The two are deliberately the same shape: hold a mesh, an equation, and a BC spec; build a `Problem` per
solve; hand it to a strategy. `Solver` builds a `LinearProblem` for `LinearSolve`; `EnergySolver` builds
an `EnergyProblem` for `NewtonSolve`. Both expose `remesh(mesh)`, which rebuilds the derived state from
the mesh-independent specification, the seam a driver advances them through. Neither holds anything
index-keyed, so a stale partition cannot outlive a refinement. They differ in one way on purpose:
`Solver` takes a `Backend` and `EnergySolver` does not, because the energy path runs through
`NewtonSolve`, whose tangent is not guaranteed SPD.

### Drivers: uniform outer loops

`AdaptiveRefinement` and `TopologyOptimizer` are the two studies, and they share one shape: each owns a
solver (or strategy) and re-solves. `AdaptiveRefinement` owns a `RefinableSolver` and advances it across
meshes via `remesh`; `TopologyOptimizer` owns a `SolveStrategy` and derives a fresh `LinearProblem` from
the current density each iteration via `with_operator`. Its objective is an injected object
(`MinCompliance`, `TargetCompliance`) and its result a typed `TopologyHistory`. Neither driver reaches
into a solver's internals. `RefinableSolver` is a protocol, not a concrete `Solver`, so nonlinear
elasticity refines through the same driver the Poisson path uses.

### `Solution`: typed, one dataclass per shape

The result is a typed dataclass, not a dict of named arrays: `FieldSolution` (the field `u`),
`ElasticSolution` (adds recovered strain/stress/compliance), `TransientSolution` (a time series),
`WaveSolution` (adds the velocity series), plus `BucklingSolution` / `ModalSolution`. A steady field and
a time series are different *types*, so nothing infers which it is from an array length. `save` / `load`
round-trip any of them through `fem/io`, which reflects over the dataclass fields and stores the class
name, so the I/O follows the type rather than a naming convention.

`ElasticSolution` **owns its own derivation**: `from_solve(...)` is the single place a solved
displacement becomes one, and `Solver`, `TopologyOptimizer`, and `EnergySolver` all go through it. It
stores the full stress and strain **tensors** and exposes `von_mises` / `pressure` / `principal_stress`
as properties, because reducing at construction would decide permanently which question the result can
answer. `TopologyHistory` is deliberately *not* in this hierarchy: it is a driver-layer trajectory of
designs (its axis is optimization iteration, `rho` a design variable, not a solved field), so it stays a
standalone record aggregating the per-iteration `ElasticSolution`s.

### The integration stack and the DOF numbering: where quadrature and P2 fit

The higher-order work added no new tier. It deepened two existing layers along two orthogonal axes, the
"two coupled assumptions" behind the old P1 ceiling:

- **Fact A: how a form becomes numbers** (assembly, §1 row 4). The integration stack is
  `QuadratureRule → Element → ElementGeometry → Form`: `QuadratureRule` is reference-simplex data alone,
  `Element` is the stateless bridge to physical space (`shape_values` / `shape_gradients`,
  `quadrature(degree)`, `geometry(coords, rule)`), and its batched `ElementGeometry` carries
  `grad_phi (n_el, n_qp, N, spatial)`, `weight_detJ`, and `points`. The new quadrature-point axis on
  `grad_phi` *is* Fact A; P1 is the `n_qp == 1` special case, which is why the same forms serve both
  orders.
- **Fact B: where the DOFs live** (discretization, §1 row 2). Nothing reads DOFs off the mesh any more.
  `FunctionSpace` exposes `element_nodes` / `node_coords` / `boundary_nodes`; for P1 these are the mesh's
  own arrays, for P2 the `NodeSet` that `p2_connectivity` builds (vertices then one edge-midpoint node
  per edge). `BoundaryConditions.resolve` takes a `NodeGeometry` (which `Mesh` and `NodeSet` both
  satisfy), so a geometric condition pins the edge DOFs with no change to the resolver.

The two axes meet at **`FunctionSpace.assemble`**, the one place both are read:

```
1. degree = form.quadrature_degree (or the element's default)
2. ElementGeometry = element.geometry( node_coords[element_nodes], rule(degree) )   # A over B
3. blocks = form.element_matrices(ElementGeometry)                                  # A
4. _ScatterPlan( dof_indices(element_nodes) ).scatter(blocks)                       # B
```

So `FunctionSpace` holds `mesh + element + n_components` and produces both outputs, the geometry (cached
per rule in `geometry_at`) and the DOF map, and every form flows through this path, which is what kept
the P1 numbers identical while the machinery under them changed.

---

## The pattern, found repeatedly

`fem/regions.py` + `fem/boundary.py` is the original model: a **mesh-independent specification**
(`BoundaryConditions`, a list of `(type, region, value)`) cleanly separated from its **resolution
against one discretization** (`ResolvedBC`, frozen, keyed by mesh *and* component count). It detects
conflicts rather than letting last-write-win, refuses what it cannot honour, and its docstring explains
*why* the split exists.

The same shape, a derived immutable object keyed by the discretization, replacing mutable state that
would otherwise drift, recurs four more times, which is the argument that it is the right shape here
rather than a stylistic preference:

- `FunctionSpace` is `ResolvedBC` for the discretization.
- `Form` is the resolved, assembly-ready view of an `Equation`'s physics, and `Equation.operator` is the
  derivation itself.
- `Problem` is the resolved view of a whole composition, and `Solution` is the typed, immutable result
  the strategy hands back.
- `LinearSolver` is a `Backend` resolved against one assembled matrix: the immutable config, separated
  from the factorization or preconditioner that one operator produced.

You found the right shape once, and the composition model is that shape applied all the way up the stack.
