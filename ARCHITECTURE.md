# Architecture — the object model

The package's object model: which concepts exist and which object owns each job, with the open
structural and dead-weight items collected at the end. Anchored on symbol names rather than line
numbers, which drift with every refactor.

The numeric roadmap that remains — quadrature, higher-order elements, an error estimator, a
hand-rolled multigrid preconditioner — lives in `BACKLOG.md`.

---

## The thesis in one paragraph

A solve is not a method you look up by PDE — it is a **composition** you assemble from parts.
The package has the parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedBC`),
the object that *holds* a composition (`Problem`), the strategies that *consume* one
(`LinearSolve`, `NewtonSolve`, the time integrators), and the drivers that *wrap* a strategy to
re-solve (`AdaptiveRefinement`, `TopologyOptimizer`). A transient problem is a steady operator
paired with a time integrator — `problem.heat(...)`, `problem.wave(...)` — not a PDE type, so
`Equation` carries only the *identity* of a PDE plus its physical constants. The line the
`Equation` docstring draws — "*what* to solve" vs "*how*" — is the `Problem` / strategy boundary,
made structural.

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
by hand (5). Each is a local change — the point of the object model that follows.

## 2. The four tiers of a solve

The layers above are the *concepts*; the tiers below are how the objects stack. Each tier
consumes the one beneath it and varies independently.

| Tier | Role | Objects |
|---|---|---|
| **1 · Primitives** | the parts a composition is built from | `Form` / `EnergyForm` (+ `ScaledForm`, `MaskedMassForm`), `Material`, `FunctionSpace`, `BoundaryConditions` / `ResolvedBC`, `DiscreteSystem` + `Backend` |
| **2 · `Problem`** | a composition: space + operator + load + constraints ("what to solve") | `LinearProblem`, `EnergyProblem`; named factories `poisson`, `linear_elastic`, `heat`, `wave`, `projection` |
| **3 · Solve strategy** | consumes a `Problem`, returns the solution ("how") | `LinearSolve`, `NewtonSolve`; time integrators `ThetaMethod`, `NewmarkMethod` |
| **4 · Driver** | wraps a strategy, re-solving | `AdaptiveRefinement`, `TopologyOptimizer` |

Tier 3 has a second, orthogonal axis: the strategy picks linear vs. Newton, a `Backend` picks
direct vs. iterative (§4).

Named PDEs survive as **factory functions**, not dispatch keys: `poisson(mesh, f, bc)` builds the
space and returns `LinearProblem(space, LaplacianForm(), f, bc)`. You do not *dispatch* Poisson; you *are* Poisson
when your operator is a Laplacian and your load is a source. A PDE with no name in any taxonomy
(advection–diffusion, a Robin-loaded plate) is just a different composition, not a new registry
entry — and the composition is still fully typed, so nothing is given up against the repo's
"typed over stringly-typed" rule.

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

Read the columns. Constraints (5) has one clear owner: the `Problem`, whose constructor resolves
the boundary conditions against the space and folds the Dirichlet partition into its
`constraints`, the Neumann load into its load vector, and any Robin contribution into *both*
sides. Algebra (6) is split in two: `DiscreteSystem` owns the Dirichlet elimination — every solve
strategy sits on it rather than re-deriving that — and the `Backend` owns how the remaining
free-free block is solved. Time (7) is owned by the integrators. Drivers (8) are uniform — each
owns a solver.

Physics (3) is owned by the physics layer alone: an `Equation` answers "which form do I assemble?"
(`operator`) and "which strain energy do I minimise?" (`energy_density`) itself, so no facade holds
a mapping from equation to material and `Solver._steady_problem` needs no "which PDE is this?"
branch.

The one place a solver reads its equation is `Solver._backend_for`, which hands an elastic AMG
solve its rigid-body near-kernel — a `▒` share, because the near-kernel is a property of *which*
operator is being solved, so only the equation-aware facade can supply it without the
physics-agnostic backend guessing.

Post-processing (9) is **distributed on purpose**, under one rule:

> A derived quantity lives on the object that owns the data it needs.

So `FunctionSpace` owns `integrate` and `mean_value` (they need the mass matrix) and
`element_to_vertex` (it needs the element measures); a `Form` owns `derived_fields` (it needs `B`
and `D`, or the energy's derivative chain); `Solution` owns the packaging and `deformed_mesh`;
`invariants` owns the frame-independent reductions, which need only a tensor. The rule is what
keeps this from being a junk drawer — and what makes a misplacement visible, which is how the
element↔vertex projection was found sitting on `Mesh`, a geometry object with no measures to
weight by. §5 lists what the rule has not yet been applied to.

---

## 4. Role-by-role

### `Mesh` / `FunctionSpace` — geometry vs discretization

`Mesh` is geometry: vertices, elements, boundary, topology queries. `FunctionSpace` has a mesh
and owns the discretization — element geometry, DOF numbering, cached operators. Two spaces can
share one domain, which is the property that made the split necessary. `assemble` takes a `Form`
rather than an untyped material bag, so the space forwards nothing it cannot interpret. `fem/mesh`
imports no plot code, so the geometry layer is clean of the core → plot dependency.

### `Form` / `Material` — the physics, and its one open axis

The constitutive law lives on the form rather than the element, and **every assembly path goes
through a form**:

- **Bilinear forms** — `MassForm` (`∫u·v`), `LaplacianForm` and `LinearElasticForm` (the
  `Gᵀ C G · volume` stiffness family) — scatter through `FunctionSpace.assemble`, one loop that
  does not know what it is scattering. `ScaledForm(c², form)` and `MaskedMassForm(mask)` are the
  two combinators that exist because a term needed them (the wave operator's `c²K`, the Robin
  boundary integral); no speculative `OperatorSum` waits ahead of a second use. `PrecomputedForm`
  is the escape hatch beside them, for a driver that can derive its element matrices more cheaply
  than by re-integrating them — SIMP rescales one set by `rho^p`.
- **The nonlinear energy path** is `EnergyForm`, the sibling that maps an element *and a state*
  to an energy, residual, and tangent; the energy path scatters it through
  `FunctionSpace.assemble_residual`/`assemble_tangent`, which `EnergyProblem` calls. A quadratic energy has a constant tangent, so the
  bilinear `Form` is `EnergyForm`'s state-independent special case.
- **Stress recovery is on the form**, as the `RecoversElasticFields` capability.
  `derived_fields(geometry, u_elements)` is the mirror of `element_matrices`: the same physics
  contracted against the solved displacement instead of assembled into a stiffness. Both elastic
  paths implement it — `LinearElasticForm` from its `B` and `D`, `EnergyForm` by pushing its
  `dW_dF` (the first Piola–Kirchhoff stress, previously computed each Newton step and discarded)
  forward to Cauchy — so a finite-strain solve reports the stress state a small-strain one does.
  Two implementations are also what earn the protocol its place; `Solver` asks for the capability
  rather than naming a form class.

**Voigt stops at the assembly boundary.** Voigt packing stores a symmetric tensor as a vector so
that an element stiffness is the matrix product `Bᵀ D B`. It is valid *only* under the contraction
it was designed for: strain packs engineering shear (`γ = 2ε`) and stress does not, an asymmetry
that makes the Voigt dot product equal the tensor double contraction and makes every other
operation on the packed form wrong. `derived_fields` therefore unpacks to full `(n_elements, 3, 3)`
tensors before returning, and `fem/invariants.py` operates on tensors alone. That boundary is not
stylistic: reducing a Voigt vector with `np.linalg.norm` — which is what both call sites used to do
— counts the off-diagonal terms once where the tensor holds them twice, giving a stress scalar that
changes by √2 when the coordinate frame rotates. `tests/test_invariants.py` pins invariance by
rotating the input.

`Material` owns `D`, and the strain-displacement matrix `B` sits in `fem/forms.py` next to the
form that contracts it against `D`, which is what keeps `Element` pure geometry.

The two constitutive representations are the same material, *pinned* rather than asserted:
`energies.py`'s `calculate_W_from_S` and the `½εᵀDε` implied by `Material` are one energy
`W(ε) = ½λ(tr ε)² + μ tr(εᵀε)`, and a test checks `D = ∂²W/∂ε²` in 2D. `D` is left in its
Lamé-parameter closed form rather than derived from `W` on purpose: the closed form is
dimension-general, whereas `energies.py` is fixed-rank-2, so deriving `D` from the density would
forfeit the 3D path. The other axis is **kinematics**: the two solver paths differ only in the
strain measure fed to that one `W` — Green–Lagrange `S = ½(FᵀF − I)` (St-VK) versus the
small-strain `ε`. Both are named (`SmallStrain`, `StVenantKirchhoff`) and pinned in
`tests/test_elasticity_models.py`. So the physics layer decomposes as **material** (the energy
`W`) × **kinematics** (the strain measure), and choosing the kinematics point is an equation-level
choice: `LinearElastic(kinematics=StrainMeasure.SMALL | GREEN_LAGRANGE)`, which
`LinearElastic.energy_density` maps to the density. The linear path assembles a constant stiffness,
so `LinearElastic.operator` accepts only `SMALL` and rejects finite strain rather than silently
linearising it.

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
  *both* `A` (`κ·` boundary mass) and `b` (the `∫g·v` term) — a term that contributes to the
  operator and the load at once, which is what Robin needs.
- `EnergyProblem` — `tangent(u) = ∇²Π(u)`, `residual(u) = ∇Π(u)`, the state-dependent sibling.

The `Problem` **owns its constraints**: nothing index-keyed is carried across a mesh change,
because a driver that remeshes just builds a new `Problem`. No solver has to re-resolve boundary
conditions by hand, and no stale DOF partition can outlive the mesh it was built for.
`LinearProblem` is the special case of `EnergyProblem` where the tangent does not depend on `u`,
exactly the relationship `Form` has to `EnergyForm`.

### Solve strategies — `LinearSolve`, `NewtonSolve`, one engine

Every strategy sits on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet partition +
factor-once solve), and knows nothing about which PDE produced the `Problem`. `LinearSolve`
assembles once and solves once; `NewtonSolve` iterates, re-factoring the tangent each step and
checking convergence before applying the increment. A `LinearProblem` has a constant tangent and
an affine residual, so `NewtonSolve` reaches its solution in a single applied step from any seed —
`LinearSolve` is that step done directly. The two are one engine: `EnergySolver`'s Newton loop *is*
`NewtonSolve`, applied to an `EnergyProblem`. `SolveStrategy` is the protocol both satisfy, and
`TopologyOptimizer` takes one as an injectable parameter — a driver that accepts any strategy,
which is the protocol earning its place.

### `Backend` — the second, orthogonal axis of the solve

Which strategy runs and which linear algebra it uses vary independently, so they are two
injections rather than a class per combination: `LinearSolve(IterativeBackend())` is a
composition, not an `IterativeLinearSolve`. `DiscreteSystem` eliminates the Dirichlet DOFs and
hands the free-free block to a `Backend`, which `prepare`s it into a `LinearSolver`: one matrix,
factored or preconditioned once, solved against many right-hand sides. That is what a time-stepper
or a constant-tangent Newton loop reuses across steps. Config and bound solver are two objects
because the matrix actually solved is born *inside* `DiscreteSystem` — a caller can only hand in
a recipe, never the solver itself.

- `DirectBackend` — sparse LU (`splu`), the default, robust for any nonsingular operator,
  indefinite ones included. Its fill-in on a 3D mesh grows super-linearly, which is the
  resolution ceiling.
- `IterativeBackend` — CG with a smoothed-aggregation AMG preconditioner (`pyamg`). CG is
  **SPD-only**, so it is opt-in: Poisson, small-strain elasticity, the mass matrix and the
  time-stepping operators qualify; Newton tangents away from a minimum do not. For vector
  elasticity, `rigid_body_modes` supplies the near-kernel that keeps CG's iteration count flat
  under refinement — the constant vector pyamg assumes by default does not.

`LinearSolve`, `ThetaMethod`, and `NewmarkMethod` take a `backend`; `NewtonSolve` does not,
because a Newton tangent is not guaranteed SPD. The choice is offered only where the operator is
SPD by construction. `Solver` forwards a backend to its steady solve and attaches the elastic
near-kernel (§3). `pyamg` is a base dependency rather than an extra — it needs only numpy/scipy,
and the iterative path is the core scaling story, not a niche feature.

### `EigenSolve` — the solve that is not `Ax = b`

Linearised buckling steps outside the `DiscreteSystem` engine, because an eigenproblem is a
different question, not a different matrix. `K φ = -λ K_g φ` has no right-hand side to
back-substitute, so the factor-once-solve-many atom every other strategy sits on does not apply.
What it *does* share with a linear solve is the Dirichlet elimination — the free-DOF reduction —
so that reduction, the `scipy.sparse.linalg.eigsh` call, and the lift of each eigenvector back to
a full DOF vector live in one place: `EigenSolve`, the eigen-analogue of `LinearSolve` in
`fem/solve.py`. `BucklingSolver` is a thin facade over it, assembling `K` and `K_g` and reading
the eigenvalues back as load factors (`μ = 1/λ`). Everything upstream is reused
unchanged: the reference (pre-buckling) state comes from an ordinary `Solver`, the geometric
stiffness is a `Form` (`GeometricStiffnessForm`) scattered by the same `FunctionSpace.assemble`,
and the prestress that parameterises it is the recovered stress a linear elastic solve already
produces. The one physics primitive that is new — the geometric stiffness — is exactly `term1`
of the St-Venant–Kirchhoff tangent (`EnergyForm.element_tangents`), so the nonlinear machinery
already carried its kernel; buckling is the linearisation of the ellipticity loss that tangent
models under compression.

### Time integration — a strategy per ODE order

The domain has **two ODE orders** — heat is first (`M u̇ + K u = b`), wave is second
(`M ü + c²K u = b`) — and that split is the real structure, so there is one integrator family per
order rather than one first-order interface for both. `dt` and the step count are constructor data
of the integrator, not fields on an equation; initial conditions arrive through `run(...)`. Each
forms a *constant* effective operator from the problem's mass and stiffness, factors it once
through `DiscreteSystem`, and steps by updating only the right-hand side. Both take a `Backend`,
since those effective operators are SPD.

- `ThetaMethod` (θ=½ Crank–Nicolson default, θ=1 backward Euler) solves `(M + θ dt K) u_{n+1} = …`.
- `NewmarkMethod` (β=¼, γ=½ average-acceleration) solves for the acceleration against the SPD,
  N-sized `M + β dt² K` — **not** a 2N first-order block. The wave speed lives in the operator as
  `ScaledForm(c², …)`, so the integrator sees only `c²K` and never learns `c`; constant Dirichlet
  displacement means zero velocity and acceleration at fixed nodes, so those DOFs are the ordinary
  constraint, with no lifting into a block DOF space. Keeping the operator SPD is what lets a wave
  step run on `IterativeBackend`; a first-order block reformulation would not be. `wave_energy` is
  the invariant this scheme conserves for a linear system, kept as a diagnostic.

### `Equation` — identity, physical constants, and the physics they imply

`Equation` is typed data: it says *what* to solve and carries the genuinely physical parameters
(`E`/`nu` on `LinearElastic`), while a strategy owns *how*. `Projection`, `Poisson`, and
`LinearElastic` are the members; a transient problem is a steady operator paired with an
integrator (`problem.heat(...)`, `problem.wave(...)`), not a distinct PDE type, so there is no
`Heat` or `Wave` class. `Equation` carries no time-discretization parameters and no mutable
material — `TopologyOptimizer` builds a fresh material each iteration — so it is immutable
specification.

It also answers what its constants *mean*, along one method per assembly path: `operator(...)`
returns the bilinear `Form` the linear path assembles (`MassForm` for a projection, the
material-free `LaplacianForm` for the scalar family, a `LinearElasticForm` built from its own
`E`/`nu`), and `energy_density()` returns the density the nonlinear path differentiates. Both
refuse rather than approximate: a Green–Lagrange `LinearElastic` has no constant stiffness, so it
has no `operator`, and a scalar equation has no stored energy, so it has no `energy_density`. That
is why neither facade holds a `_select_*` mapping, and it is the natural home for the per-equation
error estimator the backlog wants.

`fem/equations.py` is its own module for the same reason: both facades consume equations, so
neither owns them.

### `Solver` / `EnergySolver` — thin facades over the core

The two are deliberately the same shape: hold a mesh, an equation, and a BC spec; build a
`Problem` per solve; hand it to a strategy. `Solver` builds a `LinearProblem` for `LinearSolve`;
`EnergySolver` builds an `EnergyProblem` for `NewtonSolve`. Both expose `remesh(mesh)`, which
rebuilds the derived state from the mesh-independent specification — the seam a driver advances
them through without reaching into their state.

Neither holds anything index-keyed. The DOF partition belongs to the `Problem` built for the
current mesh, so a stale partition cannot outlive a refinement; `EnergySolver` seeds its Newton
iteration from `problem.constraints`.

They differ in one way on purpose: `Solver` takes a `Backend` and `EnergySolver` does not, because
the energy path runs through `NewtonSolve`, whose tangent is not guaranteed SPD (§4, `Backend`).

### Drivers — uniform outer loops

`AdaptiveRefinement` and `TopologyOptimizer` are the two studies, and they share one shape: each
owns a solver (or strategy) and re-solves. `AdaptiveRefinement` owns a `RefinableSolver` and
advances it across meshes via `remesh`; `TopologyOptimizer` owns a `SolveStrategy` and derives a
fresh `LinearProblem` from the current density each iteration, via `with_operator`, over the
constraints and load the density does not reach. Its objective is an injected object
(`MinCompliance`, `TargetCompliance`) and its result a typed `TopologyHistory`. Neither driver
reaches into a solver's internals: adaptivity uses `remesh` and `solve`, and the optimizer only
builds and solves fresh `Problem`s.

`RefinableSolver` is a protocol, not a concrete `Solver`: adaptivity needs a mesh, a BC spec,
`remesh`, and `solve`, and both facades satisfy that, so nonlinear elasticity refines through the
same driver the Poisson path uses.

### `Solution` — typed, one dataclass per shape

The result is a typed dataclass, not a dict of named arrays: `FieldSolution` (the field `u`),
`ElasticSolution` (adds recovered strain/stress/compliance), `TransientSolution` (a time series),
`WaveSolution` (adds the velocity series). A steady field and a time series are different *types*,
so nothing has to infer which it is from the length of an array. `save`/`load` round-trip any of
them through `fem/io`, which reflects over the dataclass fields and stores the class name — so the
I/O follows the type rather than a naming convention.

`ElasticSolution` **owns its own derivation**: `from_solve(mesh, n_components, u, form, geometry)`
is the single place a solved displacement becomes one, and both `Solver` and `TopologyOptimizer`
(and now `EnergySolver`) go through it. It was written out at each call site before, which is how a
reduction that was not rotation invariant came to exist in two copies. It stores the full stress
and strain **tensors** and exposes `von_mises` / `pressure` / `principal_stress` as properties: a
Frobenius norm cannot be turned back into a von Mises stress, so reducing at construction would
decide permanently which question the result can answer. `TopologyHistory` is
deliberately *not* in this hierarchy: it is a driver-layer trajectory of designs (its axis is
optimization iteration, and `rho` is a design variable, not a solved field), so it stays a
standalone record that aggregates the per-iteration `ElasticSolution`s rather than being one.

### The integration stack and the DOF numbering — where quadrature and P2 fit

The higher-order work added no new tier. It deepened two of the
existing layers along two orthogonal axes, the "two coupled assumptions" behind the old P1 ceiling:

- **Fact A — how a form becomes numbers** (the assembly layer, §1 row 4). The *integration stack*:

  ```
  QuadratureRule ──→ Element ──→ ElementGeometry ──→ Form
   (reference          (a shape:      (physical, batched   (physics:
    pts + weights)      basis fns)     geometry at a rule)   integrand)
  ```

  `QuadratureRule` is reference-simplex data alone. `Element` is the bridge to physical space: a
  stateless shape that answers `shape_values` / `shape_gradients` (the basis at reference points),
  `quadrature(degree)` (pick a rule), and `geometry(coords, rule)` (the affine, corners-only map).
  Its batched output `ElementGeometry` carries `grad_phi (n_el, n_qp, N, spatial)`, `weight_detJ`,
  and `points` — the resolved, per-mesh geometry a form integrates against. The new
  quadrature-point axis on `grad_phi` *is* Fact A; P1 is the `n_qp == 1` special case, which is why
  the same forms serve both orders.

- **Fact B — where the DOFs live** (the discretization layer, §1 row 2). Nothing reads DOFs off the
  mesh any more. `FunctionSpace` exposes `element_nodes` / `node_coords` / `boundary_nodes`; for P1
  these are the mesh's own arrays, for P2 the `NodeSet` that `p2_connectivity` builds — vertices
  then one edge-midpoint node per edge. `BoundaryConditions.resolve` takes a `NodeGeometry` (which
  `Mesh` and `NodeSet` both satisfy), so a geometric condition pins the edge DOFs with no change to
  the resolver: the boundary edge-midpoint satisfies the same region its endpoints do.

The two axes meet at **`FunctionSpace.assemble`**, the one place both are read:

```
1. degree = form.quadrature_degree (or the element's default)
2. ElementGeometry = element.geometry( node_coords[element_nodes], rule(degree) )   # A over B
3. blocks = form.element_matrices(ElementGeometry)                                  # A
4. _ScatterPlan( dof_indices(element_nodes) ).scatter(blocks)                       # B
```

So `FunctionSpace` holds `mesh + element + n_components` and produces both outputs — the geometry
(cached per rule in `geometry_at`) and the DOF map — and every form flows through this path, which
is what kept the P1 numbers identical while the machinery under them changed. `ElementGeometry` is
the resolved view of *(element, rule)* against a mesh, and `NodeSet` the resolved numbering of a P2
discretization: both are the "spec → resolved-per-discretization" shape the rest of the model uses.
Three consumers that had baked in "nodes == vertices" learned the numbering axis in one line each —
`Source.vector` samples `node_coords`, `ElasticSolution.from_solve` gathers `element_nodes`, and
`Solver` / `rigid_body_modes` build from `node_coords`.

---

## 5. What the model leaves open

The *vertical* extension the roadmap wanted — new layers between existing ones — has largely
landed: `quadrature.py` is now the real reference-element layer,
`ElementGeometry` carries a quadrature-point axis, `DiffusionForm` / `LinearForm` sample
variable coefficients and sources, and `QuadraticTriangleElement` numbers DOFs over vertices ∪
edges for O(h³). What remains is additive against the composition model rather than blocked by it:

| Wanted (from `BACKLOG.md`) | Where it sits | State |
|---|---|---|
| Quadratic / higher-order elements | DOFs over vertices ∪ edges; the quadrature-point axis | **done (2D)**; 3D P2 open |
| Variable coefficients / a `LinearForm` | `DiffusionForm`, `LinearForm`, sampled at quadrature points | **done** |
| Time-varying loads / BCs | loads are built once; the field callables take position only, no `t` | open |
| A geometric two-grid preconditioner | a `Backend` implementation; the seam exists, `pyamg` currently fills it | open |

The two open rows are each a further implementation of a seam that now exists — the two-grid one
a second `Backend`, the time-varying one a `t` argument on the field callables. Speculative
generality widens; real extension deepens.

**Post-processing is organised but not complete.** The rule in §3 has an owner for every derived
quantity that exists, and the elastic paths both report through `RecoversElasticFields` — a
capability whose result bundle (strain, stress, compliance) is elasticity's shape, so it abstracts
over linear-vs-energy elasticity rather than over physics. What the layer does not yet have is
*coverage*, and the first row below needs a sibling capability rather than an implementation of
this one:

| Gap | Where it would sit |
|---|---|
| Poisson flux `-∇u` | a `derived_fields` on the scalar family's form |
| Derived fields for a transient solve | `TransientSolution`; the per-step series has no recovery |
| A-posteriori error estimator | a method on `Equation` (`BACKLOG.md`); per-equation by nature |
| Plane *stress* as an alternative 2D reduction | a second branch in `LinearElasticMaterial` |

None is blocked by the structure — each is an additional implementation of a seam that now exists,
which is the difference between this list and the vertical items above.

---

## The pattern, found repeatedly

`fem/regions.py` + `fem/boundary.py` is the original model: a **mesh-independent specification**
(`BoundaryConditions`, a list of `(type, region, value)`) cleanly separated from its **resolution
against one discretization** (`ResolvedBC`, frozen, keyed by mesh *and* component count). It
detects conflicts rather than letting last-write-win, refuses what it cannot honour, and its
docstring explains *why* the split exists.

The same shape — a derived, immutable object keyed by the discretization, replacing mutable state
that would otherwise drift — recurs four more times, which is the argument that it is the right
shape here rather than a stylistic preference:

- `FunctionSpace` is `ResolvedBC` for the discretization.
- `Form` is the resolved, assembly-ready view of an `Equation`'s physics, and `Equation.operator`
  is the derivation itself: `LinearElastic` hands back a `LinearElasticForm` built from its own
  constants, holding no mutable state.
- `Problem` is the resolved view of a whole composition, and `Solution` is the typed, immutable
  result the strategy hands back.
- `LinearSolver` is a `Backend` resolved against one assembled matrix: the immutable config,
  separated from the factorization or preconditioner that one operator produced.

You found the right shape once, and the composition model is that shape applied all the way up the
stack.

---

## Open items — dead weight and small refactors

A structural read of what is left, deliberately skipping the sparse-matrix and performance items
already covered in `BACKLOG.md` §2. Claims about what is dead or uncalled were verified by grepping
for callers and definitions, not inferred. These are priorities, not a defect list.

Legend: 🟡 design / maintainability · 🟢 small

### Dead paths and unused code

- 🟢 **`fem/numerics.py` `class color`** — no callers, superseded by the move to `logging`.
- 🟢 **`fem/numerics.py` `timer`** — no callers.

(`fem/quadrature.py` was the standout here — rules shaped `(func, polygon_vertices)` with no
callers. It is now the real reference-element layer, `QuadratureRule` + Gauss rules per simplex,
wired into `ElementGeometry` and every form.)

### Structural items

**🟢 The element→vertex projection weighting.** `FunctionSpace.element_to_vertex` weights by
element measure, which is the defensible projection and the reason the space rather than the mesh
owns it. The stricter choice is the mass-matrix L2 projection (solve `M u = ∫ f φ`); it is more
accurate on a graded mesh and costs a solve. Worth revisiting only if a nodal-output consumer
needs the accuracy — plotting does not.

**🟢 The load vector, now with a `LinearForm` beside `Source`.** The load `L(v) = ∫ f·v` still
has the typed `Source` term assembled as `mass_matrix @ f` — the mass form as a load operator, the
*exact* integral of `f`'s P1 interpolant (`M_ij = ∫ φ_i φ_j`), with `Traction` the boundary
sibling — and that stays the cheap default for a source given at the nodes. `fem.forms.LinearForm`
is now its quadrature-sampled sibling (`FunctionSpace.assemble_load`), which captures an `f` that
varies *within* an element, the same machinery `DiffusionForm`'s `∫ κ(x) ∇u·∇v` and the P2 load
use. It arrived with the quadrature layer, exactly where this note said it belonged.

**🟢 `import fem` re-exports the plot layer.** `fem/__init__.py` re-exports `Plotter` and
`PlotMode` as public API — a deliberate core → plot edge, worth revisiting only if the package
should be importable without a plotting backend installed. (The other core → plot path,
`numerics` importing `matplotlib` at module scope, is closed: those imports are local to the
`check_gradient` / `check_hessian` dev tools.) `pyamg` raises the same question from the other
end: `fem.backends` imports it at module scope, so `import fem` always pulls it in. A
minimal-import goal would want both edges lazy.

### Suggested order

1. ~~**Quadrature, then `LinearForm`**~~ — done. The two coupled
   assumptions behind the P1 ceiling both fell: the quadrature-point axis is on `ElementGeometry.grad_phi`, and
   the DOF numbering runs through `FunctionSpace.element_nodes` / `node_coords` rather than the
   mesh vertices. What is left is the 3D P2 element and P2-aware plotting / adaptivity (`BACKLOG.md`).
2. **Clear the unused modules** — delete `numerics.py`'s `color` / `timer`.
3. **Fill in the post-processing coverage gaps** (§5) — each is an implementation of an existing
   seam (Poisson flux, transient derived fields).
4. **Clear the remaining core → plot re-export** — only if headless import becomes a goal.
