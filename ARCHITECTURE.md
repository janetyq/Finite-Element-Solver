# Architecture — current model, target model, and the gap

Companion to `ARCHITECTURE_REVIEW.md`. That document lists defects; this one is about the
object model: which concepts exist, which are missing, and which objects are doing more than
one job. Anchored on symbol names rather than line numbers, which drift with every refactor.

---

## The thesis in one paragraph

The package was missing **one object** — the discretization, or function space — and most of
its role conflations were downstream of that absence. `FunctionSpace` now exists, and the two
conflations that depended on it are gone with it: the mesh no longer owns assembly, and the
`dim` that meant both components-per-node and spatial dimension is now two named quantities.

**All three of the original conflations are now closed.** The last — `Element` owning the
constitutive law — is gone: `Form` + `Material` exist, `D` and the strain-displacement matrix
`B` moved off the element, and assembly runs through a typed `Form` rather than an untyped
material bag. `Element` is pure geometry.

Two smaller, independent problems remain, and are the subject of most of what follows:
`Equation` still carries time-step parameters (a Time-layer split), and the physics layer,
though no longer on the element, is not yet *complete* — the strain measure is named but not
a selectable axis, and the linear solver still reaches for `B` and `D` directly to recover
stresses. (Deriving `D` from the energy `W` was the third item here and has been deliberately
closed the other way — see §3.) The package contains a **worked example of the right pattern**
in three places now — see "The layer that is already right" at the end.

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

The test of a layering is substitution: you should be able to swap a layer without touching
its neighbours. Swap dense→sparse (6) without touching physics (3). Swap P1→P2 (2) without
touching boundary conditions (5). Right now most of these swaps require edits in three or
four files.

## 2. Where the current classes sit

`█` = owns the layer · `▒` = shares it cleanly with another owner · `◧` = holds a piece it
should not — the conflation

Three symbols rather than two because the second and third had been collapsed into one, and
they read as opposite things: `DiscreteSystem` and `Solver` splitting layer 6 is the design
working, `Equation` holding `dt` is the defect list.

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh` | █ | | | | | | | | |
| `Element` / `ElementGeometry` | | ▒ | | | | | | | |
| `FunctionSpace` | | ▒ | | █ | | | | | ▒ |
| `FieldShape` (`Scalar` / `Vector`) | | ▒ | | | | | | | |
| `Form` / `EnergyForm` | | | █ | ▒ | | | | | |
| `Material` / energy densities | | | █ | | | | | | |
| `Equation` | | | █ | | | | ◧ | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `DiscreteSystem` | | | | | ▒ | █ | | | |
| `Solver` | | | ◧ | ◧ | ▒ | ▒ | █ | ◧ | ▒ |
| `EnergySolver` | | | ◧ | | ▒ | ▒ | | | |
| `TopologyOptimizer` | | | ◧ | | | | | █ | ▒ |
| `Solution` | | | | | | | ▒ | | █ |
| `RedGreenRefiner` | █ | | | | | | | | |
| `Plotter` / `io` | | | | | | | | | █ |

Read the rows: both solvers have dropped out of *most* of physics and assembly — they build a
form and hand it to the space — and out of the algebra too: the constrained solve lives in
`DiscreteSystem`, so `Solver` shares that layer rather than owning it. What they have not
dropped is marked `◧` and is small but real. `Solver.assemble_everything` `isinstance`-
dispatches the equation to a form and constructs the `LinearElasticMaterial` itself; it builds
the load vector inline; and `solve_linear_elastic` rebuilds `B` and `D` from scratch to recover
strain, stress, and compliance — constitutive code in a driver. `EnergySolver._select_energy`
makes the same kind of choice, mapping `LinearElastic` to `StVenantKirchhoff`.

Read the columns: layer 5 (constraints) has exactly one owner, and so does layer 6 (algebra).
Layer 2 (space) is split three ways but cleanly — the element supplies the reference basis,
`FieldShape` the component count, `FunctionSpace` the numbering. Layer 4 (assembly) is likewise
a clean split, `FunctionSpace` owning the scatter and `Form` the integrand; the only thing out
of place in that column is the solver's inline load vector. Layer 3 (physics) is *placed*:
`Form`/`Material`/energy densities own it and `Equation` names it, with only the solver
leftovers above outstanding. Layer 7 (time) is still split between `Equation` (holds `dt`,
`iters`) and `Solver` (holds the scheme), and layer 8 still has one driver inside the thing it
drives.

That layers 2, 4, 5, and 6 came out with clean ownership is not a coincidence: each got
designed deliberately.

---

## 3. Role-by-role

### `Mesh` / `FunctionSpace` — clean

`Mesh` is geometry: vertices, elements, boundary, topology queries. `FunctionSpace` has a
mesh and owns the discretization — element geometry, DOF numbering, cached operators. Two
spaces can share one domain, which is the property that made the split necessary.
`assemble` takes a `Form` rather than an untyped material bag, so the space forwards nothing
it cannot interpret.

`Mesh.plot` is gone — it had no callers and was the last core → plot dependency, so `fem/mesh`
no longer imports `fem/plot` at all. Plotting is entirely the plot layer's to initiate.

### `Form` / `Material` — placed, not yet unified

The constitutive law is off the element, and **every assembly path now goes through a form**:

- **Bilinear forms** — `MassForm` (`∫u·v`), `LaplacianForm` and `LinearElasticForm` (the
  `Gᵀ C G · volume` stiffness family) — scatter through `FunctionSpace.assemble`, one loop that
  no longer knows what it is scattering.
- **The nonlinear energy path** is `EnergyForm`, the sibling that maps an element *and a state*
  to an energy, residual, and tangent; `EnergySolver` scatters it through
  `assemble_residual`/`assemble_tangent`. A quadratic energy has a constant tangent, so the
  bilinear `Form` is `EnergyForm`'s state-independent special case.
- **The load** `L(v) = ∫f·v` is the mass form applied to the nodal source (`M @ f`), which is
  the exact integral of `f`'s P1 interpolant — form-assembled, a load operator rather than a
  system matrix. A first-class `LinearForm` waits on quadrature, which is what lets `f` vary
  *within* an element (a time-varying `f(·, t)` needs only per-step re-evaluation, not
  quadrature).

`Material` owns `D`, and the strain-displacement matrix `B` sits in `fem/forms.py` next to the
form that contracts it against `D`. That split is what let `Element` drop to pure geometry.

The two constitutive representations are the same material, and this is now *pinned* rather
than merely asserted: `energies.py`'s `calculate_W_from_S` and the `½εᵀDε` implied by `Material`
are one energy `W(ε) = ½λ(tr ε)² + μ tr(εᵀε)`, and `test_hooke_matrix_is_the_second_derivative_of_the_small_strain_energy`
checks that `D = ∂²W/∂ε²` in 2D. `D` is *left* in its Lamé-parameter closed form rather than
derived from `W` on purpose: that closed form is correct and dimension-general, whereas
`energies.py` is fixed-rank-2, so deriving `D` from the energy density would forfeit the 3D
path for no gain. The duplication is a two-line closed form checked against its source, not a
drift risk. The other axis is **kinematics**: the two solver paths
differ only in the strain measure fed to that one `W` — `energies.py` uses Green–Lagrange
`S = ½(FᵀF − I)` (geometrically nonlinear — St-VK), the linear path the small-strain
`ε = ½(∇u + ∇uᵀ)`. Both measures are now named (`SmallStrain`,
`StVenantKirchhoff`), pinned in `tests/test_elasticity_models.py`, but selecting
between them is not yet an equation-level choice.

So the physics layer decomposes as **material** (the energy `W`) × **kinematics** (the strain
measure), and `Form` is where selecting a point in that product becomes declarative.

One piece of the layer has not moved: `Solver.solve_linear_elastic` recovers strain, stress,
and compliance by rebuilding `B` and `D` itself, duplicating exactly what `LinearElasticForm`
already assembles from. A `stresses(geometry, u_elements)` on the form would put the recovery
next to the assembly it mirrors and let `solver.py` drop its `strain_displacement` and
`LinearElasticMaterial` imports — which is what would make "the solvers own no physics" true
rather than nearly true.

### `Element` — stateless types, batched geometry

Element types are stateless: `LinearTetrahedralElement` describes a shape and holds no
per-element data, so there is one of them in a program rather than one per tet. The
per-element data lives in `ElementGeometry`, which holds it for the whole mesh at once — one
`(n_elements, N, spatial_dim)` array of `grad_phi`, one `(n_elements,)` array of measures —
and `Form.element_matrices` computes every element matrix in a single vectorized pass.

This was the last of the scaling work. Assembling a 3D elastic solve at n=17 went from
18.7s to 0.48s, and the 3D MMS test now asserts a real O(h²) rate instead of an approach to
one. `examples/benchmark_assembly.py` measures the split; the cost is back on the sparse
factorization, which is the natural next target (see the iterative-solver item in
`BACKLOG.md`).

`EnergyForm` is batched too: the energy densities (`fem/energies.py`) evaluate the full
derivative chain — W, dW/dF, dS/dF, d²S/dF², d²W/dS² — over all elements at once, and
the form contracts those tensors against `dF_dx` in vectorized einsum calls. The densities
are dimension-general (parameterized on `d = grad_u.shape[-1]`, not a fixed DIM = 2), so
`EnergySolver` now accepts 3D meshes.

### `Equation` — four roles in one object

`Heat(u_initial, dt, iters, source)` carries:

1. **PDE identity** — that this is the heat equation (the class itself)
2. **Material / physical data** — `c` on `Wave`, `E`/`nu` on `LinearElastic`
3. **Initial conditions** — `u_initial`, `dudt_initial`
4. **Time-discretization parameters** — `dt`, `iters`

Only (1) and arguably (2) belong. `dt` and `iters` describe *how you numerically integrate*,
not what the equation is: the heat equation is `∂u/∂t = ∇²u + f` whether you take 10 steps or
10000. The class docstring is explicit that `Equation` is "what to solve" and `Solver` is
"how" — `dt` is unambiguously "how", and it is on the wrong side of a line the file itself
draws.

This is exactly why the `TimeIntegrator` abstraction in the backlog is awkward to add: the
parameters it would own are currently constructor arguments of the equation, so introducing it
is a breaking change to the public API rather than an additive one.

There is also a **mutation problem**. `TopologyOptimizer.set_rho` does:

```python
self.equation.E = self.rho**self.penalty * self.base_E
```

It reaches into the solver's equation and rewrites its material parameter every iteration. The
equation is being used as a mutable parameter carrier while also being the immutable problem
specification. Separating a `Material` (or a per-element coefficient field owned by the form)
is what removes the mutation itself; handing the driver a fresh `Material` each iteration is
the shape of the fix.

Two pieces of machinery that had grown around the mutation are gone. It used to clone the whole
equation to remember one number, which forced a bespoke `Equation.copy` built on `__new__` to
dodge subclass constructor signatures; storing `base_E` directly retired both. And the SIMP
exponent, previously the literal `3` in `set_rho` *and* a second literal `3` in
`compliance_gradient` — a correctness coupling with nothing stating it — is now one `penalty`
argument feeding both.

### `Solver` — seven layers, and an inverted driver

Covered in the review: god class, exact-type dispatch, temporal coupling on `self.mu`. Three
additional structural notes.

**`adaptive_refinement` is a driver living inside the thing it drives.** It mutates
`self.mesh`, re-resolves BCs, replaces `self.solution`, and loops. Compare
`TopologyOptimizer`: the *same* kind of outer loop, but correctly implemented as a separate
class that *owns* a `Solver`. Two outer loops, two opposite structures. `adaptive_refinement`
should be an `AdaptiveRefinement` driver taking a solver factory — which also fixes the
awkwardness that it has to rebuild BC resolution by hand after each remesh.

**Time integration is inline and duplicated.** `solve_heat` hardcodes backward Euler;
`solve_wave` hardcodes Crank–Nicolson, hand-builds a `2N` block system with `block_array`, and
needs `_wave_block_constraints` to lift nodal Dirichlet indices into block-DOF space.

`DiscreteSystem` was predicted here as the fix and only half delivered on it. It landed, and
`solve_wave` does now hand it the block operator and reuse one factorization across steps — but
it composes nothing: the blocking and the constraint lifting are still written out in the
solver. A system that knew **how to compose and block itself** would make `solve_wave` a scheme
applied to a system rather than a solver that knows how to index around a `2N` matrix. That is
the missing half, and it is a prerequisite for `TimeIntegrator` being additive rather than
another rewrite of the same indexing.

The second Newton loop is gone: `Solver.solve_nonlinear_system` had no callers and has been
deleted, leaving `EnergySolver.newton_solve` as the only one.

### `Solution` — result, field container, and time series

Three roles in one dict-backed object: the output of a solve, a bag of named post-processing
fields, and — via `combine_solutions` and the `_list` suffix convention — an iteration series.
The typed-dataclass fix is in the review; the architectural point here is the **coupling to
`io.py`**. `save_solution` walks `solution.values` and writes each entry as an npz array. Any
move to typed solutions has to land together with an I/O change, so sequence them as one
effort rather than discovering it midway.

---

## 4. Flexibility in the wrong places

Your instinct that it "needs and lacks flexibility in some areas" is well-founded — and they
are *different* areas. There is a fair amount of machinery built for extension that never
happened, and rigidity exactly where extension is on the roadmap.

**Generality that is not paying rent:**

| Mechanism | Reality |
|---|---|
| `TopologyOptimizer._select_objective` / `_select_optimization` | a plugin system with one optimization method, an ignored args bag, and an objective value that is never evaluated |
| `Solution.get_values(name, iter_idx, mode)` | three-axis generality; the `mode` axis is now covered by `tests/test_regressions.py` but still has no caller outside them |
| `quadrature.py` | five rules, zero callers — and shaped wrong for the layer that would replace them: they take `(func, polygon_vertices)`, where a real quadrature layer needs reference-element points and weights |

`Solver.solve_nonlinear_system` was the fourth row and has been deleted.

Each of these is a *string-or-kwargs-parameterized* extension point. That is the shape
flexibility takes when it is added speculatively, and it is the shape `AGENTS.md` warns
against — dead parameters stay invisible precisely because nothing types them.

**Rigidity where the roadmap actually goes:**

| Wanted (from `BACKLOG.md`) | Blocked by |
|---|---|
| Quadratic / higher-order elements | DOFs assumed one-per-vertex (`dof_indices`, `Mesh._get_all_edges`, `n_dofs`); needs real quadrature |
| Time-integrator abstraction | `dt`/`iters` live on `Equation`; schemes inlined in `Solver`; `DiscreteSystem` does not compose or block itself |
| Robin BCs | needs a *boundary stiffness* form and somewhere in `assemble_everything` for it to contribute to the LHS — the space assembles only a boundary mass matrix today |
| Variable coefficients | assembly uses closed-form linear-simplex integrals, no quadrature hook |
| Time-varying loads / BCs | `evaluate_field` takes position only, no `t` |
| Nonlinear materials | the two representations are now pinned as one energy `W`, but there is still no common interface to select a material through |
| Selectable kinematics | `SmallStrain` and `StVenantKirchhoff` exist and are tested, but choosing between them is a test-only injection, not an equation-level choice |

Two rows have been struck since this table was written, both by the performance work: sparse
matrices (assembly now emits COO triplets into a `csr_array`) and batched assembly
(`ElementGeometry` holds the whole mesh at once). Robin BCs kept its row but changed its
blocker — "assembly has no concept of a form" stopped being true when `Form` landed.

Note the pattern: the unused flexibility is all *lateral* (more string options on existing
operations), while the needed flexibility is all *vertical* (new layers between existing
ones). Speculative generality tends to widen; real extension tends to deepen.

---

## 5. Target model

The first two layers exist; the rest is the gap.

```python
# 1. geometry — pure, immutable                                    [done]
mesh = Mesh(vertices, elements, boundary)

# 2. discretization — owns DOF numbering, element geometry, operators   [done]
V = FunctionSpace(mesh, element=P1Triangle, n_components=2)

# 3. physics — material separate from form, form owns the weak statement   [done]
material = LinearElasticMaterial(E=210e9, nu=0.3)   # or ElementField for SIMP
form     = LinearElasticForm(material)

# 4. problem — space + form + constraints + data
problem = Problem(V, form, bcs=bc, source=f)

# 5. assembly + algebra — substitutable independently
system = assemble(problem)                        # -> DiscreteSystem(A, b, constraints)
u      = DirectSolve().solve(system)              # or SparseSolve(), NewtonSolve()

# 7. time — a scheme applied to a system, not a field on the equation
history = BackwardEuler(dt=1e-3, steps=100).run(problem, u0=u_initial)

# 8. drivers — own solvers, uniformly
AdaptiveRefinement(problem_factory, estimator).run()
TopologyOptimizer(problem_factory, objective=MinCompliance()).run()
```

What each move buys, concretely:

- **`Form` + `Material`** — *done*: removed `**kwargs` from assembly and the
  `n_components == 1` physics branch from `Element`, which is now pure geometry. *Remaining*:
  make the strain measure a selectable axis; move stress recovery off `Solver` onto the form;
  give Robin conditions a boundary form to contribute to; retire `TopologyOptimizer`'s mutation
  of `equation.E` in favour of handing the driver a fresh `Material` each iteration.
- **`DiscreteSystem`** — *mostly done*: it is the one place that knows "matrix + rhs + which
  DOFs are fixed", and it was the single seam where dense became sparse. *Remaining*: it does
  not compose or block itself, so the wave system is still hand-built with `block_array` and
  hand-lifted by `_wave_block_constraints`.
- **`TimeIntegrator`** — deduplicates backward Euler / Crank–Nicolson; moves `dt`/`iters` off
  `Equation`; makes θ-method / generalized-α additive.
- **Uniform drivers** — `adaptive_refinement` stops being a method on the thing it drives.

`Equation` as typed data survives all of this — it just sheds `dt`, `iters`, and its
mutable material, ending up as the *identity* of the PDE plus its genuinely physical
constants. That is what the docstring already claims it is. It has already shed `copy`,
which existed only to service the mutation.

---

## 6. A migration order that keeps the MMS test green

The convergence tests in `tests/test_convergence.py` and
`tests/test_convergence_elasticity.py` are the safety net; each step below should leave them
passing without modification.

1. **Extract `Form` + `Material`, and make every assembly path a form.** *Done.* `Form` owns
   the bilinear integrand, `EnergyForm` the nonlinear one, `Material` owns `D`, `Element` is
   pure geometry, and the load is the mass form applied to the source. `EnergySolver` scatters
   through the space like `Solver` does. Three follow-ons remain, all smaller and independently
   landable:
   - **1a. Pin `D = ∂²W/∂ε²`.** *Done.* `Material` keeps `D` in its Lamé-parameter closed
     form — correct and dimension-general — and a test cross-checks it against the small-strain
     energy density in 2D. Deriving `D` from `W` was considered and rejected: it would trade a
     checked two-line closed form for a contraction of the energy's rank-4 Hessian.
   - **1b. Make kinematics selectable.** `SmallStrain` and
     `StVenantKirchhoff` are the two members today; `Form`/`EnergyForm` is where
     choosing between them becomes an equation-level choice rather than the test-only injection
     it is now.
   - **1c. Move stress recovery onto the form.** `solve_linear_elastic` rebuilds `B` and `D`
     to compute strain, stress, and compliance — the last constitutive code outside the
     physics layer, and the cheapest of the three.
2. **`DiscreteSystem` + dense→sparse.** *Done.* Both solvers eliminate constraints through
   `DiscreteSystem`, the time-steppers factor their constant LHS once, assembly emits sparse
   CSR, and the factorization is `splu`. The linear algebra is off the critical path; the
   per-element assembly loop that replaced it as the top cost has been batched too (see
   `Element`, above). The scaling work is done; the remaining limit is the direct sparse
   factorization. *Remaining:* self-composition, which step 4 needs (see `Solver`, above).
3. **Typed `Solution`,** together with the `io.py` rework they jointly require.
4. **Extract `TimeIntegrator`;** move `dt`/`iters` off `Heat`/`Wave`. Breaking API change —
   worth batching with (3).
5. **Uniform drivers:** `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation.

Steps 1 and 2 are done, batched assembly with them. Steps 3–5 are independent and can be
done any time.

The pattern from the completed work is worth keeping: a mechanical rename that makes two
concepts unspellable as one name is cheap and buys more than it looks like, and the checkpoint
for each step should be a test the old architecture *could not have passed* — the 3D
elasticity MMS test played that role for `FunctionSpace`.

---

## The layer that is already right

`fem/regions.py` + `fem/boundary.py` is the model. It cleanly separates a **mesh-independent
specification** (`BoundaryConditions`, a list of `(type, region, value)`) from a **resolution
against one particular discretization** (`ResolvedBC`, frozen, keyed by mesh *and*
component count). It
detects conflicts rather than letting last-write-win. It refuses what it cannot honour
(`check_remeshable`, `BCType.ROBIN`). Its module docstring explains *why* the split exists.
`FunctionSpace` is now the second instance of the same pattern: a derived, immutable object
keyed by the discretization, replacing mutable state that used to drift. That it was arrived
at independently, and ended up shaped like `ResolvedBC`, is the argument that the pattern is
the right one here rather than a stylistic preference.

`Form` is now the third instance of the same pattern:

- `Form` is to `Equation` what `ResolvedBC` is to `BoundaryConditions` — the resolved,
  assembly-ready view of a specification that stays declarative. `LinearElasticForm(material)`
  is derived from a `LinearElastic` equation, produces element matrices, and holds no mutable
  state of its own.

You found the right shape once and it has now been applied three times. What remains in the
physics layer is not a missing object: it is making the kinematics axis selectable and pulling
the last constitutive code (stress recovery) out of `Solver` and into the form that already
knows the same `B` and `D`.
