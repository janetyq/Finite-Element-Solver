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
though no longer on the element, is not yet *unified* — `D` is built from Lamé parameters
rather than derived from the energy `W` that `energies.py` already holds, and the strain
measure is not yet a selectable axis. The package contains a **worked example of the right
pattern** in three places now — see "The layer that is already right" at the end.

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

`█` = owns the layer · `▒` = partially owns it, usually the conflation

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh` | █ | | | | | | | | ▒ |
| `FunctionSpace` | | █ | | █ | | | | | ▒ |
| `Element` | ▒ | | | | | | | | |
| `Form` / `Material` | | | █ | | | | | | |
| `Equation` | | | █ | | | | ▒ | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `Solver` | | | ▒ | ▒ | ▒ | █ | █ | ▒ | ▒ |
| `EnergySolver` | | | ▒ | ▒ | ▒ | █ | | | |
| `TopologyOptimizer` | | | ▒ | | | | | █ | ▒ |
| `Solution` | | | | | | | ▒ | | █ |
| `RedGreenRefiner` | █ | | | | | | | | ▒ |

Read the rows: `Solver` still touches six of nine layers. Read the columns: layer 2 (space)
and layer 4 (assembly) each now have exactly one owner — `FunctionSpace` owns the loop, and
`Form` owns the integrand, so nothing forwards material data it cannot interpret. Layer 3
(physics) is spread across `Equation` (identity + parameters), `Form`/`Material`, and
`energies.py`, but it is now *placed* rather than conflated: no non-physics object holds it.
Its open work is unification, not extraction — see below. Layer 7 (time) is still split
between `Equation` (holds `dt`, `iters`) and `Solver` (holds the scheme).

Layers 2, 4, and 5 each have exactly one owner. None is a coincidence: each got designed
deliberately.

---

## 3. Role-by-role

### `Mesh` / `FunctionSpace` — clean

`Mesh` is geometry: vertices, elements, boundary, topology queries. `FunctionSpace` has a
mesh and owns the discretization — element geometry, DOF numbering, cached operators. Two
spaces can share one domain, which is the property that made the split necessary.
`assemble` takes a `Form` rather than an untyped material bag, so the space forwards nothing
it cannot interpret.

`Mesh` should still lose `plot`, which is the last core → plot dependency (review §2).

### `Form` / `Material` — placed, not yet unified

The constitutive law is off the element. `Form` owns every bilinear integrand the linear
solvers assemble — `MassForm` (`∫u·v`), `LaplacianForm` and `LinearElasticForm` (the
`Gᵀ C G · volume` stiffness family, `G` geometric and `C` material) — and `FunctionSpace.assemble`
is a single scatter loop that no longer knows what it is scattering. `Material` owns `D`, and
the strain-displacement matrix `B` sits in `fem/forms.py` next to the form that contracts it
against `D`. That split is what let `Element` drop to pure geometry. The only assembly left
outside a `Form` is `EnergySolver`'s nonlinear energy path, which is a different method, not a
bilinear form.

What is *not* yet done is unifying the two constitutive representations. There is **one**
energy density `W(ε) = ½λ(tr ε)² + μ tr(εᵀε)`, shared verbatim: `energies.py`'s
`calculate_W_from_S` and the `½εᵀDε` implied by `Material` are the same function. `D` is
`∂²W/∂ε²` — a *derivative* of the energy in the other file, currently transcribed by hand from
Lamé parameters rather than computed from `W`. Deriving it would delete the duplication, at
the cost of bridging Voigt and full-tensor notation (a follow-on, guarded by the golden
matrices in `tests/test_forms.py`). The other axis is **kinematics**: the two solver paths
differ only in the strain measure fed to that one `W` — `energies.py` uses Green–Lagrange
`S = ½(FᵀF − I)` (geometrically nonlinear — St-VK), the linear path the small-strain
`ε = ½(∇u + ∇uᵀ)`. Both measures are now named (`SmallStrainEnergyDensity`,
`StVenantKirchhoffEnergyDensity`), pinned in `tests/test_elasticity_models.py`, but selecting
between them is not yet an equation-level choice.

So the physics layer decomposes as **material** (the energy `W`) × **kinematics** (the strain
measure), and `Form` is where selecting a point in that product becomes declarative.

### `Element` — now pure geometry, but stateful and per-instance

`Element` holds `grad_phi`, `volume`, and `dF_dx` and nothing physical. One issue remains,
and it is performance, not layering. `FunctionSpace` builds one object
per element, each caching `vertices`, `volume`, `grad_phi`, and `dF_dx` — a rank-4 tensor
built by a 4-deep Python loop, needed only by `EnergySolver`, computed unconditionally in
`LinearElement.__init__`. On a 40×40 mesh that is 3200 objects and 3200 unnecessary tensors.
The list itself is a `cached_property` now, so nothing is paid until something asks; the
waste is per-element rather than per-mesh.

The scalable alternative is element *types* as stateless strategies plus batched geometry:
one `(n_elements, …)` array of `grad_phi`, one array of volumes, computed vectorized. This is
a genuine fork and it interacts with the sparse migration — assembly over 3200 Python objects
will dominate once the linear algebra stops dominating. Worth deciding before, not after.

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
self.solver.equation.E = self.rho**3 * self.orig_equation.E
```

It reaches into the solver's equation and rewrites its material parameter every iteration,
which forces `orig_equation = equation.copy()` and the bespoke `Equation.copy` built on
`__new__` to dodge subclass constructor signatures. The equation is being used as a mutable
parameter carrier while also being the immutable problem specification. Separating a
`Material` (or a per-element coefficient field owned by the form) removes the mutation, the
`copy`, and the `__new__` hack together. Note also that the SIMP exponent `3` is hardcoded
inside `set_rho` — a modelling parameter buried in an assignment.

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
`solve_wave` hardcodes Crank–Nicolson and hand-builds a block system with `np.block`, then
needs `_wave_block_constraints` to lift nodal Dirichlet indices into block-DOF space. That
lifting is a strong signal: a **`DiscreteSystem` (matrix + rhs + constraints + dof map)** that
knows how to compose and block itself would make `solve_wave` a scheme applied to a system,
rather than a solver that knows how to index around a `2N`-sized matrix.

**`solve_nonlinear_system` has no callers.** A general Newton solver sits unused on `Solver`
while `EnergySolver` implements its own, differently and worse (see the review's singular-
Hessian finding). One of these should exist.

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
| `Solution.get_values(name, iter_idx, mode)` | three-axis generality; the `mode` axis has zero callers and no test |
| `Solver.solve_nonlinear_system` | general Newton hook, zero callers |
| `quadrature.py` | five rules, zero callers |

Each of these is a *string-or-kwargs-parameterized* extension point. That is the shape
flexibility takes when it is added speculatively, and it is the shape `AGENTS.md` warns
against — dead parameters stay invisible precisely because nothing types them.

**Rigidity where the roadmap actually goes:**

| Wanted (from `BACKLOG.md`) | Blocked by |
|---|---|
| Quadratic / higher-order elements | DOFs assumed one-per-vertex (`dof_indices`, `Mesh._get_all_edges`, `n_dofs`); needs real quadrature |
| Time-integrator abstraction | `dt`/`iters` live on `Equation`; schemes inlined in `Solver` |
| Robin BCs | needs a form contributing to the LHS; assembly has no concept of a form |
| Variable coefficients | assembly uses closed-form linear-simplex integrals, no quadrature hook |
| Time-varying loads / BCs | `evaluate_field` takes position only, no `t` |
| Nonlinear materials | two unrelated constitutive representations, no common interface |
| Sparse matrices | dense `np.zeros` hardcoded in `FunctionSpace._assemble`, per-element Python objects |

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
  derive `D` from the energy `W` rather than from Lamé parameters (unifying the two
  constitutive representations); make the strain measure a selectable axis; give Robin
  conditions a boundary form to contribute to; retire `TopologyOptimizer`'s mutation of
  `equation.E` in favour of handing the driver a fresh `Material` each iteration.
- **`DiscreteSystem`** — one place that knows "matrix + rhs + which DOFs are fixed"; makes the
  wave block system a composition instead of hand-indexed `np.block`; the single seam where
  dense becomes sparse.
- **`TimeIntegrator`** — deduplicates backward Euler / Crank–Nicolson; moves `dt`/`iters` off
  `Equation`; makes θ-method / generalized-α additive.
- **Uniform drivers** — `adaptive_refinement` stops being a method on the thing it drives.

`Equation` as typed data survives all of this — it just sheds `dt`, `iters`, and its
mutable material, ending up as the *identity* of the PDE plus its genuinely physical
constants. That is what the docstring already claims it is.

---

## 6. A migration order that keeps the MMS test green

The convergence tests in `tests/test_convergence.py` and
`tests/test_convergence_elasticity.py` are the safety net; each step below should leave them
passing without modification.

1. **Extract `Form` + `Material`.** *Done.* `Form` owns the integrand, `Material` owns `D`,
   `Element` is pure geometry, and `FunctionSpace.assemble` takes a `Form` instead of an
   untyped material bag — mass, stiffness, and boundary mass all through it. Two follow-ons
   remain from the ideal end state, both smaller and independently landable:
   - **1a. Derive `D` from `W`.** `Material` holds `D` built from Lamé parameters; the energy
     `W` in `energies.py` already implies it as `∂²W/∂ε²`. Computing it from `W` deletes the
     duplication but needs a Voigt↔full-tensor bridge, so it is isolated and guarded by the
     golden matrices in `tests/test_forms.py`.
   - **1b. Make kinematics selectable.** `SmallStrainEnergyDensity` and
     `StVenantKirchhoffEnergyDensity` are the two members today; `Form` is where choosing
     between them becomes an equation-level choice rather than the test-only injection it is
     now.
2. **Introduce `DiscreteSystem`,** then migrate dense→sparse behind it. The backlog's
   highest-leverage change becomes a one-layer edit rather than a cross-cutting one. Now the
   next load-bearing step.
3. **Typed `Solution`,** together with the `io.py` rework they jointly require.
4. **Extract `TimeIntegrator`;** move `dt`/`iters` off `Heat`/`Wave`. Breaking API change —
   worth batching with (3).
5. **Uniform drivers:** `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation.

Step 1 is done; step 2 is now the load-bearing one. Steps 3–5 are independent and can be done
any time.

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
physics layer is unification within it (deriving `D` from `W`), not a missing object.
