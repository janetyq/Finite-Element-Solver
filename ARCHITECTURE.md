# Architecture — current model, target model, and the gap

Companion to `ARCHITECTURE_REVIEW.md`. That document lists defects; this one is about the
object model: which concepts exist, which are missing, and which objects are doing more than
one job. Anchored on symbol names rather than line numbers, which drift with every refactor.

---

## The thesis in one paragraph

There is **one missing object** — the discretization, or function space — and almost every
role conflation in the package is downstream of its absence. Because nothing owns "this mesh,
discretized with these elements, carrying this many components per node," that knowledge is
smeared across seven classes as a bare `int` called `dim`. `FEMesh` ends up owning assembly
because it is the only thing that knows the DOF map. `Element` ends up owning constitutive
law because it is the only thing that knows the element geometry. `Equation` ends up owning
time-step parameters because it is the only per-problem object the solver receives. Introduce
the missing object and those three conflations dissolve without anyone having to be clever.

The good news: the package already contains a **worked example of the right pattern**. See
"The layer that is already right" at the end.

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
| `FEMesh` | █ | ▒ | | █ | | | | | ▒ |
| `Element` | ▒ | ▒ | ▒ | ▒ | | | | | |
| `Equation` | | ▒ | █ | | | | ▒ | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `Solver` | | ▒ | ▒ | ▒ | ▒ | █ | █ | ▒ | ▒ |
| `EnergySolver` | | | ▒ | ▒ | ▒ | █ | | | |
| `TopologyOptimizer` | | | ▒ | | | | | █ | ▒ |
| `Solution` | | ▒ | | | | | ▒ | | █ |
| `RedGreenRefiner` | █ | | | | | | | | ▒ |

Read the rows: `Solver` touches seven of nine layers. Read the columns: layer 2 (space) has
no owner — five classes hold a fragment. Layer 3 (physics) is split across `Equation`,
`Element.calculate_D`, `energies.py`, and `materials.py`. Layer 7 (time) is split between
`Equation` (holds `dt`, `iters`) and `Solver` (holds the scheme).

Only layer 5 has exactly one owner and one home. That is not a coincidence — it is the layer
that got designed deliberately.

---

## 3. The `dim` problem

`dim` is the clearest symptom, so it is worth tracing in full. The same integer is stored on:

- `Equation.dim` — a `ClassVar` (`Poisson.dim = 1`, `LinearElastic.dim = 2`)
- `Solver.dim`, `EnergySolver.dim` — copied from the equation
- `FEMesh.dim` — **mutable**, set by whichever `prepare_matrices` call ran last
- `ResolvedBC.dim` — frozen, correctly keyed
- `Solution.dim` — copied at construction
- parameters: `assemble_matrix(dim)`, `calculate_mass_matrix(dim)`, `dof_indices(element, dim)`

Seven homes for one number, one of them mutable and shared. But the deeper problem is that
**`dim` means two different things** and the code conflates them:

- **components per node** — 1 for a scalar PDE, 2 or 3 for a displacement field
- **spatial dimension** — 2 for a triangle mesh, 3 for a tet mesh

For `Poisson`, `dim = 1` means "scalar field". For `LinearElastic`, `dim = 2` means "two
components per node" *and*, silently, "2D only". Those coincide for 2D elasticity and come
apart everywhere else. This is precisely why:

- `LinearElastic.dim` is a `ClassVar` and therefore **cannot express 3D elasticity**. The
  correct statement is `n_components == mesh.spatial_dim`, which a class constant cannot say.
  The "N-D elasticity" backlog item is blocked on this, not on the einsums.
- `LinearElement.calculate_stiffness_matrix` branches `if dim == 1: <Laplacian> else:
  <elasticity>` — it is using components-per-node as a proxy for "which PDE is this", which
  works only because the two currently correlate.
- `FEMesh.prepare_matrices(dim=...)` has to be *re-run* to change components, rebuilding
  geometry-only quantities that never depended on `dim` at all.

### The fix: store a *kind*, derive the *count*

The instinct is to move the number somewhere better. That is still wrong — the number should
not be written down at all. It is a three-way split in which nobody stores the count:

| Concept | Owner | Form it takes |
|---|---|---|
| What kind of value the unknown takes | `Equation` | a **rank** — `SCALAR` or `VECTOR`, not a number |
| What the domain is | `Mesh` | `spatial_dim`, from `vertices.shape[1]` |
| How many DOFs per node | *derived* | `rank.components_for(spatial_dim)` |

The mathematical fact being encoded: elasticity's unknown is a *vector field on the domain*,
so its component count is determined by the domain rather than chosen. A class constant cannot
say that; a rank can.

```python
class FieldRank(Enum):
    '''What kind of value the unknown field takes at each point.

    Rank 0 is a scalar (temperature, potential), rank 1 a vector (displacement).
    The component count follows from the rank and the domain, so storing the rank
    rather than the count is what lets one Equation class describe both 2D and 3D
    elasticity. A rank-2 (tensor) member belongs here if a mixed formulation ever
    needs one; it is omitted until something does.
    '''
    SCALAR = 'scalar'
    VECTOR = 'vector'

    def components_for(self, spatial_dim: int) -> int:
        return 1 if self is FieldRank.SCALAR else spatial_dim


class Equation:
    rank: ClassVar[FieldRank] = FieldRank.SCALAR   # Projection, Poisson, Heat, Wave inherit

class LinearElastic(Equation):
    rank: ClassVar[FieldRank] = FieldRank.VECTOR   # was: dim = 2
```

```python
class Mesh:
    @property
    def spatial_dim(self) -> int:
        '''Dimension of the space the nodes live in. Distinct from the element's
        reference dimension (`N - 1`): a triangle mesh embedded in 3D has
        spatial_dim 3 but reference dimension 2.'''
        return self.vertices.shape[1]
```

**Where the derivation lives.** Before `FunctionSpace` exists, in `Solver.__init__` /
`EnergySolver.__init__` — a one-line replacement for `self.dim = self.equation.dim`. After, in
`FunctionSpace` construction. The change is coherent at both stages, which is why it can land
first and independently:

```python
V = FunctionSpace(mesh, element=P1Triangle, n_components=2)
V.n_dofs            # len(mesh.vertices) * n_components
V.dof_indices(e)    # element -> global DOFs
V.spatial_dim       # from the mesh, distinct from n_components
```

`FunctionSpace` keeps taking `n_components` as an explicit low-level argument rather than an
`Equation`; the rank-based derivation happens one layer up. That keeps the space usable for
mixed formulations later without the equation taxonomy constraining it.

**Most of the diff is a rename.** Every downstream `dim` parameter means components-per-node,
so it should say so: `dof_indices(element, n_components)`, `evaluate_field(value, points,
n_components)`, `ResolvedBC.n_components`, `Solution.n_components`. Once renamed the two
meanings are no longer spellable the same way, which is the actual defect. `fem/typing.py`
needs the same pass — its shape comments use `dim` for both (`(n_vertices, dim)` is spatial;
`(n_vertices * dim,)` is components).

### Evidence the framing is right

`EnergySolver`'s 2D guard is currently **dead code**:

```python
self.dim = self.equation.dim      # LinearElastic.dim is a ClassVar == 2, always
if self.dim != 2:                 # can never fire
    raise NotImplementedError(f'EnergySolver only supports 2D for now (got dim={self.dim})')
```

It also asserts `isinstance(equation, LinearElastic)`, so `self.dim` is unconditionally 2.
Hand it a tet mesh today and it sails past this guard into `LinearElasticEnergyDensity.
set_grad_u`, which rejects the `(3, 2)` gradient — still loud, but several frames from the
cause. Written against the right quantity the guard becomes live and correct:

```python
if femesh.spatial_dim != 2:   # the energy densities are built at fixed rank 2
```

The guard's author meant *spatial* dimension and only had `dim` available to say it with.

### Limits of this change — three things it does not settle

**1. It removes the blocker on 3D elasticity; it does not demonstrate 3D elasticity.**
`LinearTetrahedralElement` already has both `calculate_B` and `calculate_D`, so the element
side appears ready and the path may simply work once the component count is right. That is
untested. Treat "3D elasticity works" as a claim requiring an MMS test, not a consequence.

**2. Rank → count is wrong for two real cases**, neither currently in the repo:

- *Surface meshes.* Triangles embedded in 3D would correctly derive 3 components — but
  `LinearTriangleElement.calculate_B` builds a 3×6 matrix hardcoding 2. The derivation would
  be right and the element wrong, so this needs the element work regardless.
- *Systems.* A k-species reaction–diffusion problem has k components unrelated to spatial
  dimension. `SCALAR`/`VECTOR` cannot express it; it needs an explicit-count variant. Adding
  one now would be speculative, but it is the known edge of the taxonomy.

**3. `rank` may not belong on `Equation` permanently.** Once physics moves into `Form`, an
`ElasticityForm` arguably implies vector-valued already, making the rank redundant. Keeping it
on `Equation` is right for the user-facing declaration and right for the interim; it is the
piece of this proposal most likely to move later.

Note also what this does **not** touch: `LinearElement.calculate_stiffness_matrix`'s
`if dim == 1: <Laplacian> else: <elasticity>` branch. Fixing the count makes that branch's
fragility sharper rather than better — it uses component count as a proxy for *which PDE this
is*, and those genuinely diverge (1D elasticity would be `n_components == 1` and silently get
the Laplacian). That needs the `Form` seam, and is a separate step.

---

## 4. Role-by-role

### `Mesh` / `FEMesh` — geometry, discretization, assembly, and wave physics

`FEMesh` inherits geometry and adds: element objects, the DOF map, four assembled operators,
and a set of `calculate_*` metrics. Two problems beyond the mutability already covered in the
review:

**Assembly does not belong on the mesh.** A mesh is a domain; assembly is a function of
(space, form, quadrature). Putting `assemble_matrix` on `FEMesh` is why the material
parameters have to arrive as `**kwargs` and why the method needs two `Literal` string
selectors — the mesh is being asked to do something it lacks the information for, so the
information gets smuggled in as untyped keywords.

**`calculate_energy` is wave-equation physics living on the mesh.** It takes `c` — a wave
speed — and computes `½(c²·uᵀKu + u̇ᵀMu̇)`. That is a post-processing quantity for one specific
PDE. `calculate_dirichlet_energy` is similar. These are layer-9 functions parked in layer 1
because the mesh happened to be the object with `K` and `M` on it. Once operators move to
`FunctionSpace`, these become free functions in a `postprocess` module taking `(V, u, …)`.

`Mesh` itself is close to right — geometry, topology, `with_topology`, edges. The `MeshT`
TypeVar and `with_topology` are genuinely good: they are what makes remeshing type-preserving.
Keep that. Drop `plot`, drop `save`/`load` delegation if you want it pure, keep the rest.

### `Element` — geometry, shape functions, *and* constitutive law

`calculate_D(mu, lamb)` returns the Hooke tensor. That is a **material law**, not element
geometry, and it lives on the element only because the element was in scope. Meanwhile:

- `fem/materials.py` holds `Enu_to_Lame` and nothing else (16 lines)
- `fem/energies.py` holds `LinearElasticEnergyDensity` — an *independent second*
  representation of the same material, used only by `EnergySolver`

So linear elasticity is currently modelled twice, in two files, with no shared abstraction —
and **the two do not agree**: `energies.py` uses the Green–Lagrange strain
`S = ½(FᵀF − I)`, while `elements.py` uses the small-strain `B`/`D` form. `EnergySolver`'s
header comment acknowledges the discrepancy. That is a legitimate modelling choice (one is
geometrically nonlinear), but it should be *expressed* as one, by two named material models
under a common interface, not by two unrelated classes in different layers.

Second issue: **elements are stateful, per-instance, and eager**. `FEMesh.__init__` builds one
object per element, each caching `vertices`, `volume`, `grad_phi`, and `dF_dx` — a rank-4
tensor built by a 4-deep Python loop, needed only by `EnergySolver`, computed always. On a
40×40 mesh that is 3200 objects and 3200 unnecessary tensors before anyone has asked for
anything.

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
`self.femesh`, re-resolves BCs, replaces `self.solution`, and loops. Compare
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

## 5. Flexibility in the wrong places

Your instinct that it "needs and lacks flexibility in some areas" is well-founded — and they
are *different* areas. There is a fair amount of machinery built for extension that never
happened, and rigidity exactly where extension is on the roadmap.

**Generality that is not paying rent:**

| Mechanism | Reality |
|---|---|
| `assemble_matrix(matrix_type_name, element_type_name, **kwargs)` | 2×2 string-selected combos, leaks untyped material data |
| `TopologyOptimizer._select_objective` / `_select_optimization` | a plugin system with one working objective, one method, an ignored args bag, and a broken second objective |
| `Solution.get_values(name, iter_idx, mode)` | three-axis generality; the `mode` axis has zero callers and no test |
| `Solver.solve_nonlinear_system` | general Newton hook, zero callers |
| `quadrature.py` | five rules, zero callers |

Each of these is a *string-or-kwargs-parameterized* extension point. That is the shape
flexibility takes when it is added speculatively, and it is the shape `AGENTS.md` warns
against — dead parameters stay invisible precisely because nothing types them.

**Rigidity where the roadmap actually goes:**

| Wanted (from `BACKLOG.md`) | Blocked by |
|---|---|
| Quadratic / higher-order elements | no `FunctionSpace`; DOFs assumed one-per-vertex throughout (`dof_indices`, `Mesh._get_all_edges`, every `len(vertices)` sizing) |
| 3D elasticity | `dim` as a `ClassVar` conflating components with spatial dimension |
| Time-integrator abstraction | `dt`/`iters` live on `Equation`; schemes inlined in `Solver` |
| Robin BCs | needs a form contributing to the LHS; assembly has no concept of a form |
| Variable coefficients | assembly uses closed-form linear-simplex integrals, no quadrature hook |
| Time-varying loads / BCs | `evaluate_field` takes position only, no `t` |
| Nonlinear materials | two unrelated constitutive representations, no common interface |
| Sparse matrices | assembly on the mesh, dense `np.zeros` hardcoded, per-element Python objects |

Note the pattern: the unused flexibility is all *lateral* (more string options on existing
operations), while the needed flexibility is all *vertical* (new layers between existing
ones). Speculative generality tends to widen; real extension tends to deepen.

---

## 6. Target model

```python
# 1. geometry — pure, immutable
mesh = Mesh(vertices, elements, boundary)

# 2. discretization — owns DOF numbering, element geometry, cached operators
V = FunctionSpace(mesh, element=P1Triangle, n_components=2)

# 3. physics — material separate from form, form owns the weak statement
material = LinearElastic(E=210e9, nu=0.3)        # or ElementField for SIMP
form     = ElasticityForm(material)

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

- **`FunctionSpace`** — deletes `dim` from six classes; makes P2 elements additive rather than
  invasive; gives operator caching a natural home (the backlog's "assemble every solve" item);
  makes `FEMesh` unnecessary, collapsing the `Mesh`/`FEMesh` split.
- **`Form` + `Material`** — removes `**kwargs` from assembly; removes the `dim == 1` physics
  branch from `Element`; gives Robin conditions somewhere to contribute; unifies the
  `D`-matrix and energy-density representations; removes `TopologyOptimizer`'s mutation of
  `equation.E`.
- **`DiscreteSystem`** — one place that knows "matrix + rhs + which DOFs are fixed"; makes the
  wave block system a composition instead of hand-indexed `np.block`; the single seam where
  dense becomes sparse.
- **`TimeIntegrator`** — deduplicates backward Euler / Crank–Nicolson; moves `dt`/`iters` off
  `Equation`; makes θ-method / generalized-α additive.
- **Uniform drivers** — `adaptive_refinement` stops being a method on the thing it drives.

`Equation` as typed data survives all of this — it just sheds `dim`, `dt`, `iters`, and its
mutable material, ending up as the *identity* of the PDE plus its genuinely physical
constants. That is what the docstring already claims it is.

---

## 7. A migration order that keeps the MMS test green

The convergence test in `tests/test_convergence.py` is the safety net; each step below should
leave it passing without modification.

1. **Fix the broken paths** (`ARCHITECTURE_REVIEW.md` §1). Clears the deck; no design commitment.
2. **`FieldRank` + the `dim` → `n_components` rename** (§3). Three read sites plus a mechanical
   rename; no new layers, no public API change (`Solver(mesh, Poisson(...), bc)` is unaffected).
   Lands before `FunctionSpace` because it is smaller and independent, and it makes the
   subsequent extraction obvious — once the count is derived, the thing that derives it wants
   to be an object.
3. **Extract `FunctionSpace` from `FEMesh`.** Move `dof_indices`, element objects, and the
   cached operators. Keep `FEMesh` as a thin deprecated shim initially so call sites migrate
   incrementally. This is the keystone — do it before anything else structural.
4. **Extract `Form` + `Material`.** Move `calculate_D` off `Element` and unify with
   `energies.py` under one constitutive interface. Assembly becomes `assemble(space, form)`,
   and the `**kwargs` and `Literal` selectors disappear. This is also what retires the
   `if dim == 1` physics branch that (2) deliberately leaves alone.
5. **Introduce `DiscreteSystem`,** then migrate dense→sparse behind it. The backlog's
   highest-leverage change becomes a one-layer edit rather than a cross-cutting one.
6. **Extract `TimeIntegrator`;** move `dt`/`iters` off `Heat`/`Wave`. Breaking API change —
   worth batching with (7).
7. **Typed `Solution`,** together with the `io.py` rework they jointly require.
8. **Uniform drivers:** `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation.

Steps 3–5 are the load-bearing ones. Step 2 is the cheapest real improvement and a good first
commit. Steps 1, 7, 8 are independent and can be done any time.

A 3D-elasticity MMS test is the natural checkpoint after (2) — it is the first thing the
current architecture cannot express, so it doubles as proof the change did what it claims.

---

## The layer that is already right

`fem/regions.py` + `fem/boundary.py` is the model. It cleanly separates a **mesh-independent
specification** (`BoundaryConditions`, a list of `(type, region, value)`) from a **resolution
against one particular discretization** (`ResolvedBC`, frozen, keyed by mesh *and* `dim`). It
detects conflicts rather than letting last-write-win. It refuses what it cannot honour
(`check_remeshable`, `BCType.ROBIN`). Its module docstring explains *why* the split exists.
And it is the one layer with exactly one owner in the table in §2.

Every recommendation above is the same pattern applied elsewhere:

- `FunctionSpace` is to `Mesh` what `ResolvedBC` is to `BoundaryConditions` — a derived,
  immutable object keyed by the discretization, replacing mutable state that currently drifts
  (`FEMesh.dim`).
- `Form` is to `Equation` what `ResolvedBC` is to `BoundaryConditions` — the resolved,
  assembly-ready view of a specification that stays declarative.

You already found the right shape once. The rest of the package has not caught up to it yet.
