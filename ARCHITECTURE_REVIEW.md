# Architecture Review — `fem/`

A structural read of the package, deliberately skipping the sparse-matrix and performance
items already covered in `BACKLOG.md` §2. Everything in "Confirmed broken" was verified by
grepping for callers/definitions, not inferred.

Legend: 🔴 broken today · 🟡 design / maintainability · 🟢 small

---

## 1. Confirmed broken or dead paths

### 🔴 `Solution.get_values(mode=...)` calls methods that do not exist

`fem/solution.py:50` and `fem/solution.py:57` call:

```python
self._convert_vertex_values_to_element_values(values)
self._convert_element_values_to_vertex_values(values)
```

Neither is defined on `Solution`. The real implementations live on `Mesh`
(`fem/mesh/mesh.py:27,34`) without the leading underscore. Any call passing
`mode='element'` or `mode='vertex'` raises `AttributeError`.

It has never surfaced because nothing passes `mode`. Every caller in the repo — tests,
`examples/`, the README snippet — uses the bare `get_values("u")` form.

**Fix:** either delegate to `self.femesh.convert_*` and add a test, or delete the `mode`
parameter and the `ValueMode` alias. Half the body of `get_values` is currently unreachable.

### 🔴 `TopologyOptimizer`'s `target_compliance` objective cannot run

`fem/topology.py:123` and `:127`:

```python
def target_compliance_objective(self, args):
    target = args[0]
    return (self.compliance() - target)**2      # TypeError
```

`compliance` and `compliance_gradient` (`:115`, `:118`) both take a *required* `args`
parameter, so both of these calls raise `TypeError` immediately. Selecting
`objective_name='target_compliance'` is a guaranteed crash.

Two related smells in the same method:

- `objective_func` is unpacked at `:91` and never used. Only `gradient_func` is —
  the objective value is never evaluated during optimization.
- `optimization_args` is a parameter of `solve` (`:87`), threaded into
  `_select_optimization` (`:137`), and then ignored entirely.

That dead parameter is exactly what AGENTS.md means by "typing the data is what makes dead
parameters visible" — the `Sequence[Any] | None` bag is what conceals it. Objectives would
be better as small classes or closures carrying their own configuration, rather than a
string plus a positional-args tuple resolved through `_select_*`.

### 🔴 Unused modules and members

- `fem/quadrature.py` — no importers anywhere in the repo. `BACKLOG.md` already flags
  "decide its fate"; worth deciding, since it is also referenced in the README's project
  structure as if it were live.
- `fem/numerics.py:91` `class color` — no callers. Superseded by the move to `logging`.
- `fem/numerics.py:79` `timer` — no callers.
- `check_gradient` / `check_hessian` (`fem/numerics.py:27,53`) both end in `plt.show()`,
  so they block. They are dev tools that cannot be called from a test, and they pull
  `matplotlib` into a top-level import of a core module.

---

## 2. Structural issues worth a refactor

### 🟡 `FEMesh` conflates geometry with assembled operators, and holds mutable solver state

`FEMesh.__init__` (`fem/mesh/femesh.py:49`) eagerly calls `prepare_matrices()`, assembling
four dense N×N matrices at `dim=1` before anything has asked for them. `Solver.assemble_everything`
(`fem/solver.py:169`) then calls `prepare_matrices(dim=2)` and discards all of that work.

Worse, `prepare_matrices` mutates `self.dim` (`femesh.py:52`). A mesh's state therefore
depends on whichever solver touched it last. Two solvers sharing one `FEMesh` at different
`dim` silently corrupt each other's operators.

**Concrete trap falling out of this:** `Solution.get_deformed_mesh` (`fem/solution.py:67`)
does `self.femesh.copy()` — a full re-assembly, purely to plot — and then mutates
`femesh_deformed.vertices += u.reshape(-1, self.dim)`. The copy's `element_objs`, with
their cached `volume` and `grad_phi`, were built from the *undeformed* vertices in
`__init__` and are now stale with respect to the vertices they claim to describe. Harmless
for plotting today; wrong the moment anything computes on a deformed mesh.

**Suggested shape:** make `Mesh` immutable geometry, and move `M`, `M_b`, `K`, `K_b` onto a
`dim`-keyed, lazily-built `Operators` object. This also delivers the assembly caching
`BACKLOG.md` §2 asks for ("`assemble_everything` runs on every `solve()`") as a side effect,
and removes the `# TODO: don't call this every time` at `fem/solver.py:144`.

### 🟡 `Solver` is a god class with exact-type dispatch

`Solver.solve` (`fem/solver.py:147`) dispatches through a dict literal keyed by
`type(self.equation)`:

```python
equation_solvers = {Projection: self.solve_projection, Poisson: self.solve_poisson, ...}
solver_fn = equation_solvers.get(type(self.equation))
```

Problems:

- **Exact type, not `isinstance`.** Subclassing any `Equation` silently fails to dispatch
  and raises "No solver for equation type".
- **Open/closed violation.** Adding a PDE means editing `Solver`, not adding a class. This
  is the dict-of-params dispatch AGENTS.md rejects, relocated one level up.
- **Implicit temporal coupling.** `self.mu` / `self.lamb` are set only inside the
  `LinearElastic` branch of `assemble_everything` (`:168`) and read unconditionally in
  `solve_linear_elastic` (`:324`). The two methods must be called in order, and nothing
  encodes that.
- The class now carries five PDE solve routines, the linear and Newton solvers, assembly,
  and the adaptive-refinement loop.

Note that the obvious fix — putting `solve()` on `Equation` — is *wrong*, and the
`Equation` docstring says why: one equation may be served by several solvers (`Solver` vs
`EnergySolver` for `LinearElastic`). The fix that preserves that separation is a small
strategy registry (`Poisson → PoissonStrategy`, …), resolved by an MRO walk so subclasses
dispatch correctly, with each strategy owning its own assembly and its own result fields.

### 🟡 `EnergySolver` duplicates `Solver`, and handles Dirichlet worse

`fem/energy_solver.py` reimplements the Newton loop (`newton_solve`, `:140`) that already
exists as `Solver.solve_nonlinear_system` (`fem/solver.py:207`), and re-unpacks the resolved
BC by hand into `self.free` / `self.fixed` / `self.fixed_values` (`:50-52`).

The more substantive divergence is Dirichlet handling. `Solver.solve_linear_system`
eliminates fixed DOFs properly (`solver.py:202-204`). `energy_hessian` instead zeroes fixed
rows *and* columns:

```python
total_energy_hessian[self.fixed, :] = 0
total_energy_hessian[:, self.fixed] = 0
```

That zeroes the diagonal too, making the Hessian **structurally singular by construction** —
which is precisely why `newton_solve` needs its fallback:

```python
except np.linalg.LinAlgError:
    logger.warning("Singular hessian, adding regularization")
    newton_step = np.linalg.solve(hessian + 1e-8 * np.eye(...), -gradient)
```

The regularization is papering over a self-inflicted singularity. Routing through the same
elimination path removes both the special case and the fallback.

Neither solver implements a shared interface, and `TopologyOptimizer` hardcodes `Solver`
(`fem/topology.py:38`). A `SolverProtocol` — `(femesh, equation, bc) -> Solution` — would
make them substitutable and let the optimizer accept either.

### 🟡 Elements know physics they should not

`LinearElement.calculate_stiffness_matrix` (`fem/elements.py:49`) branches on DOF count to
decide which PDE it is discretizing:

```python
if dim == 1:
    return self.grad_phi @ self.grad_phi.T * self.volume
# otherwise, the equation is linear elastic
idx = kwargs['idx']
B, D = self.calculate_B(), self.calculate_D(kwargs['mu'][idx], kwargs['lamb'][idx])
```

An element type is inferring the equation from the number of DOFs per node. It also receives
material parameters as untyped `**kwargs` carrying the *global* `mu`/`lamb` arrays plus its
own index, rather than its own scalar values — the element reaches into a global array to
find itself.

`FEMesh.assemble_matrix` (`femesh.py:65`) compounds it with two `Literal` string parameters
and its own admission: `# TODO: term "element" is overloaded here, and its a bit hacky`.

**Suggested seam:** a `BilinearForm` that owns the physics and the material data, with
elements providing only geometry and shape functions. This is the same refactor the
quadrature / higher-order-element work in `BACKLOG.md` §3 requires anyway — doing it first
makes that work additive rather than another round of `dim`-branching.

### 🟡 `Solution` is a stringly-typed dict

Against the repo's own "prefer typed over stringly-typed" convention:

- Keys are undiscoverable: `set_values("u", …)`, `values['compliance']`,
  `get_values('u_values', iter_idx=-1)`.
- `combine_solutions` (`solution.py:75`) invents a `'_list'` key-suffix convention, which
  `TopologyOptimizer._get_deformed_mesh` (`topology.py:151`) then probes with
  `try/except (KeyError, IndexError)`.
- `get_values(None)` returning a zero array (`solution.py:38`) is a plotting convenience
  leaking into the data model.
- Steady results (`u`) and time series (`u_values`, `t_values`) share one container with
  nothing in the type distinguishing them, so `mode` conversion has to *infer* meaning from
  `len(values)` (`:47`, `:55`) — which silently picks the wrong branch whenever
  `n_elements == n_vertices`.
- `combine_solutions` hardcodes `dim=2` with its own `# TODO: bit weird`.

Per-solve-type dataclasses (`SteadySolution`, `TransientSolution`, `ElasticSolution`) would
make the fields discoverable and delete the length-guessing entirely.

### 🟡 Core still depends on the plot layer

`fem/mesh/mesh.py:7` imports `Plotter, PlotMode` at module scope, for the convenience method
`Mesh.plot()` (`:42`). Commit 2413319 decoupled core solver/mesh code from the plotter;
this is a leftover. Same in `fem/numerics.py:8` (`import matplotlib.pyplot as plt`).

Consequences: the dependency direction is core → plot rather than plot → core, and
`import fem` pulls in matplotlib. A free function `plot_mesh(mesh)` in `fem.plot` inverts it.

---

## 3. Smaller items

- 🟢 `Mesh.convert_element_values_to_vertex_values` (`mesh.py:34`) is last-writer-wins at
  shared vertices, while its inverse (`:27`) averages. The asymmetry is probably unintended —
  an area-weighted average is the usual choice.
- 🟢 `FEMesh.calculate_total_value` (`femesh.py:93`) has no `else`: a `u` whose length matches
  neither elements nor vertices returns `None` instead of raising. `calculate_mean_value`
  then fails on `None / float` one frame away from the actual mistake.
- 🟢 `FEMesh.get_edges_in_idxs` (`:131`) and `get_boundary_idxs_in_rect` (`:147`) are 2D-only
  (both unpack `x, y`) and duplicate what `fem.regions` now does properly and
  dimension-generally. They look like pre-`regions` leftovers — candidates for deletion.
- 🟢 `LinearElement.calculate_dF_dx` (`elements.py:62`) is a 4-deep Python loop building a
  Kronecker product, with its own `# TODO: figure out kronecker product`. Same pattern in
  `energies.py` `calculate_d2S_dF2` (6-deep). Both are per-element and run inside the Newton
  loop.

---

## Suggested order

1. **The three broken paths** (§1) — cheap, and #1/#2 are latent landmines that only look
   fine because nothing exercises them.
2. **`FEMesh` geometry/operator split** (§2) — unblocks the sparse migration in `BACKLOG.md`
   §2 and kills the stale-`element_objs` trap.
3. **Strategy registry for `Solver`** (§2) — then fold `EnergySolver` onto the shared
   elimination path and a common protocol.
4. **`BilinearForm` seam** (§2) — prerequisite for quadrature and higher-order elements.
5. **Typed `Solution`** (§2) — best done last, since it touches every call site.
