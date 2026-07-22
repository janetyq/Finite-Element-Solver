# Architecture Review — `fem/`

A structural read of the package, deliberately skipping the sparse-matrix and performance
items already covered in `BACKLOG.md` §2. Claims about what is dead or uncalled were
verified by grepping for callers and definitions, not inferred.

Nothing here is a known-broken path any more; what remains is design and dead weight.

Line references are against `cfaa0f6`. They drift; the symbol names are the durable anchor,
and `ARCHITECTURE.md` deliberately uses those alone.

Legend: 🟡 design / maintainability · 🟢 small

---

## 1. Dead paths and unused code

### 🟡 `TopologyOptimizer`'s objective plumbing carries dead parameters

`solve` (`fem/topology.py:88`) resolves two callables and then uses one of them:

- `objective_func` is unpacked at `:97` and never called. Only `gradient_func` is —
  the objective value is never evaluated during optimization, so selecting an
  objective picks a gradient and nothing else.
- `optimization_args` is a parameter of `solve` (`:93`), threaded into
  `_select_optimization` (`:144`), and then ignored entirely.

That dead parameter is exactly what AGENTS.md means by "typing the data is what makes dead
parameters visible" — the `Sequence[Any] | None` bag is what conceals it. Objectives would
be better as small classes or closures carrying their own configuration, rather than a
string plus a positional-args tuple resolved through `_select_*`.

### 🟡 Unused modules and members

- `fem/quadrature.py` — no importers anywhere in the repo. `BACKLOG.md` already flags
  "decide its fate"; worth deciding, since it is also referenced in the README's project
  structure as if it were live.
- `fem/numerics.py:91` `class color` — no callers. Superseded by the move to `logging`.
- `fem/numerics.py:79` `timer` — no callers.
- `check_gradient` / `check_hessian` (`fem/numerics.py:27,53`) both end in `plt.show()`,
  so they block. They are dev tools that cannot be called from a test, and they pull
  `matplotlib` into a top-level import of a core module. Their only live caller is
  `StVenantKirchhoffEnergyDensity.check_gradients` (`fem/energies.py:102`), itself uncalled;
  the `EnergySolver` uses are the parked commented-out blocks at `energy_solver.py:62-66`.

---

## 2. Structural issues worth a refactor

### 🟡 `Solver` is a god class with exact-type dispatch

`Solver.solve` (`fem/solver.py:161`) dispatches through a dict literal keyed by
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
  `LinearElastic` branch of `assemble_everything` (`:191`) and read unconditionally in
  `solve_linear_elastic` (`:361`). The two methods must be called in order, and nothing
  encodes that.
- The class now carries five PDE solve routines, the linear and Newton solvers, assembly,
  and the adaptive-refinement loop.

Note that the obvious fix — putting `solve()` on `Equation` — is *wrong*, and the
`Equation` docstring says why: one equation may be served by several solvers (`Solver` vs
`EnergySolver` for `LinearElastic`). The fix that preserves that separation is a small
strategy registry (`Poisson → PoissonStrategy`, …), resolved by an MRO walk so subclasses
dispatch correctly, with each strategy owning its own assembly and its own result fields.

### 🟡 `EnergySolver` duplicates `Solver`, and handles Dirichlet worse

`fem/energy_solver.py` reimplements the Newton loop (`newton_solve`, `:146`) that already
exists as `Solver.solve_nonlinear_system` (`fem/solver.py:241`), and re-unpacks the resolved
BC by hand into `self.free` / `self.fixed` / `self.fixed_values` (`:56-58`).

The more substantive divergence is Dirichlet handling. `Solver.solve_linear_system`
eliminates fixed DOFs properly (`solver.py:214`). `energy_hessian` (`:126`) instead zeroes
fixed rows *and* columns:

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
(`fem/topology.py:40`). A `SolverProtocol` — `(mesh, equation, bc) -> Solution` — would
make them substitutable and let the optimizer accept either.

### 🟡 The load vector waits on quadrature, not on a `LinearForm`

Assembly is now uniformly form-based: bilinear forms (mass, stiffness, boundary mass) scatter
through `FunctionSpace.assemble`, the nonlinear energy path is `EnergyForm`, and the load
`L(v) = ∫ f·v` is the mass form applied to the nodal source:

```python
self.b = (self.M @ source_load.flatten()).flatten()
```

This is the *exact* integral of `f`'s P1 interpolant (`M_ij = ∫ φ_i φ_j`), so the load is
already form-assembled — the mass form used as a load operator rather than a system matrix. A
standalone `LinearForm` adds capability only when `f` varies *within* an element, which needs
quadrature to sample it at interior points — the same machinery non-constant coefficients
(`∫ κ(x) ∇u·∇v`) and P2 elements need, and the reason `quadrature.py` has no callers yet. So a
`LinearForm` belongs with the quadrature work, not before it. Two things it is *not* blocked
on: a time-varying source `f(·, t)` just needs re-evaluating `M @ f_t` per step (and fixing the
Crank–Nicolson `b_n`/`b_{n+1}` average `solve_wave` flags), and Robin conditions need a
*bilinear* boundary form, which `assemble(form, boundary=True)` already supports.

### 🟡 `Solution` is a stringly-typed dict

Against the repo's own "prefer typed over stringly-typed" convention:

- Keys are undiscoverable: `set_values("u", …)`, `values['compliance']`,
  `get_values('u_values', iter_idx=-1)`.
- `combine_solutions` (`solution.py:77`) invents a `'_list'` key-suffix convention, which
  `TopologyOptimizer._get_deformed_mesh` (`topology.py:159`) then probes with
  `try/except (KeyError, IndexError)`.
- `get_values(None)` returning a zero array (`solution.py:39`) is a plotting convenience
  leaking into the data model.
- Steady results (`u`) and time series (`u_values`, `t_values`) share one container with
  nothing in the type distinguishing them, so `mode` conversion has to *infer* meaning from
  `len(values)` (`:47`, `:54`) — which silently picks the wrong branch whenever
  `n_elements == n_vertices`.
- The `mode` axis of `get_values` has no callers anywhere in the repo — tests, `examples/`,
  and the README snippet all use the bare `get_values("u")` form — so the length-guessing
  above is unexercised as well as unsound.
- `combine_solutions` hardcodes a component count of 2 with its own `# TODO: bit weird`.

Per-solve-type dataclasses (`SteadySolution`, `TransientSolution`, `ElasticSolution`) would
make the fields discoverable and delete the length-guessing entirely.

### 🟡 Core still depends on the plot layer

`fem/mesh/mesh.py:7` imports `Plotter, PlotMode` at module scope, for the convenience method
`Mesh.plot()` (`:52`). Commit 2413319 decoupled core solver/mesh code from the plotter;
this is a leftover. Same in `fem/numerics.py:8` (`import matplotlib.pyplot as plt`).

`fem/mesh/refinement.py` shows the shape of the fix: it exposes `leaf_classifications()` as
plain data and lets `fem.plot.helpers.plot_refinement` render it, so the module imports no
plot code at all. Dropping `Mesh.plot` for a free function `plot_mesh(mesh)` does the same
here.

Consequences: the dependency direction is core → plot rather than plot → core, and
`import fem` pulls in matplotlib. A free function `plot_mesh(mesh)` in `fem.plot` inverts it.

---

## 3. Smaller items

- 🟢 `Mesh.convert_element_values_to_vertex_values` (`mesh.py:37`) is last-writer-wins at
  shared vertices, while its inverse (`:30`) averages. The asymmetry is probably unintended —
  an area-weighted average is the usual choice.
- 🟢 `LinearElement.calculate_dF_dx` (`elements.py:94`) is a 4-deep Python loop building a
  Kronecker product, with its own `# TODO: figure out kronecker product`. Same pattern in
  `energies.py` `calculate_d2S_dF2` (6-deep). Both are per-element and run inside the Newton
  loop.

---

## Suggested order

1. **Typed `Solution`** (§2) — independent of everything below, and the largest felt
   improvement for callers. Touches every call site, so it wants to land on its own.
2. **`DiscreteSystem`, then dense → sparse** — the single seam where the algebra layer
   changes, and the item `BACKLOG.md` §2 calls the highest-leverage one. The load-bearing
   step now that `Form` + `Material` has landed.
3. **Quadrature, then `LinearForm`** (§2) — a real quadrature layer is what lets `f` vary in
   space/time; the linear form (and variable-coefficient bilinear forms) follow from it. Until
   then the load is `M @ f` and needs no new object.
4. **Strategy registry for `Solver`** (§2) — then fold `EnergySolver` onto the shared
   elimination path and a common protocol.
5. **`TimeIntegrator`; move `dt`/`iters` off `Heat`/`Wave`** — breaking API change.
6. **Uniform drivers** — `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation, which also removes the `_select_*`
   plumbing in §1.
7. **Invert the core → plot dependency** (§2) and clear §1's unused modules — small, and
   independent of all of the above.
