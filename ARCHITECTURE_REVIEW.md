# Architecture Review — `fem/`

A structural read of the package, deliberately skipping the sparse-matrix and performance
items already covered in `BACKLOG.md` §2. Claims about what is dead or uncalled were
verified by grepping for callers and definitions, not inferred.

Nothing here is a known-broken path any more; what remains is design and dead weight.

Line references are against `0d8b3ba`. They drift; the symbol names are the durable anchor,
and `ARCHITECTURE.md` deliberately uses those alone.

Legend: 🟡 design / maintainability · 🟢 small

---

## 1. Dead paths and unused code

### 🟡 `TopologyOptimizer`'s objective plumbing carries dead parameters

`solve` (`fem/topology.py:97`) resolves two callables and then uses one of them:

- `objective_func` is unpacked at `:106` and never called. Only `gradient_func` is —
  the objective value is never evaluated during optimization, so selecting an
  objective picks a gradient and nothing else.
- `optimization_args` is a parameter of `solve` (`:102`), threaded into
  `_select_optimization` (`:155`), and then ignored entirely.

That dead parameter is exactly what AGENTS.md means by "typing the data is what makes dead
parameters visible" — the `Sequence[Any] | None` bag is what conceals it. Objectives would
be better as small classes or closures carrying their own configuration, rather than a
string plus a positional-args tuple resolved through `_select_*`.

### 🟡 Unused modules and members

- `fem/quadrature.py` — no importers anywhere in the repo. Not a head start on the quadrature
  layer §2 wants, either: the rules take `(func, polygon_vertices)`, where a real layer needs
  reference-element points and weights. Delete or rewrite; `BACKLOG.md` flags "decide its fate".
- `fem/numerics.py:91` `class color` — no callers. Superseded by the move to `logging`.
- `fem/numerics.py:79` `timer` — no callers.
- `check_gradient` / `check_hessian` (`fem/numerics.py:27,53`) both end in `plt.show()`,
  so they block. They are dev tools that cannot be called from a test, and they pull
  `matplotlib` into a top-level import of a core module. Their only live caller is
  `StVenantKirchhoff.check_gradients` (`fem/energies.py:163`), itself uncalled; the
  `EnergySolver` uses are the parked commented-out lines at `energy_solver.py:57-58`.

---

## 2. Structural issues worth a refactor

### 🟡 `Solver` is a god class with exact-type dispatch

`Solver.solve` (`fem/solver.py:152`) dispatches through a dict literal keyed by
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
- **Implicit temporal coupling.** `self.material` is set only inside the `LinearElastic`
  branch of `assemble_everything` (`:179`) and read unconditionally in
  `solve_linear_elastic` (`:333`). The two methods must be called in order, and nothing
  encodes that.
- The class carries five PDE solve routines, the linear solver, assembly, and the
  adaptive-refinement loop.

Note that the obvious fix — putting `solve()` on `Equation` — is *wrong*, and the
`Equation` docstring says why: one equation may be served by several solvers (`Solver` vs
`EnergySolver` for `LinearElastic`). The fix that preserves that separation is a small
strategy registry (`Poisson → PoissonStrategy`, …), resolved by an MRO walk so subclasses
dispatch correctly, with each strategy owning its own assembly and its own result fields.

### 🟡 `EnergySolver` and `Solver` share no interface

The two solvers reimplement their step loops separately and re-unpack the resolved BC by hand,
and neither implements a shared interface: `TopologyOptimizer` hardcodes `Solver`
(`fem/topology.py:44`). A `SolverProtocol` — `(mesh, equation, bc) -> Solution` — would make
them substitutable and let the optimizer accept either.

### 🟡 The load vector waits on quadrature, not on a `LinearForm`

The load `L(v) = ∫ f·v` is `self.b = self.M @ source_load` (`solver.py:199`) — the mass form as
a load operator, which is the *exact* integral of `f`'s P1 interpolant (`M_ij = ∫ φ_i φ_j`). A
standalone `LinearForm` adds capability only once `f` varies *within* an element, which needs
quadrature to sample it at interior points — the same machinery non-constant coefficients
(`∫ κ(x) ∇u·∇v`) and P2 elements need. So a `LinearForm` belongs with the quadrature work, not
before it. It is *not* blocked on a time-varying source `f(·, t)` (just re-evaluate `M @ f_t`
per step) or on Robin conditions (those need a *bilinear* boundary form, which
`assemble(form, boundary=True)` already supports).

### 🟡 `Solution` is a stringly-typed dict

Against the repo's own "prefer typed over stringly-typed" convention:

- Keys are undiscoverable: `set_values("u", …)`, `values['compliance']`,
  `get_values('u_values', iter_idx=-1)`.
- `combine_solutions` (`solution.py:77`) invents a `'_list'` key-suffix convention, which
  `TopologyOptimizer._get_deformed_mesh` (`topology.py:168`) then probes with
  `try/except (KeyError, IndexError)`.
- `get_values(None)` returning a zero array (`solution.py:38`) is a plotting convenience
  leaking into the data model.
- Steady results (`u`) and time series (`u_values`, `t_values`) share one container with
  nothing in the type distinguishing them, so `mode` conversion has to *infer* meaning from
  `len(values)` (`:46`, `:53`) — which silently picks the wrong branch whenever
  `n_elements == n_vertices`.
- The `mode` axis of `get_values` has no *production* callers — nothing in `fem/`, `examples/`,
  or the README snippet passes it — so the length-guessing above is unused as well as unsound.
- `combine_solutions` hardcodes a component count of 2 with its own `# TODO: bit weird`.

Per-solve-type dataclasses (`SteadySolution`, `TransientSolution`, `ElasticSolution`) would
make the fields discoverable and delete the length-guessing entirely.

### 🟡 `import fem` still pulls in matplotlib

Two paths reach the plot layer from core:

- `fem/numerics.py:8` imports `matplotlib.pyplot` at module scope, only for the blocking
  `check_gradient` / `check_hessian` dev tools in §1. `fem.topology` imports `numerics` for
  `calculate_smoothing_matrix`, so this is on a live path — a genuine layering violation that
  goes away with the §1 dev tools.
- `fem/__init__.py:54` re-exports `Plotter` and `PlotMode` as public API. Deliberate, and only
  worth revisiting if the package should be importable without a plotting backend installed.

---

## 3. Smaller items

- 🟢 `Mesh.convert_element_values_to_vertex_values` (`mesh.py:43`) is last-writer-wins at
  shared vertices, while its inverse (`:36`) averages. The asymmetry is probably unintended —
  an area-weighted average is the usual choice.

---

## Suggested order

1. **Typed `Solution`** (§2) — independent of everything below, and the largest felt
   improvement for callers. Touches every call site, so it wants to land on its own.
2. **Quadrature, then `LinearForm`** (§2) — a real quadrature layer is what lets `f` vary
   within an element; the linear form (and variable-coefficient bilinear forms) follow from it.
   Until then the load is `M @ f` and needs no new object.
3. **Strategy registry for `Solver`** (§2) — then fold both solvers onto a common protocol.
4. **`TimeIntegrator`; move `dt`/`iters` off `Heat`/`Wave`** — breaking API change, and its
   shape is an open fork (`ARCHITECTURE.md` §3). Settle that first.
5. **Uniform drivers** — `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation, which also removes the `_select_*`
   plumbing in §1.
6. **Clear the remaining core → plot paths** (§2) and §1's unused modules — small, and
   independent of all of the above.
