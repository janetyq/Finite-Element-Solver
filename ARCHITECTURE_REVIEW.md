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

### 🟡 Elements know physics they should not

`LinearElement.calculate_stiffness_matrix` (`fem/elements.py:61`) branches on DOF count to
decide which PDE it is discretizing:

```python
if n_components == 1:
    return self.grad_phi @ self.grad_phi.T * self.volume
# otherwise, the equation is linear elastic
idx = kwargs['idx']
B, D = self.calculate_B(), self.calculate_D(kwargs['mu'][idx], kwargs['lamb'][idx])
```

An element type is inferring the equation from the number of DOFs per node. It also receives
material parameters as untyped `**kwargs` carrying the *global* `mu`/`lamb` arrays plus its
own index, rather than its own scalar values — the element reaches into a global array to
find itself.

`FunctionSpace.assemble_stiffness` (`space.py`) now passes the material through as
`**kwargs` rather than a mesh doing it, which narrows the smell to one method but does not
remove it: the space still has to forward data it has no opinion about.

There is also a hidden physics assumption in the load assembly:

```python
self.b = (self.M @ source_load.flatten()).flatten()
```

That is `L(v) = ∫ f·v` evaluated by multiplying through the mass matrix — correct for P1
elements, but it is a *linear form* wearing a disguise. Time-varying sources and
non-constant-coefficient equations have nowhere natural to go because the linear form has
no representation.

### 🟡 The missing `Form` abstraction

FEM turns a PDE into "find `u` such that `a(u,v) = L(v)` for all test functions `v`":

| PDE | `a(u,v)` — the bilinear form | becomes |
|---|---|---|
| L2 projection | `∫ u·v` | mass matrix |
| Poisson | `∫ ∇u·∇v` | stiffness matrix |
| Elasticity | `∫ ε(u)ᵀ D ε(v)` | stiffness matrix |

The assembly loop is identical for all three: loop elements, compute a local matrix from the
integrand, scatter into the global one. **The only thing that varies per PDE is the
integrand.** That integrand is the form — and it currently has no home, so it is smeared
across `Element.calculate_stiffness_matrix` (the physics branch),
`FunctionSpace.assemble_stiffness` (the `**kwargs` passthrough), and
`Solver.assemble_everything` (which computes the material and hands it over).

A `BilinearForm` makes the integrand an object:

```python
class BilinearForm(Protocol):
    def element_matrix(self, element: Element, e_idx: int) -> Matrix: ...

class DiffusionForm:
    def element_matrix(self, element, e_idx):
        return element.grad_phi @ element.grad_phi.T * element.volume

class ElasticityForm:
    def __init__(self, material: Material):
        self.material = material
    def element_matrix(self, element, e_idx):
        B = element.calculate_B()
        D = self.material.constitutive_matrix(e_idx)
        return B.T @ D @ B * element.volume
```

`FunctionSpace._assemble` is already this loop, parameterised by a callable that returns a
local matrix. Taking a form instead is a change of what gets passed in, not new machinery:

```python
def assemble(space: FunctionSpace, form: BilinearForm) -> Matrix:
    return space._assemble(
        space.mesh.elements,
        lambda e_idx: form.element_matrix(space.element_objs[e_idx], e_idx),
    )
```

One loop. No `**kwargs`, no `n_components` branch. The element supplies geometry and
shape functions; the form supplies physics; the material supplies constitutive constants.

**What this concretely unblocks:**

1. **Retires `if n_components == 1`.** Which physics you get comes from *which form you
   passed*, not from counting DOFs.
2. **Robin BCs.** `BCType.ROBIN` exists and `resolve` refuses it. A Robin condition is
   `∫ α·u·v ds` — a bilinear form on the boundary. With forms you assemble it and add to
   the LHS.
3. **`quadrature.py` gets a purpose.** Five rules, zero importers — because no object's job
   is "integrate this integrand over an element." A form is that object.
4. **Variable coefficients.** The Laplacian is currently a hardcoded closed-form
   `∇φ·∇φ·volume`. A form with quadrature does `∫ κ(x)∇u·∇v`.
5. **Higher-order elements become additive.** P2 needs new shape functions *and* real
   quadrature. `Form` + `FunctionSpace` is the pair that makes it a new element class
   rather than another component-count branch.
6. **Unifies the two elasticity models.** `elements.py` has D-matrices (small strain),
   `energies.py` has energy densities (Green–Lagrange). A shared `Material` consumed by a
   linear `ElasticityForm` and a nonlinear energy form makes them two materials under one
   interface rather than two unrelated subsystems.

**Interaction with `FieldShape`.** An `ElasticityForm` inherently *is* vector-valued — its
`B` matrix assumes one component per spatial direction. So the form knows the component count
without being told. The counter-argument is that `Equation` is the user-facing declaration
(`LinearElastic(E, nu)`) and `Form` is internal machinery; the field shape stays on the
declaration and the form derives from it. They are complementary, not redundant — but
`Equation.field` is the piece most likely to simplify once forms exist.

`FunctionSpace` already owns the DOF map and an `_assemble` loop of exactly this shape, so
this is now a matter of parameterising that loop by an integrand rather than building the
layer underneath it.

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
2. **Extract `Form` + `Material`** (§2) — move `calculate_D` off `Element`, unify with
   `energies.py`, and parameterise `FunctionSpace._assemble` by an integrand. Retires the
   `**kwargs`, and the `if n_components == 1` physics branch with it. The keystone now that
   the space exists.
3. **`DiscreteSystem`, then dense → sparse** — the single seam where the algebra layer
   changes, and the item `BACKLOG.md` §2 calls the highest-leverage one.
4. **Strategy registry for `Solver`** (§2) — then fold `EnergySolver` onto the shared
   elimination path and a common protocol.
5. **`TimeIntegrator`; move `dt`/`iters` off `Heat`/`Wave`** — breaking API change.
6. **Uniform drivers** — `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation, which also removes the `_select_*`
   plumbing in §1.
7. **Invert the core → plot dependency** (§2) and clear §1's unused modules — small, and
   independent of all of the above.

Step 2 is the load-bearing one: it is what the `FunctionSpace` work was clearing the way for.
