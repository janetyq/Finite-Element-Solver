# Architecture Review — `fem/`

A structural read of the package, deliberately skipping the sparse-matrix and performance
items already covered in `BACKLOG.md` §2. Everything in "Confirmed broken" was verified by
grepping for callers/definitions, not inferred.

Line references are against `c9cdb66`. They drift; the symbol names are the durable anchor,
and `ARCHITECTURE.md` deliberately uses those alone.

Legend: 🔴 broken today · 🟡 design / maintainability · 🟢 small

---

## 1. Confirmed broken or dead paths

### 🔴 `TopologyOptimizer`'s `target_compliance` objective cannot run

`fem/topology.py:124` and `:128`:

```python
def target_compliance_objective(self, args):
    target = args[0]
    return (self.compliance() - target)**2      # TypeError
```

`compliance` and `compliance_gradient` (`:118`, `:121`) both take a *required* `args`
parameter, so both of these calls raise `TypeError` immediately. Selecting
`objective_name='target_compliance'` is a guaranteed crash.

Two related smells in the same method:

- `objective_func` is unpacked at `:94` and never used. Only `gradient_func` is —
  the objective value is never evaluated during optimization.
- `optimization_args` is a parameter of `solve` (`:90`), threaded into
  `_select_optimization` (`:140`), and then ignored entirely.

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
  `matplotlib` into a top-level import of a core module. Their only live caller is
  `LinearElasticEnergyDensity.check_gradients` (`fem/energies.py:102`), itself uncalled;
  the `EnergySolver` uses are the parked commented-out blocks at `energy_solver.py:62-66`.

---

## 2. Structural issues worth a refactor

### 🟡 `FEMesh` conflates geometry with assembled operators, and holds mutable solver state

`FEMesh.__init__` eagerly calls `prepare_matrices()`, assembling four dense N×N matrices at
`dim=1` before anything has asked for them. `Solver.assemble_everything`
(`fem/solver.py:179`) then calls `prepare_matrices(dim=2)` and discards all of that work.

Worse, `prepare_matrices` mutates `self.dim` (`femesh.py:61`). A mesh's state therefore
depends on whichever solver touched it last. Two solvers sharing one `FEMesh` at different
`dim` silently corrupt each other's operators.

**Concrete trap falling out of this:** `Solution.get_deformed_mesh` (`fem/solution.py:70`)
does `self.femesh.copy()` — a full re-assembly, purely to plot — and then mutates
`femesh_deformed.vertices += u.reshape(-1, self.dim)`. The copy's `element_objs`, with
their cached `volume` and `grad_phi`, were built from the *undeformed* vertices in
`__init__` and are now stale with respect to the vertices they claim to describe. Harmless
for plotting today; wrong the moment anything computes on a deformed mesh.

**Suggested shape:** make `Mesh` immutable geometry, and move `M`, `M_b`, `K`, `K_b` onto a
`dim`-keyed, lazily-built `Operators` object. This also delivers the assembly caching
`BACKLOG.md` §2 asks for ("`assemble_everything` runs on every `solve()`") as a side effect,
and removes the `# TODO: don't call this every time` at `fem/solver.py:160`.

### 🟡 `Solver` is a god class with exact-type dispatch

`Solver.solve` (`fem/solver.py:159`) dispatches through a dict literal keyed by
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
  `LinearElastic` branch of `assemble_everything` (`:184`) and read unconditionally in
  `solve_linear_elastic` (`:343`). The two methods must be called in order, and nothing
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
exists as `Solver.solve_nonlinear_system` (`fem/solver.py:223`), and re-unpacks the resolved
BC by hand into `self.free` / `self.fixed` / `self.fixed_values` (`:50-52`).

The more substantive divergence is Dirichlet handling. `Solver.solve_linear_system`
eliminates fixed DOFs properly (`solver.py:196`). `energy_hessian` (`:120`) instead zeroes
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
(`fem/topology.py:41`). A `SolverProtocol` — `(femesh, equation, bc) -> Solution` — would
make them substitutable and let the optimizer accept either.

### 🟡 `dim` conflates two quantities

`dim` appears on `Equation` (ClassVar), `Solver`, `EnergySolver`, `FEMesh` (mutable),
`ResolvedBC`, `Solution`, and as a parameter on `assemble_matrix`, `calculate_mass_matrix`,
`dof_indices`, `evaluate_field`. Seven homes for one number, one of them mutable and shared.

The deeper problem: **`dim` means two different things.**

- **components per node** (`n_components`) — 1 for a scalar PDE, 2 or 3 for displacement.
  Mathematically, this is `k` in the unknown field `u: Ω → ℝᵏ`.
- **spatial dimension** (`spatial_dim`) — 2 for a triangle mesh, 3 for a tet mesh. Already
  available as `vertices.shape[1]`.

For 2D elasticity the two coincide (`dim = 2` means both). They diverge for every other
combination: Poisson on a tet mesh (`spatial_dim = 3`, `n_components = 1`), 3D elasticity
(`spatial_dim = 3`, `n_components = 3`), 1D elasticity (`spatial_dim = 1`, `n_components = 1`
— and the `if dim == 1` branch would silently apply the Laplacian instead).

This conflation is why:

- `LinearElastic.dim: ClassVar[int] = 2` **cannot express 3D elasticity** — the correct
  statement is `n_components == spatial_dim`, which a constant cannot say.
- `EnergySolver`'s 2D guard (`if self.dim != 2`) is **dead code** — `self.dim` is copied from
  `LinearElastic.dim` which is always 2, so the guard can never fire. The author meant
  *spatial* dimension and only had `dim` to say it with.
- `FEMesh.prepare_matrices(dim=...)` must be re-run to change components, rebuilding
  geometry-only quantities that never depended on `dim`.

**The fix: store a *kind*, derive the *count*.** The component count is a consequence of two
facts — what kind of value the field takes (scalar vs. vector) and what the domain's spatial
dimension is. Store the kind on the equation as a `FieldRank` enum; read spatial dimension
from the mesh; derive the count at the point where the two meet (solver construction,
eventually `FunctionSpace` construction):

```
n_components = equation.rank.components_for(mesh.spatial_dim)
```

Most of the diff is a rename: every downstream `dim` that means components-per-node becomes
`n_components`, and the two meanings become unspellable as the same name. The rename doesn't
fix the element's physics branch — that needs `Form` — but it makes the branch's fragility
legible instead of hidden.

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

`FEMesh.assemble_matrix` (`femesh.py:73`) compounds it with two `Literal` string parameters
and its own admission: `# TODO: term "element" is overloaded here, and its a bit hacky`.

There is also a hidden physics assumption in the load assembly:

```python
self.b = (self.femesh.M @ source_load.flatten()).flatten()
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
across `Element.calculate_stiffness_matrix` (the physics branch), `FEMesh.assemble_matrix`
(the string selectors and `**kwargs`), and `Solver.assemble_everything` (the material
passthrough).

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

Assembly becomes a single generic function:

```python
def assemble(space: FunctionSpace, form: BilinearForm) -> Matrix:
    A = np.zeros((space.n_dofs, space.n_dofs))
    for e_idx in range(space.n_elements):
        idxs = space.dof_indices(e_idx)
        A[np.ix_(idxs, idxs)] += form.element_matrix(space.element(e_idx), e_idx)
    return A
```

One loop. No strings, no `**kwargs`, no `dim` branch. The element supplies geometry and
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
   rather than another `dim` branch.
6. **Unifies the two elasticity models.** `elements.py` has D-matrices (small strain),
   `energies.py` has energy densities (Green–Lagrange). A shared `Material` consumed by a
   linear `ElasticityForm` and a nonlinear energy form makes them two materials under one
   interface rather than two unrelated subsystems.

**Interaction with `FieldRank`.** An `ElasticityForm` inherently *is* vector-valued — its
`B` matrix assumes one component per spatial direction. So the form knows the component count
without being told. The counter-argument is that `Equation` is the user-facing declaration
(`LinearElastic(E, nu)`) and `Form` is internal machinery; rank stays on the declaration and
the form derives from it. They are complementary, not redundant — but rank is the piece most
likely to simplify once forms exist.

**Depends on `FunctionSpace`.** `assemble(space, form)` needs something that owns the DOF
map. This is why `Form` is step 4, not step 3.

### 🟡 `Solution` is a stringly-typed dict

Against the repo's own "prefer typed over stringly-typed" convention:

- Keys are undiscoverable: `set_values("u", …)`, `values['compliance']`,
  `get_values('u_values', iter_idx=-1)`.
- `combine_solutions` (`solution.py:77`) invents a `'_list'` key-suffix convention, which
  `TopologyOptimizer._get_deformed_mesh` (`topology.py:153`) then probes with
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
- `combine_solutions` hardcodes `dim=2` with its own `# TODO: bit weird`.

Per-solve-type dataclasses (`SteadySolution`, `TransientSolution`, `ElasticSolution`) would
make the fields discoverable and delete the length-guessing entirely.

### 🟡 Core still depends on the plot layer

`fem/mesh/mesh.py:8` imports `Plotter, PlotMode` at module scope, for the convenience method
`Mesh.plot()` (`:45`). Commit 2413319 decoupled core solver/mesh code from the plotter;
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
- 🟢 `FEMesh.get_edges_in_idxs` (`femesh.py:146`) and `get_boundary_idxs_in_rect` (`:162`) are
  2D-only (both unpack `x, y`) and duplicate what `fem.regions` now does properly and
  dimension-generally. They look like pre-`regions` leftovers — candidates for deletion.
- 🟢 `LinearElement.calculate_dF_dx` (`elements.py:78`) is a 4-deep Python loop building a
  Kronecker product, with its own `# TODO: figure out kronecker product`. Same pattern in
  `energies.py` `calculate_d2S_dF2` (6-deep). Both are per-element and run inside the Newton
  loop.

---

## Suggested order

1. **Fix the broken paths** (§1) — cheap, and the `target_compliance` objective is a latent
   landmine that only looks fine because nothing exercises it.
2. **`FieldRank` + `dim` → `n_components` rename** (§2) — small, independent, no new layers.
   Makes the two meanings of `dim` unspellable as the same name and unblocks 3D elasticity.
   Most of the diff is a rename; a 3D patch test is the natural checkpoint.
3. **Typed `Solution`** (§2) — independent of 4–6; the largest felt improvement for callers.
4. **Extract `FunctionSpace` from `FEMesh`** — the keystone. Move `dof_indices`, element
   objects, and cached operators. Kills the mutable-`dim` trap and gives operator caching a
   natural home.
5. **Extract `Form` + `Material`** (§2) — move `calculate_D` off `Element`, unify with
   `energies.py`, make assembly `assemble(space, form)`. Retires the `**kwargs`, the
   `Literal` selectors, and the `if n_components == 1` physics branch.
6. **`DiscreteSystem`, then dense → sparse** — the single seam where the algebra layer changes.
7. **`TimeIntegrator`; move `dt`/`iters` off `Heat`/`Wave`** — breaking API change.
8. **Uniform drivers** — `adaptive_refinement` becomes a class; `TopologyOptimizer` takes a
   problem factory rather than mutating an equation.

Steps 2–3 are independent and can land in either order. Steps 4–5 are the load-bearing
structural changes. Steps 1, 7, 8 are independent and can be done any time.
