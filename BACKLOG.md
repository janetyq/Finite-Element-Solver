# Backlog — Finite Element Solver

The living list of still-open work on the solver: correctness, performance, architecture,
and open-ended ideas. This is a genuinely impressive project — a hand-rolled FEM solver
spanning meshing, multiple PDEs, adaptive refinement, and topology optimization is a large
amount of correct, non-trivial numerical code. The notes below are about hardening and
scaling what's already here, not a knock on the design.

Legend: 🔴 bug / correctness · 🟠 performance / scaling · 🟡 design / maintainability · 💡 idea
· effort 🟢 low · 🟡 medium · 🔴 high

## At a glance

| Area | Item | Effort | Detail |
|---|---|:---:|---|
| Scaling | Sparse matrices + solver — **highest leverage** | 🔴 | [§2](#2-performance--scaling) |
| Scaling | Cache assembly across `solve()` calls | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Sparsify smoothing matrix / EnergySolver Hessian | 🟡 | [§2](#2-performance--scaling) |
| Scaling | O(n²) linear scans in refinement/meshing | 🟡 | [§2](#2-performance--scaling) |
| Correctness | Triangle-only `range(3)` edge/boundary extraction | 🟡 | [§1](#1-bugs--correctness) |
| Correctness | `adaptive_refinement` bug + inverted loop | 🔴 | [§1](#1-bugs--correctness) |
| Correctness | Wave solver ignores Neumann/force; `np.roll` load bug | 🟡 | [§1](#1-bugs--correctness) |
| Correctness | 2D-only assertions/tensors in "general" paths | 🟡 | [§3](#3-architecture--maintainability) |
| Design | `pickle` persistence → consolidate into `fem/io.py` | 🟡 | [§3](#3-architecture--maintainability) |
| Nit | `BoundaryConditions.add()` silently overwrites duplicates | 🟢 | [§4](#4-smaller-nits) |
| Nit | `solve_linear_elastic` redundant re-fetch/indexing | 🟢 | [§4](#4-smaller-nits) |
| Nit | `oc_density` fragile bisection | 🟢 | [§4](#4-smaller-nits) |
| Numerics | Gaussian quadrature layer (decide `quadrature.py`'s fate) | 🔴 | [§5](#5-open-ended-suggestions--future-ideas) |
| Numerics | Higher-order (quadratic) elements | 🔴 | [§5](#5-open-ended-suggestions--future-ideas) |
| Numerics | Time-integrator abstraction | 🟡 | [§5](#5-open-ended-suggestions--future-ideas) |
| Numerics | A-posteriori error estimator | 🔴 | [§5](#5-open-ended-suggestions--future-ideas) |
| Tooling | Coverage (`pytest-cov`), type hints, pre-commit, README refresh | 🟢–🟡 | [§5](#5-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

### 🔴 Triangle-only edge extraction
`Mesh._get_all_edges` (`fem/mesh/mesh.py`) loops `for i in range(3)`, so `edges` is only
correct for 2D triangle meshes; for 1D line elements and 3D tets it produces wrong edges.
Since the base `Mesh` is used for all dimensions, edge extraction should key off
`self.elements.shape[1]` (or live in the element classes). `get_boundary_from_vertices_elements`
in `fem/geometry.py` has the same triangle-only assumption.

### 🔴 `adaptive_refinement` has a bug and an inverted loop condition
`fem/solver.py:Solver.adaptive_refinement` carries an explicit `# TODO: there's a bug
somewhere`, and the guard
```python
while len(self.femesh.elements) < max_triangles or max_iters == 0:
```
is almost certainly meant to be `... and max_iters > 0`. The residual scaffolding it relies
on is commented out, and the demo path in `examples/solver_demos.py` raises
`NotImplementedError`, so the feature is currently non-functional.

### 🟠 Wave solver doesn't honor Neumann/force BCs
`fem/solver.py:solve_wave` time-steps with `use_bc=False`, and `b_right` uses
`np.roll(self.b, -1)`, which rolls a *spatial* load vector by one index — a time-averaging
idea applied to the wrong axis. It's harmless while `b == 0` (the demos have no forcing),
but a correctness trap the moment someone adds a source term.

---

## 2. Performance & Scaling

### 🟠 Everything is dense — this is the single biggest limiter
`FEMesh.assemble_matrix` (`fem/mesh/femesh.py`) builds `A = np.zeros((dim*N, dim*N))` and
solves via `np.linalg.solve` (`fem/solver.py:solve_linear_system`). FEM matrices are
extremely sparse (each row has a handful of nonzeros), so this is `O(N²)` memory and
`O(N³)` solve time. On the 40×40 meshes in the tests that's fine; it will fall over well
before "interesting" resolutions. Concretely:
- Assemble with `scipy.sparse.lil_matrix`/COO triplets, convert to CSR.
- Solve with `scipy.sparse.linalg.spsolve` (or `cg`/`splu` with caching for the time-stepping
  loops, where the matrix is constant across iterations).

This one change probably unlocks 1–2 orders of magnitude in mesh size, and the MMS
convergence test guards correctness through the migration.

### 🟠 `assemble_everything` runs on every `solve()`
`fem/solver.py:Solver.solve` even flags it: `# TODO: don't call this every time`. For
heat/wave the system matrices are rebuilt implicitly each timestep via re-assembly; for
topology optimization `solve()` is called every iteration. Cache `M`, `M_b`, `K` and only
re-factor when the mesh or material actually changes. For time-stepping, pre-factor
`(M + K·dt)` once (LU) and reuse.

### 🟠 `calculate_smoothing_matrix` is dense `O(n_elem²)`
`fem/numerics.py:calculate_smoothing_matrix` materializes a full element-by-element
distance matrix. For topology optimization at any real resolution this dominates memory. A
spatial hash / KD-tree (`scipy.spatial.cKDTree.query_ball_point`) building a sparse weight
matrix would scale far better and is a near drop-in.

### 🟠 Refinement and meshing are `O(n²)` from linear scans
`fem/mesh/refinement.py` is self-described as "very inefficient": `get_shared_triangle`,
`get_triangle_idx`, and `get_point_idx` each do a full linear scan of all triangles/vertices,
inside refinement loops. `get_point_idx` in particular scans every vertex to dedupe midpoints
— an edge→midpoint-index dict would make it `O(1)`. Similarly
`get_boundary_from_vertices_elements` (`fem/geometry.py`) is `O(edges × elements)`; a single
pass counting edge occurrences in a dict is `O(elements)`.

### 🟠 `EnergySolver` Hessian is dense and rebuilt each Newton step
`fem/energy_solver.py:energy_hessian` allocates an `(n·dim, n·dim)` dense Hessian every
iteration and solves it densely. Same sparse story as above; here it matters even more
because it's inside a Newton loop.

---

## 3. Architecture & Maintainability

### 🟡 `pickle` for persistence
`Solution.save/load` (`fem/solution.py`) uses `pickle`, which executes arbitrary code on
load and is fragile across refactors (it stores the class path). For a research tool
loading only your own files it's low-risk, but a plain `.npz`/JSON schema for the numeric
arrays would be safer and more portable. `Mesh.save/load` (JSON) and this would sit well
together in a single `fem/io.py`.

### 🟡 Hardcoded 2D assumptions in "generalized" paths
Some spots still assume 2D: `EnergySolver` asserts `dim == 2`, and
`LinearElasticEnergyDensity` (`fem/energies.py`) is hardwired to `np.eye(2)` / `(2,2,2,2)`
tensors. Worth an explicit "not yet N-D" boundary so the limitation is visible.

---

## 4. Smaller Nits

- `Solver.solve_linear_elastic` (`fem/solver.py`) re-fetches `element = self.femesh.elements[e_idx]`
  on the first line of the loop even though it's already the loop variable, and indexes
  `self.femesh.element_objs[e_idx]` three separate times — bind both to locals.
- `BoundaryConditions.add()` (`fem/boundary.py`) silently overwrites a duplicate BC on the
  same node of the same type (dict assignment, last write wins, no warning).
- `oc_density` upper bound `hi = 1e15` with a `while (lo*(1+1e-15)) < hi` termination is a
  fragile way to bisect; a fixed iteration count or relative tolerance on `hi-lo` is more
  predictable.
- The README's "Project Structure" and described capabilities have drifted from the code; a
  refresh pass would help.

---

## 5. Open-Ended Suggestions & Future Ideas

**Numerics**
- 💡 **Higher-order elements.** Already on the roadmap (quadratic basis). The `Element` class
  hierarchy is well-positioned — add `QuadraticTriangleElement` with its own shape functions
  and a real quadrature rule (the `fem/quadrature.py` rules are written but not yet wired into
  assembly).
- 💡 **Proper Gaussian quadrature.** Assembly currently uses closed-form linear-element
  integrals. A general quadrature layer (reference element + Gauss points + Jacobian) would
  make adding new element types and variable coefficients far easier, and is a prerequisite for
  the quadratic elements above. Decide `quadrature.py`'s fate: integrate it or mark it WIP.
- 💡 **Iterative solvers + preconditioning.** Once sparse, add CG with a Jacobi/AMG
  preconditioner for the SPD systems (Poisson, elasticity) — where large 3D problems become
  tractable.
- 💡 **A posteriori error estimator** so adaptive refinement is fully closed-loop — the
  residual scaffolding is already sketched in `fem/solver.py`.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear
  hyperelasticity via the existing `EnergySolver`/`Energies` machinery) all fit the current
  architecture well. Finishing `NeohookeanEnergyDensity` would immediately give a nonlinear
  material through the already-working Newton solver.
- 💡 **Time-integration abstraction.** Backward-Euler (heat) and Crank–Nicolson (wave) are
  hand-coded inline. A small `TimeIntegrator` interface (θ-method / generalized-α) would
  deduplicate and make it trivial to add new dynamics.
- 💡 **Full BC support for the wave solver** (Neumann/force aren't honored), and a general
  Robin BC path (the README mentions Robin conditions but `BoundaryConditions` only models
  Dirichlet/Neumann explicitly).

**Engineering**
- 💡 **Coverage.** Add `pytest-cov`, then fill gaps — `svg`, `generation` (Rupperts/approx
  mesh), the 3D tet path, and adaptive refinement have no *correctness* tests.
- 💡 **Type hints + docstrings on the public API** (`Solver`, `FEMesh`, `BoundaryConditions`,
  `Equation`), plus `pyright`/`mypy` (gradual) — the surface most likely to be used by others
  (or future-you).
- 💡 **pre-commit hooks** (ruff + whitespace) so the CI checks run locally before each commit.
- 💡 **Benchmarks.** A tiny script timing assembly + solve vs. mesh size would make the impact
  of the sparse migration concrete and guard against future regressions.

---

## Suggested Priority Order

1. **Sparse matrices + solver** (§2) — the highest-leverage single change for capability.
2. **Quick nits** (§4) and the correctness bugs (§1) — cheap, and clear the deck.
3. **Coverage + type hints** (§5) — deepen the safety net before the bigger numerics work.
4. **Then the numerics roadmap** — quadrature → higher-order elements → time-integrator →
   adaptive refinement.
