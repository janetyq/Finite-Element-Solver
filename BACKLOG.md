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
| Numerics | Gaussian quadrature layer (decide `quadrature.py`'s fate) | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Higher-order (quadratic) elements | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Time-integrator abstraction | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | A-posteriori error estimator | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Coverage (`pytest-cov`), API docstrings, pre-commit, README refresh | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

*(No open correctness bugs. Adaptive refinement is now closed-loop except for the error
estimator itself — see [§3](#3-open-ended-suggestions--future-ideas).)*

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
— an edge→midpoint-index dict would make it `O(1)`.

### 🟠 `EnergySolver` Hessian is dense and rebuilt each Newton step
`fem/energy_solver.py:energy_hessian` allocates an `(n·dim, n·dim)` dense Hessian every
iteration and solves it densely. Same sparse story as above; here it matters even more
because it's inside a Newton loop.

---

## 3. Open-Ended Suggestions & Future Ideas

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
  residual scaffolding is already sketched in `fem/solver.py`. `Solver.adaptive_refinement`
  takes the estimator as a callable `(solver) -> per-element error`, so this drops straight in.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear
  hyperelasticity via the existing `EnergySolver`/`Energies` machinery) all fit the current
  architecture well. Finishing `NeohookeanEnergyDensity` would immediately give a nonlinear
  material through the already-working Newton solver.
- 💡 **N-D elasticity.** `LinearElasticEnergyDensity` and `EnergySolver` now reject non-2D
  input explicitly instead of failing deep inside an einsum, but their tensors are still built
  at fixed rank. Generalizing them over `dim` is the actual feature behind that guard.
- 💡 **Time-integration abstraction.** Backward-Euler (heat) and Crank–Nicolson (wave) are
  hand-coded inline. A small `TimeIntegrator` interface (θ-method / generalized-α) would
  deduplicate and make it trivial to add new dynamics.
- 💡 **Robin BC path.** `BCType.ROBIN` exists and `resolve` refuses it. A Robin condition adds
  a term to the *system matrix* rather than the load, so the work is in
  `Solver.assemble_everything`, using the already-assembled `femesh.K_b` / `femesh.M_b`
  (`K_b` is currently built for `dim == 1` and otherwise unused).
- 💡 **Time-varying loads and Dirichlet data.** `Equation.source` and the BC values are
  functions of position only. `self.b` is built once and assumed constant in time;
  `solve_wave` notes where the Crank–Nicolson `b_n`/`b_{n+1}` average collapses because of it,
  and `_wave_block_constraints` assumes Dirichlet values are constant (so `du/dt = 0` at fixed
  nodes). Adding a `t` argument to those callables is the natural extension.
- 💡 **External work term for `EnergySolver`.** It minimizes the internal elastic energy only
  and builds no load vector, so it currently rejects `Equation.source` outright. Adding the
  external work term `-f · u` (and its gradient/Hessian contributions) would make it accept
  forced problems, which is also a prerequisite for using it on the nonlinear roadmap.

**Engineering**
- 💡 **Coverage.** Add `pytest-cov`, then fill gaps — `svg`, `generation` (Rupperts/approx
  mesh), the 3D tet path, and adaptive refinement have no *correctness* tests.
- 💡 **Docstrings on the public API.** Type hints and `pyright` are in place and gating CI;
  the prose half is still open. The biggest modules are the least documented — `solver.py`,
  `mesh/refinement.py`, `mesh/generation.py`, `elements.py` and `topology.py` have no
  module docstring, while the small helpers (`geometry`, `materials`, `quadrature`) do.
- 💡 **Tighten pyright to `standard`.** It runs in `basic`, which infers types for the
  unannotated internals rather than demanding annotations. Annotating the internals
  (`refinement`, `generation`, `energies`, `plot`) would let the mode step up.
- 💡 **pre-commit hooks** (ruff + whitespace) so the CI checks run locally before each commit.
- 💡 **README refresh.** The "Project Structure" section and described capabilities have
  drifted from the code.
- 💡 **Benchmarks.** A tiny script timing assembly + solve vs. mesh size would make the impact
  of the sparse migration concrete and guard against future regressions.
- 💡 **Mesh formats.** `fem/io.py` writes meshes as JSON; `.off`/`.obj` export would make them
  loadable by standard tools.

---

## Suggested Priority Order

1. **Sparse matrices + solver** (§2) — the highest-leverage single change for capability.
2. **The correctness bugs** (§1) — cheap relative to their blast radius, and they clear the deck.
3. **Coverage + type hints** (§3) — deepen the safety net before the bigger numerics work.
4. **Then the numerics roadmap** — quadrature → higher-order elements → time-integrator →
   adaptive refinement.
