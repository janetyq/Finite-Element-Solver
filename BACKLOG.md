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
| Scaling | Iterative solvers + preconditioning — **now the bottleneck** | 🔴 | [§2](#2-performance--scaling) |
| Scaling | Cache assembly across `solve()` calls | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Sparsify the smoothing matrix (topology) | 🟡 | [§2](#2-performance--scaling) |
| Scaling | O(n²) linear scans in meshing | 🟡 | [§2](#2-performance--scaling) |
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

### 🟠 The sparse factorization is the bottleneck
Both earlier limits are gone: matrices are sparse, and assembly is batched (a 3D solve at
n=17 assembles in 0.48s, down from 18.7s). `examples/benchmark_assembly.py` now shows the
cost sitting almost entirely in `splu` -- 2.5s against 0.48s at n=17, and ~13s at n=21,
where fill-in on a 3D tet mesh is what dominates. This is the "iterative solvers +
preconditioning" idea, promoted out of §3: the systems are SPD for Poisson and elasticity,
so CG with a Jacobi or AMG preconditioner should beat a direct factorization well before
n=21. It is the last thing standing between the solver and genuinely large 3D meshes.

### 🟠 `assemble_everything` runs on every `solve()`
`fem/solver.py:Solver.solve` even flags it: `# TODO: don't call this every time`. For
topology optimization `solve()` is called every iteration. Cache `M`, `M_b`, `K` and only
re-assemble when the mesh or material actually changes. (Time-stepping is handled: heat and
wave now build one `DiscreteSystem` and reuse its factorization across steps.)

### 🟠 `calculate_smoothing_matrix` is dense `O(n_elem²)`
`fem/numerics.py:calculate_smoothing_matrix` materializes a full element-by-element
distance matrix. For topology optimization at any real resolution this dominates memory. A
spatial hash / KD-tree (`scipy.spatial.cKDTree.query_ball_point`) building a sparse weight
matrix would scale far better and is a near drop-in.

### 🟠 Meshing generation has `O(n²)` linear scans
`fem/mesh/generation.py` still has linear-scan bottlenecks in vertex deduplication
and neighbour lookups during mesh construction.

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
- 💡 **A posteriori error estimator** so adaptive refinement is fully closed-loop — the
  residual scaffolding is already sketched in `fem/solver.py`. `Solver.adaptive_refinement`
  takes the estimator as a callable `(solver) -> per-element error`, so this drops straight in.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear
  hyperelasticity via the existing `EnergySolver`/`Energies` machinery) all fit the current
  architecture well. `NeohookeanEnergyDensity` is a stub: filling in its `W` and derivatives
  gives a nonlinear material through the already-working Newton solver. Note it is naturally
  written in invariants of `C = FᵀF` rather than in a strain tensor `S`, so it does not slot
  into the St-VK class's `S`-based derivative chain as cleanly as the shared-`W` framing above
  might suggest — it wants its own `evaluate`.
- 💡 **Time-integration abstraction.** Backward-Euler (heat) and Crank–Nicolson (wave) are
  hand-coded inline. A small `TimeIntegrator` interface (θ-method / generalized-α) would
  deduplicate and make it trivial to add new dynamics.
- 💡 **Robin BC path.** `BCType.ROBIN` exists and `resolve` refuses it. A Robin condition adds
  a term to the *system matrix* rather than the load, so the work is in
  `Solver.assemble_everything`, from a boundary stiffness `FunctionSpace` would
  assemble alongside its `boundary_mass_matrix`.
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
  mesh), and adaptive refinement have no *correctness* tests. The 3D tet path now runs to
  h = 1/20 and asserts the same O(h²) band as the 2D case.
- 💡 **The CLI demos have rotted.** Five of fifteen fail, each against an API that
  moved out from under them: `linear_elastic` calls `BoundaryConditions.plot`,
  `topology_optimization` passes `solve(plot=...)`, `energy_solver` reads a
  `'energy'` value `EnergySolver` has never set, `3d` needs an uninstalled
  `imageio`, and `adaptive_refinement` is gated behind a deliberate
  `NotImplementedError`. `rupperts` runs but takes over two minutes. Nothing in
  CI exercises them, and they are the only thing exercising the plot layer.
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
- 💡 **Mesh formats.** `fem/io.py` writes meshes as JSON; `.off`/`.obj` export would make them
  loadable by standard tools.

---

## Suggested Priority Order

1. **Iterative solvers + preconditioning** (§2) — now the top cost, after batched assembly
   moved the last Python loop off the critical path. Unblocks 3D meshes past n≈21.
2. **Coverage + type hints** (§3) — deepen the safety net before the bigger numerics work.
3. **Then the numerics roadmap** — quadrature → higher-order elements → time-integrator →
   adaptive refinement.
