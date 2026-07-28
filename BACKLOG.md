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
| Scaling | Cache assembly across `solve()` calls | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Sparsify the smoothing matrix (topology) | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Rebuild-per-insertion in Ruppert's refinement | 🟡 | [§2](#2-performance--scaling) |
| Numerics | Gaussian quadrature layer (decide `quadrature.py`'s fate) | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Higher-order (quadratic) elements | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | A-posteriori error estimator | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Hand-rolled two-grid preconditioner (drop `pyamg`) | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Globalize Newton (SPD tangents → iterative nonlinear solves) | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Physics | Plane stress as an alternative 2D reduction | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Post-proc | Poisson flux, transient derived fields | 🟢 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Coverage (`pytest-cov`), API docstrings, pre-commit | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

*(No open correctness bugs. Adaptive refinement is now closed-loop except for the error
estimator itself — see [§3](#3-open-ended-suggestions--future-ideas).)*

---

## 2. Performance & Scaling

### 🟠 Every `solve()` re-assembles from scratch
`Solver._steady_problem` builds a fresh `LinearProblem` per call, whose constructor assembles the
operator and load. `TopologyOptimizer` does this once per iteration, where only the material has
changed — the SIMP scaling suggests re-weighting the element matrices per iteration instead of
rebuilding them. (Time-stepping is already handled: the integrators build one `DiscreteSystem`
and reuse its factorization across steps.)

### 🟠 `calculate_smoothing_matrix` is dense `O(n_elem²)`
`fem/numerics.py:calculate_smoothing_matrix` materializes a full element-by-element
distance matrix. For topology optimization at any real resolution this dominates memory. A
spatial hash / KD-tree (`scipy.spatial.cKDTree.query_ball_point`) building a sparse weight
matrix would scale far better and is a near drop-in.

### 🟠 Ruppert's rebuilds the whole triangulation after every insertion
`RuppertsAlgorithm.run_algo` inserts one vertex per pass and then calls `Delaunay(...)`
from scratch, so refinement costs `O(n)` full retriangulations. With the per-segment and
per-triangle scans now vectorised (§ closed below), that rebuild is the dominant remaining
term: ~60% of a run, and it is what keeps the raw 1700-point California outline at ~450 s.
`Delaunay(..., incremental=True)` plus `add_points` is a near drop-in and measures ~6.6x
faster on that component. The catch is that qhull's incremental mode returns the same
triangles in a different `simplices` order, which changes which bad triangle
`run_algo` pops next -- so every mesh the demos and gallery produce would change (still
valid, just different). Worth doing alongside a decision to re-bless the gallery images.

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
- 💡 **A posteriori error estimator** so adaptive refinement is fully closed-loop. The driver
  and its refine/remesh loop are done; only the estimate is missing, and `AdaptiveRefinement`
  takes it as a callable `(solver) -> per-element error`, so it drops straight in. Two flavours
  are still wanted: the *a priori* bound `||e|| <= C h² ||f''||`, which needs only the mesh and
  the source, and the *a posteriori* element residual, which needs the computed solution. They
  are per-equation — the Poisson residual is not the elasticity one — so the natural home is a
  method on the `Equation` subclass rather than a dispatch table in a driver. The seam is
  deliberately *not* pre-declared: an abstract `Equation.error_estimate` with no implementations
  would be the speculative generality `ARCHITECTURE.md` §5 argues against. Write the first
  estimator and the method together.

**Post-processing coverage**

The layer has a rule and an owner per quantity (`ARCHITECTURE.md` §3), but only elasticity
recovers anything. Each item below is an implementation of a seam that already exists.

- 💡 **Poisson flux `-∇u`.** The natural derived field for the scalar family, and the quantity a
  residual error estimator needs. `FunctionSpace.gradient` already computes the gradient; what is
  missing is the packaging — and it needs a **new** result shape, not the existing one.
  `RecoversElasticFields` returns strain, stress, and compliance, none of which a scalar field has,
  so that protocol abstracts over linear-vs-energy *elasticity* rather than over physics. Expect a
  sibling capability with its own bundle, and a `Solution` subclass to carry it; the reusable part
  is the pattern, not the protocol.
- 💡 **Derived fields for transient solves.** `TransientSolution` carries a per-step series of `u`
  and nothing derived from it, so a time-stepped heat problem has no flux history.
- 💡 **Plane stress as an alternative 2D reduction.** 2D elasticity is plane strain throughout, now
  named rather than implicit (`LinearElasticMaterial.out_of_plane_stress`, and the matching
  `out_of_plane_stress` on the energy densities). Plane stress — a thin plate free to contract in
  z, so `sigma_zz = 0` and `eps_zz = -nu/(1-nu) (eps_xx + eps_yy)` — needs a different `D` as well
  as a different out-of-plane component, so it is a second branch in both places plus a way for a
  caller to choose. Worth doing when a thin-plate problem actually appears; a single-member enum
  ahead of that is generality with no second case.
- 💡 **Hand-rolled geometric two-grid V-cycle preconditioner.** The SPD iterative path
  (`fem/backends.py:IterativeBackend`, AMG-CG) currently gets its multigrid from `pyamg`. A
  geometric two-grid V-cycle built on the adaptive-refinement mesh hierarchy would drop in
  behind the same `Backend` seam without touching a caller, removing the dependency and
  being a genuinely instructive build. Full AMG is thousands of lines and not worth
  reimplementing; a two-grid cycle is small and teaches the same ideas. The yardstick to hold is
  `examples/benchmark_assembly.py`: on the 3D elastic benchmark AMG-CG overtakes `splu` at n≈13
  and is ~10× faster by n=21.
- 💡 **Globalize Newton, so the nonlinear path can reach the iterative backend.** `NewtonSolve`
  takes no `Backend` and `EnergySolver` therefore cannot either: CG is SPD-only, and a Newton
  tangent is the Hessian `∇²Π(u)` at the current iterate, which is SPD only where the energy is
  locally strictly convex. The St-Venant–Kirchhoff energy is not convex in `F` — it loses
  ellipticity under compression — so the tangent can be indefinite at the `u = 0` seed, at an
  intermediate iterate, or near a buckling configuration. Large 3D nonlinear solves pay the
  direct factorization's fill-in as a result, on the same curve where AMG-CG wins by ~10×.

  The fix is to make the tangent SPD by construction rather than to let a caller opt in and hope.
  Two standard routes, both fitting behind the existing `SolveStrategy` / `Backend` seams:
  - **Regularized (modified) Newton** — solve `(H + τI) Δu = −r`, raising `τ` until the operator
    is positive definite. Gives a descent direction even at a saddle, and makes CG safe every
    iteration.
  - **Truncated / Steihaug CG** — run CG on the tangent and stop at the first direction of
    negative curvature (`pᵀAp ≤ 0`), using the iterate reached so far. CG deliberately
    repurposed for indefinite systems, normally inside a trust region.

  Either also closes a gap that is already open: `NewtonSolve` takes a full step every iteration
  with no line search or trust region, so convergence currently depends on the seed being close
  enough.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear
  hyperelasticity via the existing `EnergySolver`/`Energies` machinery) all fit the current
  architecture well. `NeohookeanEnergyDensity` is a stub: filling in its `W` and derivatives
  gives a nonlinear material through the already-working Newton solver. Note it is naturally
  written in invariants of `C = FᵀF` rather than in a strain tensor `S`, so it does not slot
  into the St-VK class's `S`-based derivative chain as cleanly as the shared-`W` framing above
  might suggest — it wants its own `evaluate`.
- 💡 **Time-varying loads and Dirichlet data.** Source terms and BC values are functions of
  position only, so a `Problem`'s load is built once and assumed constant in time. Both
  integrators lean on it: `ThetaMethod` reuses one `problem.load` where a general θ-method
  averages `b_n` and `b_{n+1}`, and `NewmarkMethod` reads a fixed Dirichlet displacement as
  zero velocity and acceleration there. The extension is a `t` argument on those callables and
  a load the integrator re-evaluates per step.
- 💡 **Generalized-α, or another integrator family.** `ThetaMethod` and `NewmarkMethod` cover
  first- and second-order systems; the seam for a third is in place, so this is additive.
- 💡 **External work term for `EnergySolver`.** It minimizes the internal elastic energy only
  and builds no load vector, so it currently rejects `Equation.source` outright. Adding the
  external work term `-f · u` (and its gradient/Hessian contributions) would make it accept
  forced problems, which is also a prerequisite for using it on the nonlinear roadmap.

**Engineering**
- 💡 **Coverage.** Add `pytest-cov`, then fill gaps — `svg` and `generation`'s
  `create_approx_mesh` have no *correctness* tests. (Ruppert's is now covered by
  `tests/test_generation.py`, which asserts the angle bound, segment conformity and the
  area cap.) The plot layer is exercised end-to-end by
  `tests/test_demos.py` but has no assertions on what it draws. The 3D tet path now
  runs to h = 1/20 and asserts the same O(h²) band as the 2D case, and the
  `AdaptiveRefinement` driver is covered in `tests/test_refinement.py`.
- 💡 **A flow-around-an-obstacle Poisson demo.** The README's Poisson figure shows one
  (Dirichlet `u = 0` on the obstacle, Neumann inlet/outlet) and nothing in `examples/`
  reproduces it: the meshing side has no support for interior holes, so there is no obstacle
  mesh to solve on. `PSLG` takes a single closed outline plus a bounding box; holes need a
  second loop and an inside/outside rule in Ruppert's. The README now says so rather than
  implying a demo exists.
- 💡 **`adaptive_refinement` is the one demo still skipped by `tests/test_demos.py`**,
  blocked on the error estimator above and on Dirichlet conditions that survive a remesh.
- 💡 **Ruppert's output size is non-monotonic in its input size.** Triangulating the
  California outline from 37 points yields 601 triangles, but from 56 points only 403 — a
  coarser outline can cost *more* work. The likely cause is the interaction between segment
  splitting and which bad triangle gets popped, not the input size as such. Worth
  understanding before tuning the demo's simplification tolerance by feel.
  (Runtime is no longer the issue: the same runs are ~2.8 s and ~1.4 s, and the raw
  1700-point curve now completes in ~450 s where it previously did not finish at all.)
- 💡 **Docstrings on the public API.** Type hints and `pyright` are in place and gating CI;
  the prose half is still open, but narrowly: `mesh/mesh.py`, `mesh/generation.py` and
  `plot/plotter.py` are the modules left with no module docstring. The rest of the core has one.
- 💡 **Tighten pyright to `standard`.** It runs in `basic`, which infers types for the
  unannotated internals rather than demanding annotations. Annotating the internals
  (`refinement`, `generation`, `energies`, `plot`) would let the mode step up.
- 💡 **pre-commit hooks** (ruff + whitespace) so the CI checks run locally before each commit.
- 💡 **Mesh formats.** `fem/io.py` writes meshes as JSON; `.off`/`.obj` export would make them
  loadable by standard tools.
- 💡 **Derivative checks on the assembled energy path.** `fem/numerics.py` has
  `check_gradient`/`check_hessian`, and `StVenantKirchhoff.check_gradients` uses them at the
  energy-density level. The assembled level is unchecked: `EnergySolver.energy` /
  `energy_gradient` / `energy_hessian` should satisfy the same finite-difference agreement.
  The existing helpers plot a convergence curve rather than asserting, so this wants an
  assert-shaped variant (error slope over a window of `eps`) before it can be a test.
- 💡 **Contour overlay for scalar plots.** `fem/plot/helpers.py:plot_colored` draws a flat
  `tripcolor`; a `contour: int` argument adding `tricontour` isolines on top would make
  gradients readable. Was half-written and never validated — needs a look at level selection
  (`np.linspace(min, max, contour)` bunches levels badly on skewed fields).

---

## Suggested Priority Order

1. **Coverage + type hints** (§3) — deepen the safety net before the bigger numerics work.
2. **Then the numerics roadmap** — quadrature → higher-order elements → the error estimator
   that closes the adaptive-refinement loop.
