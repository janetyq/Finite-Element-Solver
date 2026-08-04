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
| Correctness | qhull precision error refining tight outlines (e.g. `cloud.svg`) | 🟡 | [§1](#1-bugs--correctness) |
| Scaling | Cache assembly across `solve()` calls | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Per-insertion `O(n)` left in Ruppert's refinement | 🔴 | [§2](#2-performance--scaling) |
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

### 🔴 Ruppert's refinement hits a qhull precision error on some outlines
`RuppertsAlgorithm.refine()` on `files/cloud.svg` (simplified with `read_svg_to_pslg`,
`tolerance=0.02`) raises `scipy.spatial._qhull.QhullError` ("wide merge ... precision
error") partway through refinement, from `self.triangulation.add_points(added)` in
`_retriangulate`. California refines cleanly on the same settings (326 vertices, 415
insertions, 887 elements), so this is specific to cloud's geometry — its outline has
tighter curvature than California's coastline, which may be pushing qhull's incremental
insertion into near-coplanar points. Not yet isolated to a minimal repro or a specific
insertion; worth a look before using cloud.svg (or any similarly tight shape) in a demo
that runs `RuppertsAlgorithm` to completion.

*(Otherwise no open correctness bugs. Adaptive refinement is closed-loop except for the
error estimator itself — see [§3](#3-open-ended-suggestions--future-ideas).)*

---

## 2. Performance & Scaling

### 🟠 Every `solve()` re-assembles from scratch
`Solver._steady_problem` builds a fresh `LinearProblem` per call, whose constructor assembles the
operator and load. (The two drivers that re-solve in a loop no longer pay this: the integrators
build one `DiscreteSystem` and reuse its factorization across steps, and `TopologyOptimizer`
rescales one set of element matrices and derives its problem with `LinearProblem.with_operator`.)

### 🟠 Ruppert's per-pass cost is down to qhull plus one integer scan
Refinement grows the triangulation incrementally, carries encroachment in a mask, and
refines off a queue of bad triangles topped up per insertion. The California demo has gone
4.5s → 1.5s → 0.84s, and 2098 triangles 6.74s → 2.92s, with the growth exponent down from
2.2 to 1.6. `_retriangulate` — qhull's own insertion — is now the largest single cost.

What is left is `O(n)` per insertion in two cheap places: `(simplices == v).any(axis=1)` to
find the triangles an insertion created, and `_live_triangle_keys` to pack and sort every
triangle so a queued one can be checked to still exist. Both are integer work, so the
constant is small, but the loop is still superlinear. Removing them needs the cavity from
qhull (`add_points` does not report it) or a hand-rolled Bowyer–Watson.

Two things measured and rejected, so they do not get proposed again:
- **Testing enclosure per candidate instead of labelling regions**, over the whole mesh:
  1.9x *slower*. A non-convex outline fails the angle bound on hundreds of triangles
  outside the hull, and there are only a handful of regions. (Per *newly created* triangle
  it is the right trade, and that is what `_bad_triangles_created_by` does.)
- **`find_simplex` for the staleness check**: 14us in a tight loop but 14ms when
  interleaved with `add_points`, which rebuilds its search structure each time. That made
  the queue 8x slower than the rescan it replaced.

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
- 💡 **Coverage.** Add `pytest-cov`, then fill gaps — `svg`'s path parsing is covered only
  through the demos.
  (Ruppert's is covered by `tests/test_ruppert.py`: the angle bound, segment conformity,
  the area cap, the even-odd fill rule and boundary attribution.) The plot layer is
  exercised end-to-end by
  `tests/test_demos.py` but has no assertions on what it draws. The 3D tet path now
  runs to h = 1/20 and asserts the same O(h²) band as the 2D case, and the
  `AdaptiveRefinement` driver is covered in `tests/test_refinement.py`.
- 💡 **A flow-around-an-obstacle Poisson demo.** The meshing side is done: the
  `stress_concentration` demo builds this mesh from `domains.plate_with_hole_pslg`, and
  `RuppertsAlgorithm.boundary_loops` says which
  outline each boundary facet came from, so the obstacle rim and the outer wall can take
  different conditions. What is left is the solve — inlet and outlet are *parts* of the
  outer loop rather than loops of their own, so they still need a coordinate region
  (`fem.regions.on_plane`) to separate them, and the demo has to wire that to the Poisson
  equation and reproduce the README figure.
- 💡 **The `refinement` demo shows both ends of the loop and not the join** — the peaked
  Poisson problem that motivates refining, and red-green splitting on a small mesh —
  because driving one from the other needs the error estimator above, plus Dirichlet
  conditions that survive a remesh. Every demo now runs under `tests/test_demos.py`,
  with no skips.
- 💡 **Report the minimum corner angle alongside the demo's simplification tolerance.**
  Output size used to be non-monotonic in *input* size, because cost tracked the sharpest
  corner Douglas-Peucker left behind rather than the point count. The corner treatment
  (shell splitting plus the exemption, `RuppertsAlgorithm._split_point` /
  `_spans_a_sharp_corner`) fixed the runaway, and the sweep is now monotonic:

  | tolerance | points | min corner | triangles |
  |---|---|---|---|
  | 0.04 | 118 | 31.9° | 612 |
  | 0.02 | 173 | 25.4° | 875 |
  | 0.01 | 242 | **20.5°** | 1068 |
  | 0.005 | 326 | 25.4° | 1565 |

  (All twelve contours, `min_angle=30`, no area cap. Without the corner treatment the last
  three do not terminate at all.) The minimum corner angle is still the number that
  predicts whether the requested bound will hold everywhere, so a sweep should report it.
  Refinement order is *not* a factor — refining the worst triangle first instead of qhull's
  arbitrary last was measured and is a wash.
- 💡 **Docstrings on the public API.** Type hints and `pyright` are in place and gating CI;
  the prose half is still open, but narrowly: `mesh/mesh.py`, `mesh/ruppert.py`,
  `mesh/svg.py` and `plot/plotter.py` are the modules left with no module docstring. The
  rest of the core has one.
- 💡 **Tighten pyright to `standard`.** It runs in `basic`, which infers types for the
  unannotated internals rather than demanding annotations. Annotating the internals
  (`refinement`, `ruppert`, `energies`, `plot`) would let the mode step up.
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
