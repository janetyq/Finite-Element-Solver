# Backlog: Finite Element Solver

The list of still-open work on the solver: correctness, performance, architecture, and open-ended
ideas.

Legend: 🔴 bug / correctness · 🟠 performance / scaling · 🟡 design / maintainability · 💡 idea.
Effort: 🟢 low · 🟡 medium · 🔴 high.

## At a glance

| Area | Item | Effort | Detail |
|---|---|:---:|---|
| Scaling | Cache assembly across `solve()` calls | 🟡 | [§2](#2-performance--scaling) |
| Scaling | Per-insertion `O(n)` left in Ruppert's refinement | 🔴 | [§2](#2-performance--scaling) |
| Numerics | 3D P2 (`QuadraticTetrahedralElement`); P2 curved-element adaptivity | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Mixed (u-p) formulation for near-incompressible elasticity | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Hand-rolled two-grid preconditioner (drop `pyamg`) | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Globalize the Newton direction (SPD tangents → iterative nonlinear solves) | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Physics | Plane stress as an alternative 2D reduction | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Design | Lazy `pyamg` import | 🟢 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Coverage (`pytest-cov`), API docstrings, pre-commit | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Demos | Stress-driven design beside compliance-driven; a motivated goal-oriented refinement demo | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

No open correctness bugs.

---

## 2. Performance & Scaling

### 🟠 Every `solve()` re-assembles from scratch
`Equation.problem` builds a fresh `LinearProblem` per call, whose constructor assembles the
operator and load. The two looping drivers already avoid this: the integrators build one
`DiscreteSystem` and reuse its factorization across steps, and `SIMPModel` rescales one set
of element matrices via `LinearProblem.with_operator`. The remaining case is a repeated steady
`solve()`.

### 🟠 Per-insertion `O(n)` left in Ruppert's refinement
Refinement grows the triangulation incrementally, carries encroachment in a mask, and refines off a
queue of bad triangles topped up per insertion. `_retriangulate` (qhull's own insertion) is the
largest single cost.

What is left is `O(n)` per insertion in two cheap places: `(simplices == v).any(axis=1)` to find
the triangles an insertion created, and `_live_triangle_keys` to pack and sort every triangle so a
queued one can be checked to still exist. Both are integer work, so the constant is small, but the
loop is still superlinear. Removing them needs the cavity from qhull (`add_points` does not report
it) or a hand-rolled Bowyer-Watson.

Two approaches measured and rejected, so they are not proposed again:
- **Testing enclosure per candidate instead of labelling regions**, over the whole mesh: 1.9x
  *slower*. A non-convex outline fails the angle bound on hundreds of triangles outside the hull,
  and there are only a handful of regions. Per *newly created* triangle it is the right trade,
  which is what `_bad_triangles_created_by` does.
- **`find_simplex` for the staleness check**: 14us in a tight loop but 14ms when interleaved with
  `add_points`, which rebuilds its search structure each time. That made the queue 8x slower than
  the rescan it replaced.

---

## 3. Open-Ended Suggestions & Future Ideas

**Numerics**
- 💡 **Finish P2: 3D, deformed geometry, adaptivity.** The 2D P2 path shipped; three pieces are
  still open. **3D P2** wants a `QuadraticTetrahedralElement` (ten nodes) and the edge/face numbering
  to match. The `Element` base and the `FunctionSpace` node set generalize, but the 3D shape functions
  and connectivity are not written. **`Mesh.displaced`** (behind `deformed_mesh` and `mode_mesh`) reads the vertex DOFs and drops the
  edge-midpoint displacements, so a P2 displacement warp draws as its P1 restriction (field plotting
  itself is P2-aware via `fem.plot.tessellation`). **Adaptive refinement** drives a P2 solve
  through either estimator: the recovery one samples the flux per quadrature point and recovers by L2
  projection, and the residual one carries the interior `div(flux)` (from the P2 shape Hessians) and a
  per-side edge jump. Both are for straight P2; a *curved* (isoparametric) element's varying Jacobian
  adds a first-derivative term the residual estimator's affine field Hessian omits, so refining a
  curved-boundary P2 solve is the remaining piece.
- 💡 **Curved (isoparametric) elements: follow-ups.** The core shipped:
  `IsoparametricTriangleElement` (a geometry map differentiated over all nodes, per quadrature
  point), `Circle` / `Arc` curves carried through `PSLG` -> `RuppertsAlgorithm` ->
  `Mesh.boundary_curves`, boundary-node projection in `p2_connectivity`, curvature-aware Ruppert and
  red-green refinement, a curved `MassForm`, P2-aware plotting (`fem.plot.tessellation` through
  `Plotter.plot(solution, ...)`), SVG cubic Beziers
  retained as `CubicBezier` curves through `read_svg_to_pslg` (adaptive flatness sampling, tag-aware
  Douglas-Peucker), and validation (`tests/test_convergence_curved.py` area fidelity and the P2 rate;
  `tests/test_curved_meshing.py` the pipeline and the Kirsch stress concentration; `tests/test_svg.py`
  the traced-outline round trip). Two follow-ups are left. **3D curved elements** and **`fem/post/io.py`
  curve serialization** (a saved mesh currently drops its curves) are the remaining gaps. `files/cloud.svg`
  now meshes and solves in the `outline_to_mesh` demo, so its Bezier boundary carries through the pipeline
  there; a dedicated *close-up* contrasting the curved boundary against its chord polygon (the isoparametric
  payoff) is unbuilt. Quadratic Beziers (degree-elevate to cubic) and
  elliptical arcs (`EllipseArc`) are unbuilt but unused by the bundled assets.
- 💡 **Mixed (u-p) formulation, to remove volumetric locking near nu -> 0.5.** The linear triangle
  has one constant strain per element, which cannot represent deviatoric and volumetric deformation
  independently. As `nu` approaches incompressibility the element gets artificially stiff, worst in
  curved or high-gradient regions (found via the `stress_concentration` demo: a nu-sweep at the hole
  rim showed real growth from `nu`=0 to 0.3, then a sharp additional rise from 0.3 to 0.499, the
  classic locking signature, while a uniform-stress patch far from any curvature stayed essentially
  flat; see the git history around the roller-boundary-conditions branch for the numbers). P2
  displacement *alleviates* this but does not cure it; the standard cure for this element family is
  a mixed formulation: interpolate pressure (mean stress) as its own field, one polynomial degree
  below displacement (Taylor-Hood P2-P1 is the standard pairing, and P2 now exists as one half of
  it), and enforce the volumetric constraint weakly instead of purely through displacement. This is
  more than an element swap: the assembled system becomes a saddle-point (indefinite) one, not the
  SPD system every solve path here currently assumes (`fem/algebra/backends.py`'s CG path and its
  rigid-body near-null-space handling, in particular), so it needs its own solver strategy alongside
  the new element. Now that P2 elements exist, selective/reduced integration on the volumetric term
  is also available as a lighter-weight partial answer the constant-strain triangle could not offer.
- 💡 **Hand-rolled geometric two-grid V-cycle preconditioner.** The SPD iterative path
  (`fem/algebra/backends.py:IterativeBackend`, AMG-CG) currently gets its multigrid from `pyamg`. A
  geometric two-grid V-cycle built on the adaptive-refinement mesh hierarchy would drop in behind
  the same `Backend` seam without touching a caller, removing the dependency and being a genuinely
  instructive build. Full AMG is thousands of lines and not worth reimplementing; a two-grid cycle
  is small and teaches the same ideas. The yardstick to hold is the `timing_benchmark` demo (`examples/demos/timing_benchmark/`):
  on the 3D elastic benchmark AMG-CG overtakes `splu` at n≈13 and is ~10× faster by n=21.
- 💡 **Globalize the Newton direction, so the nonlinear path can reach the CG backend.**
  CG is SPD-only, and a Newton tangent is the Hessian `∇²Π(u)` at the current iterate, which is
  SPD only where the energy is locally strictly convex. The St-Venant-Kirchhoff energy is not
  convex in `F` (it loses ellipticity under compression), so the tangent can be indefinite at the
  `u = 0` seed, at an intermediate iterate, or near a buckling configuration. Large 3D nonlinear
  solves pay MINRES or the direct factorization's fill-in as a result, on the same curve where
  AMG-CG wins by ~10×.

  Step *length* is globalized: `NewtonSolve` takes an optional `BacktrackingLineSearch` (Armijo on
  Π, else ½‖r‖²), which `default_strategy` uses, so a full step does not diverge from a poor seed.
  What remains is the step *direction*: at an indefinite tangent the line search has no descent
  direction to scale and falls back to the full step. The fix is to make the tangent SPD by
  construction, both routes fitting behind the existing `SolveStrategy` / `Backend` seams:
  - **Regularized (modified) Newton** solves `(H + τI) Δu = −r`, raising `τ` until the operator is
    positive definite. Gives a descent direction even at a saddle, and makes CG safe every iteration.
  - **Truncated / Steihaug CG** runs CG on the tangent and stops at the first direction of negative
    curvature (`pᵀAp ≤ 0`), using the iterate reached so far. CG deliberately repurposed for
    indefinite systems, normally inside a trust region.
- 💡 **Nonlinear post-buckling, the sequel to `BucklingAnalysis`.** Linearised buckling finds the
  critical load and the mode shape (`fem/analysis/buckling.py`), but not what the structure *does* past the
  bifurcation: the load-deflection path once it has bowed. That is a geometrically nonlinear
  (St-Venant-Kirchhoff) solve seeded with a small imperfection in the buckling mode, and it needs
  exactly the globalized Newton above *plus* arc-length (Riks) control, since the tangent goes
  indefinite and the load-displacement curve turns back on itself at the limit point, where
  load-controlled and displacement-controlled Newton both stall. The pieces line up (a `Problem`
  over the St-VK `EnergyForm`, whose `internal_residual` and `load` an arc-length strategy scales
  against each other; the buckling mode for the imperfection shape; a globalized tangent for the
  indefinite region), so this is additive once arc-length joins the `SolveStrategy` family.

**Post-processing coverage**

The layer has a rule and an owner per quantity (`ARCHITECTURE.md` §3). Steady solves now recover
their derived fields through one seam: `Form.derived_field` names the field (Poisson's gradient,
elasticity's stress, `fem.physics.derived.DerivedField`), the typed `Solution` carries it per element
(`ScalarFieldSolution.flux`, `ElasticSolution.stress`), and `fem.post.recovery.recover_nodal` turns it into
a continuous per-node field for smooth output, P2 plotting, and the recovery estimator; a
`TransientSolution` packages any step the same way through `at(i)`.

- 💡 **Plane stress as an alternative 2D reduction.** 2D elasticity is plane strain throughout, now
  named rather than implicit (`LinearElasticMaterial.out_of_plane_stress`, and the matching
  `out_of_plane_stress` on the energy densities). Plane stress (a thin plate free to contract in z,
  so `sigma_zz = 0` and `eps_zz = -nu/(1-nu) (eps_xx + eps_yy)`) needs a different `D` as well as a
  different out-of-plane component, so it is a second branch in both places plus a way for a caller
  to choose. Worth doing when a thin-plate problem actually appears; a single-member enum ahead of
  that is generality with no second case.

**Design / maintainability**

- 💡 **Lazy `pyamg` import.** `fem.algebra.backends` imports `pyamg` at module scope, so
  `import fem` always pulls it in (the plot layer is already lazy: matplotlib loads on the first
  `Plotter`). Making it lazy inside `IterativeBackend.prepare` would let the package import
  without `pyamg`. Worth doing only if a headless import becomes a goal.

**Features**
- 💡 **Adjoint sensitivity: follow-ups.** The core shipped (`fem/analysis/sensitivity.py`:
  `SensitivityAnalysis`, `Compliance` / `PointValue` quantities of interest, `DensityField` /
  `ModulusField` parameterizations) and the `DesignOptimizer` over it (`fem/analysis/design.py`, SIMP density
  design with the compliance sensitivity from the core). Design record in
  `attic/fem-adjoint-sensitivity-design-2026-08-18.md`; the follow-up plan
  is `attic/fem-adjoint-followups-2026-08-19.md`. **Stress-based quantities of interest** shipped
  (`MeanStress`, `SoftMaxStress` in `fem/analysis/sensitivity.py`): they supply the adjoint load `∂J/∂u` for a
  fixed material, validated by finite differences. The remaining piece for stress-*constrained design*
  is the explicit `∂J/∂p` term, since the stress `σ = D(E)Bu` depends on the design modulus directly,
  not only through `u`; the adjoint pass adds only `−λᵀ∂R/∂p`, so the driver needs an optional
  `∂J/∂p` from the quantity of interest (and the relaxed-stress `σ = ρ^η D0 Bu` definition topology
  optimization uses). Best done alongside the general gradient engine below. Open pieces, each additive
  behind the same three seams:
  **shape parameterization** (`∂(element geometry)/∂(node)` mesh sensitivities, the one piece needing
  new geometry-derivative code); a **general gradient engine** (`scipy.optimize` SLSQP behind the
  optimizer, for objectives the optimality-criteria update cannot take); and the **nonlinear tangent
  path** (a `Problem` with a state-dependent tangent, where the adjoint uses `tangent(u)` at the
  converged state).
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics) fits the current
  architecture well, on the same `EnergyForm` / `Energies` machinery that now carries both the
  strain-measure densities and the invariant-based `NeohookeanEnergyDensity`.
- 💡 **Prescribed motion under Newmark.** A `TimeDependent` source, traction, or Robin value is
  re-evaluated per step by both integrators, and `ThetaMethod` takes a time-dependent Dirichlet
  value too. `NewmarkMethod` refuses one: a prescribed displacement `g(t)` at the fixed DOFs also
  needs their velocity and acceleration, so the acceleration solve would carry `g''(t)` at the fixed
  block (by differencing `g`, or from a `TimeDependent` that supplies its derivatives) and the
  predictor `g'(t)`. Additive once that data has a home.
- 💡 **Generalized-α, or another integrator family.** `ThetaMethod` and `NewmarkMethod` cover first-
  and second-order systems; the seam for a third is in place, so this is additive.

**Engineering**
- 💡 **Coverage.** Add `pytest-cov`, then fill gaps: `svg`'s path parsing is covered only through the
  demos, and the plot layer is exercised end-to-end by `tests/test_demos.py` but has no assertions on
  what it draws.
- 💡 **Report the minimum corner angle alongside the demo's simplification tolerance.** The minimum
  corner angle is the number that predicts whether the requested angle bound will hold everywhere
  (the corner treatment in `RuppertsAlgorithm._split_point` / `_spans_a_sharp_corner` keeps output
  size monotonic in input size), so a tolerance sweep should report it.
- 💡 **Docstrings on the public API.** Type hints and `pyright` are in place and gating CI; the prose
  half is still open, but narrowly: `mesh/mesh.py`, `mesh/ruppert.py`, `mesh/svg.py` and
  `plot/plotter.py` are the modules left with no module docstring. The rest of the core has one.
- 💡 **Tighten pyright to `standard`.** It runs in `basic`, which infers types for the unannotated
  internals rather than demanding annotations. Annotating the internals (`refinement`, `ruppert`,
  `energies`, `plot`) would let the mode step up.
- 💡 **pre-commit hooks** (ruff + whitespace) so the CI checks run locally before each commit.
- 💡 **Mesh formats.** `fem/post/io.py` writes meshes as JSON; `.off` / `.obj` export would make them
  loadable by standard tools.

---

### Demos

- 💡 **Stress-driven design beside compliance-driven.** `DesignOptimizer` takes any
  `QuantityOfInterest` and `SoftMaxStress` exists, so the same beam optimized once for stiffness and
  once for peak stress, side by side, would show why the two are not the same design. The OC update
  assumes a monotone sensitivity, which a stress objective does not guarantee, so this may need the
  `scipy.optimize` engine above before it converges cleanly.
- 💡 **A motivated goal-oriented refinement demo.** The estimator is built and tested but has no
  gallery demo; a point value of Poisson on a square did not make the case. It needs a problem whose
  goal is sensitive somewhere other than where the solution is rough, such as a cantilever with a
  hole in its web and the tip deflection as the goal: the global estimator spends its budget on the
  hole's stress concentration, the goal-oriented one on the root and the load path.

## Suggested Priority Order

1. **Coverage + type hints** (§3): deepen the safety net.
2. **Finish the P2 story**: 3D P2, then P2-aware plotting and adaptivity (§3), so the higher-order
   path is complete rather than 2D-only.
3. **Then the harder numerics**: mixed u-p for incompressibility (P2 is now in place as its
   displacement half), or the hand-rolled two-grid preconditioner.
