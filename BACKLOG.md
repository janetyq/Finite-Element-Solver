# Backlog: Finite Element Solver

The list of still-open work on the solver: correctness, performance, architecture, and open-ended
ideas.

Legend: 🔴 bug / correctness · 🟠 performance / scaling · 🟡 design / maintainability · 💡 idea.
Effort: 🟢 low · 🟡 medium · 🔴 high.

## At a glance

| Area | Item | Effort | Detail |
|---|---|:---:|---|
| Numerics | Nonlinear transient; flow-theory J2 plasticity; advection-diffusion | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Finish P2: 3D element, curved-element adaptivity, 3D residual estimator | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Mixed (u-p) formulation, then Stokes; hand-rolled two-grid preconditioner | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Globalize the Newton direction (Steihaug CG, the unbuilt route) | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Physics | Harmonic response; arc-length post-buckling; thermoelastic follow-ups | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| API | Reaction forces, region objects, driver histories, series conveniences | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Structure | Split `forms.py` with the stress-QoI dedup | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Coverage/plot gaps, docstrings, ruff/pyright, CI matrix, API site, release | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Demos | Stress-driven design; goal-oriented, L-shape, benchmark, 3D structural | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

No open correctness bugs. A handful of small robustness guards remain, each 🟢:

- 💡 **Small guards.** Pin index math to `int64` (`Mesh` builds `elements` with `dtype=int`); the
  residual estimator's `searchsorted` edge lookup (`fem/analysis/estimators.py`) does not verify the
  hit; a zero-length edge gives NaN angles that pass any min-angle bound; the "splitting only
  subdivides" comment in Ruppert's `_bad_among` is false on curved input; shell split points beside a
  curved piece are projected onto the curve, which can break the power-of-two ladder where a Bezier
  meets a line at a sharp angle; the OC bisection bracket `[0, 1e15]` is unchecked; `Arc` accepts
  spans over 2π; `_buckling_factors` can silently return fewer modes than asked; `_in_circle` in
  `delaunay.py` has no Shewchuk-style error filter (`_orient` does).

---

## 2. Performance & Scaling

### Ruppert's refinement: approaches measured and rejected
Not open work; recorded so they are not proposed again. Refinement now inserts through
`IncrementalDelaunay`, so an insertion costs its cavity.
- **Testing enclosure per candidate instead of labelling regions**, over the whole mesh: 1.9x
  *slower*. A non-convex outline fails the angle bound on hundreds of triangles outside the hull,
  and there are only a handful of regions. Per *newly created* triangle it is the right trade,
  which is what `_bad_among` does.
- **scipy's incremental `Delaunay`** (`add_points`) for the growing triangulation: it rebuilds
  its simplex arrays after every insertion, ~10 ms each at a few thousand points, so a run was
  quadratic and that one call was over 80% of it; and its point location (`find_simplex`)
  rebuilds a search structure per call, 1000x slower interleaved with insertions than alone.

---

## 3. Open-Ended Suggestions & Future Ideas

**Numerics**
- 💡 **Finish P2: 3D, deformed geometry, adaptivity.** The 2D P2 path shipped; several pieces are
  still open. **3D P2** wants a `QuadraticTetrahedralElement` (ten nodes) and the edge/face numbering
  to match. The `Element` base and the `FunctionSpace` node set generalize, but the 3D shape functions
  and connectivity are not written. **`Mesh.displaced`** (behind `NodalField.deformed_mesh`) reads the vertex DOFs and drops the
  edge-midpoint displacements, so a P2 displacement warp draws as its P1 restriction (field plotting
  itself is P2-aware via `fem.plot.tessellation`). **Adaptive refinement** drives a P2 solve
  through either estimator: the recovery one samples the flux per quadrature point and recovers by L2
  projection, and the residual one carries the interior `div(flux)` (from the P2 shape Hessians) and a
  per-side edge jump. Both are for straight P2; a *curved* (isoparametric) element's varying Jacobian
  adds a first-derivative term the residual estimator's affine field Hessian omits, so refining a
  curved-boundary P2 solve is the remaining piece. The residual estimator is 2D only
  (`fem/analysis/estimators.py`), so 3D adaptivity has only the recovery and goal-oriented estimators;
  a **3D residual estimator** needs a codim-1 facet-adjacency table on `Mesh` (the edge table exists).
  **Superconvergent (ZZ) patch recovery** is a second recovery behind `recover_nodal`'s seam: a local
  least-squares fit per node patch at the superconvergent points, which is the standard recovery for
  the estimator and raises its effectivity, especially on P2.
- 💡 **Curved (isoparametric) elements: follow-ups.** The core shipped:
  `IsoparametricTriangleElement` (a geometry map differentiated over all nodes, per quadrature
  point), `Circle` / `Arc` pieces of an `Outline` carried through its sampled `PSLG` ->
  `RuppertsAlgorithm` -> `Mesh.boundary_curves`, boundary-node projection in `p2_connectivity`, curvature-aware Ruppert and
  red-green refinement, a curved `MassForm`, P2-aware plotting (`fem.plot.tessellation` through
  `Plotter.plot(solution, ...)`), SVG cubic Beziers
  read as `CubicBezier` pieces by `Outline.from_svg` (sampled only at mesh time; Douglas-Peucker
  touches only the straight runs), and validation (`tests/test_convergence_curved.py` area fidelity and the P2 rate;
  `tests/test_curved_meshing.py` the pipeline and the Kirsch stress concentration; `tests/test_svg.py`
  the traced-outline round trip). Two follow-ups are left. **3D curved elements** and **`fem/post/io.py`
  curve serialization** (a saved mesh currently drops its curves; the pieces need a `to_dict` /
  `from_dict` pair) are the remaining gaps. `files/cloud.svg`
  now meshes and solves in the `outline_to_mesh` demo, so its Bezier boundary carries through the pipeline
  there; a dedicated *close-up* contrasting the curved boundary against its chord polygon (the isoparametric
  payoff) is unbuilt. Quadratic Beziers (degree-elevate to cubic) and
  elliptical arcs (`EllipseArc`) are unbuilt but unused by the bundled assets.
- 💡 **Nonlinear transient.** `ThetaMethod` and `NewmarkMethod` take `problem.tangent(None)` and
  refuse an `EnergyForm`. The per-step problem is `M/(β dt²) + K_T(u)`, one Newton solve per step
  seeded from the predictor; the mass shift keeps the tangent SPD for small `dt`. It is what the
  post-buckling snap-through demo needs, and the home of any time-dependent operator (`κ(t)`,
  `E(t)`). Under a hundred lines plus four tests (linear parity, small-load parity with small strain,
  energy conservation, rigid rotation gives no stress).
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
  more than an element swap. `MinresBackend` already solves symmetric indefinite systems, so the
  saddle-point solve is reachable; what is missing is the mixed function space (a `MixedSpace` over
  ordered sub-spaces with block DOF offsets), the coupling block `B = ∫ q ∇·u`, a pressure-nullspace
  pin, and a block / Schur preconditioner for MINRES. Stokes is the same machinery with a different
  `A`, the natural second customer. The four-PR staging in `attic/fem-non-spd-solver-2026-08-18.md`
  §4 and §7 still holds. A lighter-weight stepping stone that needs no mixed space:
  **selective / reduced integration (B-bar)** on the volumetric term, a split quadrature rule, plus a
  warning above `nu ≈ 0.49`.
- 💡 **Hand-rolled geometric two-grid V-cycle preconditioner.** The SPD iterative path
  (`fem/algebra/backends.py:IterativeBackend`, AMG-CG) currently gets its multigrid from `pyamg`. A
  geometric two-grid V-cycle built on the adaptive-refinement mesh hierarchy would drop in behind
  the same `Backend` seam without touching a caller, removing the dependency and being a genuinely
  instructive build. Full AMG is thousands of lines and not worth reimplementing; a two-grid cycle
  is small and teaches the same ideas. The yardstick to hold is the `timing_benchmark` demo (`examples/demos/timing_benchmark/`):
  on the 3D elastic benchmark AMG-CG overtakes `splu` at n≈13 and is ~10× faster by n=21. The same
  iterative gap shows in `EigenSolve` (shift-invert ARPACK on a direct factorization, paying the 3D
  fill-in): `scipy.sparse.linalg.lobpcg` with the AMG preconditioner already built for CG is the
  matching iterative path for modal and buckling at scale.
- 💡 **Globalize the Newton direction, so the nonlinear path can reach the CG backend.**
  CG is SPD-only, and a Newton tangent is the Hessian `∇²Π(u)` at the current iterate, which is
  SPD only where the energy is locally strictly convex. The St-Venant-Kirchhoff energy is not
  convex in `F` (it loses ellipticity under compression), so the tangent can be indefinite at the
  `u = 0` seed, at an intermediate iterate, or near a buckling configuration. Large 3D nonlinear
  solves pay MINRES or the direct factorization's fill-in as a result, on the same curve where
  AMG-CG wins by ~10×.

  Step *length* is globalized: `NewtonSolve` takes an optional `BacktrackingLineSearch` (Armijo on
  Π, else ½‖r‖²), which `default_strategy` uses, so a full step does not diverge from a poor seed.
  The step *direction* is half-done. **Regularized (modified) Newton** shipped
  (`TangentRegularization`, applied by `NewtonSolve.regularization='auto'` under an iterative
  backend): it solves `(H + τI) Δu = −r`, raising `τ` until the operator is positive definite,
  giving a descent direction even at a saddle. The caveat is that a shift alone does not make the
  SPD-only CG backend reliable on an indefinite tangent, because CG's failure is not always
  signalled. The unbuilt route is **truncated / Steihaug CG**: run CG on the tangent and stop at the
  first direction of negative curvature (`pᵀAp ≤ 0`), using the iterate reached so far. CG
  deliberately repurposed for indefinite systems, normally inside a trust region; it is the only way
  to make the nonlinear path reach AMG-CG safely.
- 💡 **Nonlinear post-buckling, the sequel to `BucklingAnalysis`.** Linearised buckling finds the
  critical load and the mode shape (`fem/analysis/buckling.py`), but not what the structure *does* past the
  bifurcation: the load-deflection path once it has bowed. That is a geometrically nonlinear
  (St-Venant-Kirchhoff) solve seeded with a small imperfection in the buckling mode, and it needs
  exactly the globalized Newton above *plus* arc-length (Riks) control, since the tangent goes
  indefinite and the load-displacement curve turns back on itself at the limit point, where
  load-controlled and displacement-controlled Newton both stall. The pieces line up (a `Problem`
  over the St-VK `EnergyForm`, whose `internal_residual` and `load` an arc-length strategy scales
  against each other; the buckling mode for the imperfection shape; a globalized tangent for the
  indefinite region), and the load-stepping half now exists (`QuasiStaticStepping`,
  `fem/algebra/stepping.py`: the warm-started walk, the bisection retry, the history solution);
  what remains is arc-length control of that loop, so it can turn past the limit point where
  load control stalls. The design (bordered two-solve corrector, cylindrical Crisfield constraint,
  imperfection from the buckling mode, adaptive step, a `PathSolution`) is
  `attic/thermoelasticity-and-buckling-path-plans-2026-08-23.md` Plan 2.
- 💡 **Harmonic response.** `(K − ω²M)u = f`, one indefinite solve `DirectBackend` handles; an
  analysis beside `ModalAnalysis`, the last cheap `EigenSolve` sibling.
- 💡 **Flow-theory J2 plasticity.** Return mapping per quadrature point, with a `PlasticState` /
  `at_state` / `commit` state-carrying interface. The eigenstrain seam already carries a plastic
  strain as its elastic-predictor half.
- 💡 **Advection-diffusion, and a nonsymmetric backend.** `−div(κ∇u) + b·∇u = f` is a new
  non-symmetric `Form`; a direct backend serves it first, a `GmresBackend` (and SUPG stabilization
  for advection-dominated regimes) after. The README roadmap names it.
- 💡 **1D solves.** `box_mesh` builds 1D meshes but `FunctionSpace` refuses them
  (`LinearLineElement.SUB_TYPE = None`, TODO at `fem/elements.py`). The cheapest convergence check.

**Post-processing coverage**

The layer has a rule and an owner per quantity (`ARCHITECTURE.md` §3). Steady solves now recover
their derived fields through one seam: `Form.flux` names the field (Poisson's gradient,
elasticity's stress, `fem.physics.derived.Flux`), the typed `Solution` carries it per element
(`DiffusionSolution.gradient`, `ElasticSolution.stress`), and `fem.post.recovery.recover_nodal` turns it into
a continuous per-node field for smooth output, P2 plotting, and the recovery estimator; a
`TransientSolution` packages any step the same way through `history[i]`.

- 💡 **Reaction forces.** `(K u − f)[fixed]` as `Problem.reactions(u)` or `ElasticSolution.reactions`;
  the standard hand-calculation check, near-free over machinery already here.
- 💡 **Series conveniences.** `wave_energy` is a free function in `stepping.py` and a series cannot be
  resumed or thinned: `WaveSolution.energy(i)`, plus integrator `t0` and `store_every`. And a
  `TransientSolution.trace(point, component)` (λ against a chosen point component) serves the stepping
  demo, the integrators, and arc-length alike, removing the per-step loop the demos write by hand.
- 💡 **Plane stress for the finite-strain path.** `LinearElastic` takes `reduction='plane_stress'`;
  the energy densities (`FiniteStrainElastic`, `DeformationPlasticity`) are plane strain only,
  their `out_of_plane_stress` being the one place the assumption sits. Plane stress there means
  solving `S_zz = 0` for the out-of-plane stretch per quadrature point, a small nonlinear
  condensation inside `evaluate`. Worth doing when a thin-plate finite-strain problem appears.

**Design / maintainability**

- 💡 **Region objects.** `regions.py` is still callables plus a hand-propagated `mesh_bound` flag,
  and `on_tag` refuses `&` / `|`. A `Region` base with the boolean operators, `on_tag` as the one
  facet-resolved member, and a resolution cache keyed by the spec.
- 💡 **One loop shape for the drivers.** `AdaptiveRefinement.run()` returns only the final solution
  and keeps `mesh` / `solution` as mutable attributes, while `DesignOptimizer.run` returns a
  `DesignHistory`. A `RefinementHistory` and an `on_round` hook, and frozen drivers with a `steps()`
  generator, so the demos stop unrolling the loop by hand.
- 💡 **Named outline loops.** `on_tag(k)` names a boundary loop by its integer index in the
  `Outline`; an optional name on the loop would let a condition read `on_tag('hole')` while tags stay
  integers underneath. The bracket and stress-concentration demos are the readers.
- 💡 **`refine_fraction` naming.** It means "within this fraction of the peak", the opposite of a
  Dörfler bulk fraction. Rename, or add Dörfler marking.
- 💡 **Split `forms.py`.** 1126 lines mixing kinematics / Voigt helpers, combinators, operators, and
  elastic-state recovery; `LinearElasticForm` and `EnergyForm` each re-derive the plane-strain lift
  in `sample` / `recover`. Extract a `kinematics.py` and a shared recovery helper. In the same pass,
  **dedup the stress QoI**: `_VonMisesStress` carries its own `D B` and von Mises derivative while the
  form and `invariants` own both, and `MeanStress` / `SoftMaxStress` reach into its privates and are
  near-identical. Read the stress off `problem.operator` through one `stress_jacobian` hook.
- 💡 **Lazy `pyamg` import.** `fem.algebra.backends` imports `pyamg` at module scope, so
  `import fem` always pulls it in (the plot layer is already lazy: matplotlib loads on the first
  `Plotter`). Making it lazy inside `IterativeBackend.prepare` would let the package import
  without `pyamg`. Worth doing only if a headless import becomes a goal.

**Features**
- 💡 **Adjoint sensitivity: follow-ups.** The core shipped (`fem/analysis/sensitivity.py`:
  `SensitivityAnalysis`, `Compliance` / `PointValue` quantities of interest, `DensityParameterization` /
  `ModulusParameterization`) and the `DesignOptimizer` over it (`fem/analysis/design.py`, SIMP density
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
  converged state). An **inverse-problem demo** would exercise the whole chain: `ModulusParameterization`
  has no demo, and a least-squares QoI recovering a planted modulus field is mostly a demo and a
  strong end-to-end adjoint test.
- 💡 The README's roadmap (transport, fluid mechanics) fits the current
  architecture well, on the same `EnergyForm` / `Energies` machinery that now carries both the
  strain-measure densities and the invariant-based `NeohookeanEnergyDensity`.
- 💡 **Thermoelasticity: follow-ups.** The core shipped (`Eigenstrain` on `LinearElasticForm`,
  `ThermalStrain`, `LinearElastic(thermal=...)`, closed-form and MMS tests). Open, each additive:
  a **demo** extending `heat` with the warming heatsink's stress at the fin roots, with the
  thick-walled cylinder (logarithmic temperature, closed-form hoop stress) as its benchmark test
  and a critical temperature through `BucklingAnalysis` on a heated restrained bar;
  **finite-strain thermal strain**, an eigenstrain argument on `EnergyDensity.evaluate`, where
  the 2D density must keep the z component of the eigenstrain as the material does for the
  linear path; **thermoelastic sensitivities**, refused today by `SIMPModel`, `SensitivityAnalysis`, and
  the stress quantities of interest because the thermal load scales with the modulus and the
  measured stress omits the eigenstress, so the adjoint needs a `d(load)/d(rho)` term and the
  stress measures a `D eps*` correction; and a plastic strain as the second `Eigenstrain`. Three
  smaller additions sit behind the shipped seam: a **transient thermal-stress series** helper (one
  factorization shared across a `TransientSolution` of `T`), a **`TemperatureParameterization`** for
  `SensitivityAnalysis` (`∂R/∂T_j` is a constant matrix, the first inverse problem needing no
  geometry derivatives), and **thermoelastic topology optimization** as a demo once the gradient
  engine exists ("less material where it is hot").
- 💡 **Prescribed motion under Newmark.** A `TimeDependent` source, traction, or Robin value is
  re-evaluated per step by both integrators, and `ThetaMethod` takes a time-dependent Dirichlet
  value too. `NewmarkMethod` refuses one: a prescribed displacement `g(t)` at the fixed DOFs also
  needs their velocity and acceleration, so the acceleration solve would carry `g''(t)` at the fixed
  block (by differencing `g`, or from a `TimeDependent` that supplies its derivatives) and the
  predictor `g'(t)`. Additive once that data has a home.
- 💡 **Generalized-α, or another integrator family.** `ThetaMethod` and `NewmarkMethod` cover first-
  and second-order systems; the seam for a third is in place, so this is additive. **Error-controlled
  variable `dt`** belongs here too, as a step-size policy over the same seam.

**Engineering**
- 💡 **Coverage gaps.** Coverage gates CI: a whole-suite `--cov-fail-under=90` floor plus a
  `diff-cover --fail-under=85` patch gate on the changed lines, so new code is tested and the total
  cannot decay. The remaining gap is the plot layer: it is exercised end-to-end by
  `tests/test_demos.py` but has no assertions on what it draws.
- 💡 **Missing convergence rates.** Newmark in `dt` (the only integrator without one; a
  `newmark_convergence` on one eigenmode as `theta_convergence` does); finite strain in `h` (three
  meshes against a fine reference); adaptive beats uniform on a localized problem (error against DOFs),
  also for the goal-oriented variant, whose indicator is a Cauchy-Schwarz product rather than a DWR
  bound, so an **effectivity index** against the true QoI error is the number that says whether it is
  usable.
- 💡 **Report the minimum corner angle alongside the demo's simplification tolerance.** The minimum
  corner angle is the number that predicts whether the requested angle bound will hold everywhere
  (the corner treatment in `RuppertsAlgorithm._split_point` / `_spans_a_sharp_corner` keeps output
  size monotonic in input size), so a sweep over `Outline.simplified(tolerance)` should report it,
  alongside the **minimum segment length** (the number that predicted the one measured
  non-termination).
- 💡 **Docstrings on the public API.** Type hints and `pyright` are in place and gating CI; the prose
  half is still open, but narrowly: `mesh/ruppert.py` and `plot/plotter.py` are the modules left
  with no module docstring. The rest of the core has one.
- 💡 **Tighten pyright to `standard`.** It runs in `basic`, which infers types for the unannotated
  internals rather than demanding annotations. Annotating the internals (`refinement`, `ruppert`,
  `energies`, `plot`) would let the mode step up; `fem/plot` in particular has untyped helpers and an
  oversized `Plotter.plot` and is the module that blocks the step.
- 💡 **Widen the ruff rule set.** `select` is `E4, E7, E9, F` only; adding isort, bugbear, and
  pyupgrade would catch more. Do it beside the pyright tightening.
- 💡 **CI matrix.** CI runs 3.11 on Linux only while the package claims 3.10+ and dev is Windows.
  Add a 3.10 leg, ideally a Windows leg.
- 💡 **`wave.gif` weight.** Mark `*.gif` as a binary attribute; `wave.gif` is 2.8 MB.
- 💡 **pre-commit hooks** (ruff + whitespace) so the CI checks run locally before each commit.
- 💡 **Mesh formats.** `fem/post/io.py` writes meshes as JSON. Import of `.off` / `.obj` / `.stl` /
  `.msh` and `.vtu` export (via `meshio`) would make them loadable by standard tools and load
  standard meshes; the curve-serialization piece above is its dependency.
- 💡 **API reference site.** The docstrings are the documentation and nothing renders them. A `pdoc`
  or `mkdocstrings` build beside the gallery deploy would give every symbol in `ARCHITECTURE.md` a
  link.
- 💡 **Version and release.** The only tag is `v0.1.0` from 2024-10, before the package existed in
  its current form; no PyPI release, no changelog. A `v0.2.0` would give the README an install line.

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
- 💡 **L-shaped domain adaptive refinement.** `refinement` runs on a box; the re-entrant corner is
  where adaptive beats uniform, shown as error against DOFs. Pairs with the adaptive-beats-uniform
  test above.
- 💡 **Benchmark suite.** Cook's membrane, NAFEMS cases, plate with hole, L-shape, Euler column; the
  README roadmap lists it and only the Kirsch and Euler checks exist.
- 💡 **A 3D structural demo.** The only 3D solve in the gallery is `timing_benchmark`, which is about
  speed. A 3D bracket or a twisted bar with `PlotMode.SOLID`, von Mises, and a clamp would show the
  3D path exists (elasticity, AMG-CG, recovery all work there).

## Suggested Priority Order

1. **Plot-coverage gap + small guards** (§1, §3): finish the safety net now that type hints gate CI.
2. **Finish the P2 story**: 3D P2, then curved-element adaptivity and the 3D residual estimator, so
   the higher-order path is complete rather than 2D-only. (P2-aware plotting shipped.)
3. **Nonlinear transient, then arc-length**, which together unlock the post-buckling snap-through
   story.
4. **Then the harder numerics**: mixed u-p for incompressibility (P2 is now in place as its
   displacement half), or the hand-rolled two-grid preconditioner.
