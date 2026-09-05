# Backlog: Finite Element Solver

The list of still-open work on the solver: correctness, performance, architecture, and open-ended
ideas.

Legend: 🔴 bug / correctness · 🟠 performance / scaling · 🟡 design / maintainability · 💡 idea.
Effort: 🟢 low · 🔵 medium · 🟣 high.

## At a glance

| Area | Item | Effort | Detail |
|---|---|:---:|---|
| Numerics | Nonlinear transient; flow-theory J2 plasticity; advection-diffusion | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Mechanics | Cyclic plasticity, fatigue life, LEFM stress intensity | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Moonshot | General symbolic weak-form layer; phase-field fracture | 🟣 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Finish P2: curved-element adaptivity, 3D residual estimator, animated warp and 3D plotting | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Mixed (u-p) formulation, then Stokes; hand-rolled two-grid preconditioner | 🟣 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Globalize the Newton direction (Steihaug CG, the unbuilt route) | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Physics | Harmonic response; prestressed vibration; arc-length post-buckling; thermoelastic follow-ups | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| API | Reaction forces, region objects, driver histories, series conveniences | 🟢–🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Structure | Split `forms.py` with the stress-QoI dedup | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Docstrings, CI matrix, pre-commit hooks, API site, release | 🟢–🔵 | [§3](#3-open-ended-suggestions--future-ideas) |
| Demos | Stress-driven design; goal-oriented, L-shape, benchmark, 3D structural | 🔵 | [§3](#3-open-ended-suggestions--future-ideas) |

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
- 💡 **Finish P2: deformed geometry, adaptivity.** **Animated deformed geometry**:
  `plot_animation(meshes=...)` redraws a P1 mesh per frame, so a flexing P2 or curved mode animates
  as its P1 restriction (a still warps faithfully through `panel_view(..., warp=...)`;
  `Mesh.displaced` is P1 by design, a `Mesh` has no edge nodes). The fix is a `warps=` argument, one
  nodal displacement per frame routed through the still path, with `meshes=` reimplemented over it
  and then a deprecation candidate. A **3D P2 field also draws as its P1 restriction**:
  `fem.plot.tessellation`'s sub-lattice is a triangle's, so a tet's surface facets would have to be
  tessellated as quadratic triangles instead. **Adaptive refinement** drives a P2 solve
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
- 💡 **Curved (isoparametric) elements: follow-ups.** **3D curved elements** and **`fem/post/io.py`
  curve serialization** (a saved mesh currently drops its curves; the pieces need a `to_dict` /
  `from_dict` pair) are the remaining gaps. A dedicated *close-up* in the `outline_to_mesh` demo
  contrasting the curved boundary against its chord polygon (the isoparametric payoff) is unbuilt.
  Quadratic Beziers (degree-elevate to cubic) and elliptical arcs (`EllipseArc`) are unbuilt but
  unused by the bundled assets.
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

  A `TangentRegularization` shift alone does not make the SPD-only CG backend reliable on an
  indefinite tangent, because CG's failure is not always
  signalled. The unbuilt route is **truncated / Steihaug CG**: run CG on the tangent and stop at the
  first direction of negative curvature (`pᵀAp ≤ 0`), using the iterate reached so far. CG
  deliberately repurposed for indefinite systems, normally inside a trust region; it is the only way
  to make the nonlinear path reach AMG-CG safely.
- 💡 **`PostBucklingAnalysis`, the facade over `BucklingAnalysis` and `ArcLengthStepping`.**
  Tracing a post-buckling path is now four hand-written steps: solve for `(λ_cr, φ)`, seed a
  geometric imperfection by displacing the mesh in the mode, restate the problem in
  St-Venant-Kirchhoff under the same reference load, and run `ArcLengthStepping`. A facade beside
  `BucklingAnalysis` in the same module should own them, with the imperfection amplitude and the
  λ target as its parameters, and hand back the `PathSolution` with the `BucklingSolution`
  attached as the yardstick. The one real risk is rebinding: geometric region predicates
  re-resolved on the perturbed mesh can miss nodes the mode moved, so the conditions are resolved
  on the pristine mesh and restated with `at_indices`. Ships with the buckling demo's
  load-deflection figure (λ against mid-span deflection for two or three imperfection amplitudes,
  λ_cr as the line the knees flatten toward, the elastica series `P/P_cr ≈ 1 + Θ²/8` overlaid) and
  the elastica validation. Design: `attic/buckling-completion-plan-2026-09-05.md` §4 and §7.
- 💡 **Harmonic response.** `(K − ω²M)u = f`, one indefinite solve `DirectBackend` handles; an
  analysis beside `ModalAnalysis`, the last cheap `EigenSolve` sibling.
- 💡 **Prestressed vibration and stability.** `GeometricStiffnessForm` assembles `K_g(σ₀)` from a
  reference stress state, and `BucklingAnalysis` already uses it. That same `K_g` with the mass
  matrix gives prestressed modal analysis, `(K + K_g(σ₀) − ω²M)φ = 0`: how a loaded structure's
  natural frequencies shift under load (a tensioned string rises in pitch, a compressed strut drops
  toward its buckling frequency, hitting zero at the critical load). A sibling of `ModalAnalysis` and
  `BucklingAnalysis` reusing both their pieces.
- 💡 **Flow-theory J2 plasticity.** Return mapping per quadrature point, with a `PlasticState` /
  `at_state` / `commit` state-carrying interface. The eigenstrain seam already carries a plastic
  strain as its elastic-predictor half.
- 💡 **Cyclic loading, fatigue, and fracture.** A family of mechanical-durability extensions, tiered
  by the machinery each needs. Still a discussion area; the plausible pieces:
  - **Cyclic plasticity** · 🔵 · after flow-theory J2, add a **kinematic-hardening** backstress so a
    load / unload / reload traces a hysteresis loop with the Bauschinger effect, and isotropic
    hardening for cyclic hardening / softening. This is what makes ratcheting and shakedown studies
    possible. The `PlasticState` / `commit` history planned for J2 already carries what these need,
    walked over reversals by `QuasiStaticStepping`. Demo: a notched bar through several reversals,
    plotting the σ-ε loop.
  - **Fatigue life** · 🟢–🔵 · mostly a post-processing layer, not new FEM: from the stress / strain
    amplitude over one or more solved cycles, an empirical stress-life (Basquin / S-N) or strain-life
    (Coffin-Manson) law with rainflow cycle counting gives a cycles-to-failure field. Demo: a plate
    with a hole under cyclic remote stress, the fatigue hotspot at the rim where the Kirsch
    concentration already sits.
  - **Linear elastic fracture (stress intensity factors)** · 🔵 · put a crack (a slit with duplicated
    nodes on its faces) in the mesh, solve elastically, and extract `K_I` / `K_II` by a domain /
    interaction (J-) integral of the Eshelby energy-momentum tensor, a post-processing QoI over the
    existing stress / strain fields. Validate against a handbook edge-crack `K_I`.
  - **Phase-field fracture** · 🟣 · a damage field that degrades stiffness with its own evolution
    equation and a staggered solve; a moonshot, and a natural first customer of the general-forms
    layer below.
- 💡 **Advection-diffusion, and a nonsymmetric backend.** `−div(κ∇u) + b·∇u = f` is a new
  non-symmetric `Form`; a direct backend serves it first, a `GmresBackend` (and SUPG stabilization
  for advection-dominated regimes) after. The README roadmap names it.
- 💡 **1D solves.** `box_mesh` builds 1D meshes but `FunctionSpace` refuses them
  (`LinearLineElement.SUB_TYPE = None`, TODO at `fem/elements.py`). The cheapest convergence check.
- 💡 **General (symbolic) weak-form layer — a moonshot.** 🟣 · Today each `Form` hand-codes its
  element integrals (the B-matrix products, the flux, the tangent). A symbolic layer — a mini-UFL —
  would let the weak form be written as an expression (`inner(grad(u), grad(v)) * dx`,
  `inner(sigma(u), eps(v)) * dx`) and compiled to the element kernel, so a new PDE is a formula
  rather than a new `Form` subclass with hand-derived derivatives. It subsumes advection-diffusion,
  coupled multiphysics, and arbitrary constitutive laws behind one entry point, and pairs naturally
  with autodiff for the tangent. A new subsystem (expression tree, differentiation, kernel
  generation) and a rethink of the `Form` seam, so a genuine moonshot; the current explicit forms
  stay the fast path.

**Post-processing coverage**

The layer has a rule and an owner per quantity (`ARCHITECTURE.md` §3): derived fields flow through
the `Form.flux` -> typed `Solution` -> `fem.post.recovery.recover_nodal` seam.

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
- 💡 **Adjoint sensitivity: follow-ups.** Design record in
  `attic/fem-adjoint-sensitivity-design-2026-08-18.md`; the follow-up plan
  is `attic/fem-adjoint-followups-2026-08-19.md`. The remaining piece for stress-*constrained design*
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
  architecture well, on the same `EnergyForm` / `Energies` machinery.
- 💡 **Thermoelasticity: follow-ups.** Open, each additive:
  a **demo** extending `heat` with the warming heatsink's stress at the fin roots, with the
  thick-walled cylinder (logarithmic temperature, closed-form hoop stress) as its benchmark test
  and a critical temperature through `BucklingAnalysis` on a heated restrained bar;
  **finite-strain thermal strain**, an eigenstrain argument on `EnergyDensity.evaluate`, where
  the 2D density must keep the z component of the eigenstrain as the material does for the
  linear path; **thermoelastic sensitivities**, refused today by `SIMPModel`, `SensitivityAnalysis`, and
  the stress quantities of interest because the thermal load scales with the modulus and the
  measured stress omits the eigenstress, so the adjoint needs a `d(load)/d(rho)` term and the
  stress measures a `D eps*` correction; and a plastic strain as the second `Eigenstrain`. Three
  smaller additions behind the same seam: a **transient thermal-stress series** helper (one
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
- 💡 **Docstrings on the public API.** `mesh/ruppert.py` and `plot/plotter.py` are the modules left
  with no module docstring.
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
- 💡 **A free-body (no boundary conditions) demo.** Extend `modal` to an unconstrained plate: with no
  Dirichlet constraint the stiffness is singular and its null space is the rigid-body modes, which the
  SPD iterative path already projects out. The first three (2D) eigenvalues come out as the
  zero-frequency rigid-body modes, the rest as the elastic ones, making the null space the solver
  handles visible.

## Suggested Priority Order

1. **Small guards** (§1): finish the safety net.
2. **Finish the P2 story**: curved-element adaptivity and the 3D residual estimator, so the
   higher-order path is complete everywhere.
3. **Nonlinear transient, then arc-length**, which together unlock the post-buckling snap-through
   story.
4. **Then the harder numerics**: mixed u-p for incompressibility (P2 is now in place as its
   displacement half), or the hand-rolled two-grid preconditioner.
