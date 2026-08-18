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
| Numerics | 3D P2 (`QuadraticTetrahedralElement`); P2 plotting / adaptivity | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Mixed (u-p) formulation for near-incompressible elasticity | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Hand-rolled two-grid preconditioner (drop `pyamg`) | 🔴 | [§3](#3-open-ended-suggestions--future-ideas) |
| Numerics | Globalize the Newton direction (SPD tangents → iterative nonlinear solves) | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Physics | Plane stress as an alternative 2D reduction | 🟡 | [§3](#3-open-ended-suggestions--future-ideas) |
| Post-proc | Transient derived fields (steady flux/stress recovery shipped) | 🟢 | [§3](#3-open-ended-suggestions--future-ideas) |
| Design | Nodal L2 projection option; lazy plot / `pyamg` imports | 🟢 | [§3](#3-open-ended-suggestions--future-ideas) |
| Tooling | Coverage (`pytest-cov`), API docstrings, pre-commit | 🟢–🟡 | [§3](#3-open-ended-suggestions--future-ideas) |

---

## 1. Bugs & Correctness

No open correctness bugs.

---

## 2. Performance & Scaling

### 🟠 Every `solve()` re-assembles from scratch
`Solver._steady_problem` builds a fresh `LinearProblem` per call, whose constructor assembles the
operator and load. The two looping drivers already avoid this: the integrators build one
`DiscreteSystem` and reuse its factorization across steps, and `TopologyOptimizer` rescales one set
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
  and connectivity are not written. **`deformed_mesh`** reads the vertex DOFs and drops the
  edge-midpoint displacements, so a P2 displacement warp draws as its P1 restriction (field plotting
  itself is now P2-aware via `FunctionSpace.tessellation`). **Adaptive refinement** is P1-only (the
  residual estimator is 2D-P1), so a *refined* P2 solve is unsupported.
- 💡 **Curved (isoparametric) elements: follow-ups.** The core shipped:
  `IsoparametricTriangleElement` (a geometry map differentiated over all nodes, per quadrature
  point), `Circle` / `Arc` curves carried through `PSLG` -> `RuppertsAlgorithm` ->
  `Mesh.boundary_curves`, boundary-node projection in `p2_connectivity`, curvature-aware Ruppert and
  red-green refinement, a curved `MassForm`, P2-aware plotting (`FunctionSpace.tessellation` through
  `Plotter.plot(..., space=...)`, with the `curved_elements` gallery demo), SVG cubic Beziers
  retained as `CubicBezier` curves through `read_svg_to_pslg` (adaptive flatness sampling, tag-aware
  Douglas-Peucker), and validation (`tests/test_convergence_curved.py` area fidelity and the P2 rate;
  `tests/test_curved_meshing.py` the pipeline and the Kirsch stress concentration; `tests/test_svg.py`
  the traced-outline round trip). Two follow-ups are left. **3D curved elements** and **`fem/io.py`
  curve serialization** (a saved mesh currently drops its curves) are the remaining gaps. `files/cloud.svg`
  now meshes and solves in the `outline_to_mesh` demo, so its Bezier boundary carries through the pipeline
  there; a dedicated *close-up* contrasting the curved boundary against its chord polygon (the isoparametric
  payoff) still belongs beside `curved_elements`, unbuilt. Quadratic Beziers (degree-elevate to cubic) and
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
  SPD system every solve path here currently assumes (`fem/backends.py`'s CG path and its
  rigid-body near-null-space handling, in particular), so it needs its own solver strategy alongside
  the new element. Now that P2 elements exist, selective/reduced integration on the volumetric term
  is also available as a lighter-weight partial answer the constant-strain triangle could not offer.
- 💡 **Hand-rolled geometric two-grid V-cycle preconditioner.** The SPD iterative path
  (`fem/backends.py:IterativeBackend`, AMG-CG) currently gets its multigrid from `pyamg`. A
  geometric two-grid V-cycle built on the adaptive-refinement mesh hierarchy would drop in behind
  the same `Backend` seam without touching a caller, removing the dependency and being a genuinely
  instructive build. Full AMG is thousands of lines and not worth reimplementing; a two-grid cycle
  is small and teaches the same ideas. The yardstick to hold is `examples/benchmark_assembly.py`:
  on the 3D elastic benchmark AMG-CG overtakes `splu` at n≈13 and is ~10× faster by n=21.
- 💡 **Globalize the Newton direction, so the nonlinear path can reach the iterative backend.**
  `NewtonSolve` takes no `Backend` and `EnergySolver` therefore cannot either: CG is SPD-only, and a
  Newton tangent is the Hessian `∇²Π(u)` at the current iterate, which is SPD only where the energy
  is locally strictly convex. The St-Venant-Kirchhoff energy is not convex in `F` (it loses
  ellipticity under compression), so the tangent can be indefinite at the `u = 0` seed, at an
  intermediate iterate, or near a buckling configuration. Large 3D nonlinear solves pay the direct
  factorization's fill-in as a result, on the same curve where AMG-CG wins by ~10×.

  Step *length* is already globalized: `NewtonSolve` takes an optional `BacktrackingLineSearch`
  (Armijo on Π, else ½‖r‖²), which `EnergySolver` uses by default, so a full step no longer diverges
  from a poor seed. What remains is the step *direction*: at an indefinite tangent the line search
  has no descent direction to scale and falls back to the full step. The fix is to make the tangent
  SPD by construction, both routes fitting behind the existing `SolveStrategy` / `Backend` seams:
  - **Regularized (modified) Newton** solves `(H + τI) Δu = −r`, raising `τ` until the operator is
    positive definite. Gives a descent direction even at a saddle, and makes CG safe every iteration.
  - **Truncated / Steihaug CG** runs CG on the tangent and stops at the first direction of negative
    curvature (`pᵀAp ≤ 0`), using the iterate reached so far. CG deliberately repurposed for
    indefinite systems, normally inside a trust region.
- 💡 **Nonlinear post-buckling, the sequel to `BucklingSolver`.** Linearised buckling finds the
  critical load and the mode shape (`fem/buckling.py`), but not what the structure *does* past the
  bifurcation: the load-deflection path once it has bowed. That is a geometrically nonlinear
  (St-Venant-Kirchhoff) solve seeded with a small imperfection in the buckling mode, and it needs
  exactly the globalized Newton above *plus* arc-length (Riks) control, since the tangent goes
  indefinite and the load-displacement curve turns back on itself at the limit point, where
  load-controlled and displacement-controlled Newton both stall. The pieces line up (`EnergySolver`
  for the energy, the buckling mode for the imperfection shape, a globalized tangent for the
  indefinite region), so this is additive once arc-length joins the `SolveStrategy` family.

**Post-processing coverage**

The layer has a rule and an owner per quantity (`ARCHITECTURE.md` §3). Steady solves now recover
their derived fields through one seam: `Equation.derived_field` names the field (Poisson's gradient,
elasticity's stress, `fem.postprocess.DerivedField`), the typed `Solution` carries it per element
(`ScalarFieldSolution.flux`, `ElasticSolution.stress`), and `FunctionSpace.recover_nodal` turns it into
a continuous per-node field for smooth output, P2 plotting, and the recovery estimator. The remaining
gap is the transient path.

- 💡 **Derived fields for transient solves.** `TransientSolution` carries a per-step series of `u`
  and nothing derived from it, so a time-stepped heat problem has no flux history. The steady seam
  (`DerivedField` + `recover_nodal`) is the piece to lift onto each step.
- 💡 **Plane stress as an alternative 2D reduction.** 2D elasticity is plane strain throughout, now
  named rather than implicit (`LinearElasticMaterial.out_of_plane_stress`, and the matching
  `out_of_plane_stress` on the energy densities). Plane stress (a thin plate free to contract in z,
  so `sigma_zz = 0` and `eps_zz = -nu/(1-nu) (eps_xx + eps_yy)`) needs a different `D` as well as a
  different out-of-plane component, so it is a second branch in both places plus a way for a caller
  to choose. Worth doing when a thin-plate problem actually appears; a single-member enum ahead of
  that is generality with no second case.

**Design / maintainability**

- 💡 **Nodal L2 projection option for `recover_nodal`.** `FunctionSpace.recover_nodal` recovers a
  per-element field to the nodes by volume-weighted averaging, the shipped default and the reason the
  space (not the mesh) owns it. The stricter choice is the mass-matrix L2 projection (solve
  `M u = ∫ f φ`), more accurate on a graded mesh at the cost of a solve, and a higher-fidelity flux
  than the averaging recovery for the same seam. Worth adding as a `method=` option only if a
  nodal-output consumer needs the accuracy; plotting and the error estimator do not.
- 💡 **Lazy plot and `pyamg` imports for headless use.** `fem/__init__.py` re-exports `Plotter` /
  `PlotMode`, and `fem.backends` imports `pyamg` at module scope, so `import fem` always pulls in a
  plotting backend and `pyamg`. Making both edges lazy would let the package import without them.
  Worth doing only if a headless import becomes a goal.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear hyperelasticity
  via the existing `EnergySolver` / `Energies` machinery) all fit the current architecture well.
  `NeohookeanEnergyDensity` is a stub: filling in its `W` and derivatives gives a nonlinear material
  through the already-working Newton solver. Note it is naturally written in invariants of `C = FᵀF`
  rather than in a strain tensor `S`, so it does not slot into the St-VK class's `S`-based derivative
  chain as cleanly as the shared-`W` framing above might suggest; it wants its own `evaluate`.
- 💡 **Time-varying loads and Dirichlet data.** Source terms and BC values are functions of position
  only, so a `Problem`'s load is built once and assumed constant in time. Both integrators lean on
  it: `ThetaMethod` reuses one `problem.load` where a general θ-method averages `b_n` and `b_{n+1}`,
  and `NewmarkMethod` reads a fixed Dirichlet displacement as zero velocity and acceleration there.
  The extension is a `t` argument on those callables and a load the integrator re-evaluates per step.
- 💡 **Generalized-α, or another integrator family.** `ThetaMethod` and `NewmarkMethod` cover first-
  and second-order systems; the seam for a third is in place, so this is additive.
- 💡 **External work term for `EnergySolver`.** It minimizes the internal elastic energy only and
  builds no load vector, so it currently rejects `Equation.source` outright. Adding the external work
  term `-f · u` (and its gradient/Hessian contributions) would make it accept forced problems, which
  is also a prerequisite for using it on the nonlinear roadmap.

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
- 💡 **Mesh formats.** `fem/io.py` writes meshes as JSON; `.off` / `.obj` export would make them
  loadable by standard tools.

---

## Suggested Priority Order

1. **Coverage + type hints** (§3): deepen the safety net.
2. **Finish the P2 story**: 3D P2, then P2-aware plotting and adaptivity (§3), so the higher-order
   path is complete rather than 2D-only.
3. **Then the harder numerics**: mixed u-p for incompressibility (P2 is now in place as its
   displacement half), or the hand-rolled two-grid preconditioner.
