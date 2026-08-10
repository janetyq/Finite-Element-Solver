# Higher-Order Elements — the design for removing the P1 ceiling

The plan for lifting the solver off piecewise-linear, single-point-integration assembly:
a real quadrature layer, variable-coefficient forms, and quadratic (P2) elements. Written
as a design record — the reasoning and the rejected alternatives, not just the endpoint —
so the choices survive the people who made them. Anchored on symbol names rather than line
numbers, which drift.

`ARCHITECTURE.md` §5 names this as the anticipated *vertical* extension; `BACKLOG.md` §3
lists the pieces. This is the account of how they fit together and in what order.

## Status — shipped

All five phases landed on the `higher-order-elements` branch. Phase 1 is P1
value-identical (bit-identical in 2D; to round-off in 3D), and the MMS studies prove
the rest: variable-coefficient Poisson at O(h²), scalar **and** vector P2 at O(h³), plus
a constant-stress patch test. P2 is reachable through the `Solver` facade
(`element_type=QuadraticTriangleElement`). The full suite stays green throughout.

**Deferred, and honestly out of what shipped:**

- **3D P2 (`QuadraticTetrahedralElement`).** Only 2D P2 (triangles, with a P2 line
  boundary) shipped. The reference-element and numbering machinery generalizes — 3D P2
  needs the ten-node tet's shape functions and an edge/face numbering — but the O(h³)
  proof the endpoint asked for is 2D, so 3D P2 was left for a follow-up.
- **P2 plotting and `deformed_mesh`.** Both read the vertex DOFs and drop the
  edge-midpoint values, so a P2 field plots as its P1 restriction. Correct for the mesh
  drawn; the quadratic bump between vertices is simply not shown.
- **P2 adaptive refinement.** The residual estimator is P1/2D-only, so a *refined* P2
  solve is unsupported; a single P2 solve is fine.
- **Mixed u–p** stays out of scope, as decided below.

---

## The decision on record

**Goal chosen: general accuracy + variable coefficients.** The endpoint is P2 (quadratic)
displacement/scalar elements plus coefficients and sources that vary within an element. A
pure-displacement P2 stiffness is still **SPD**, so the entire effort stays inside the
algebra the solver already has.

**Out of scope, deliberately: the mixed u–p (Taylor–Hood) formulation for near-incompressible
elasticity.** It was weighed and set aside for this effort. It cures volumetric locking
where P2 only alleviates it, but it produces a **saddle-point (indefinite)** system, which
breaks the SPD assumption every solve path here makes — the CG/AMG backend
(`fem.backends.IterativeBackend`) and its rigid-body near-kernel are SPD-only. That is a
separate solver strategy, not an element swap, and it *depends on P2 existing first*. Revisit
it only if truly near-incompressible materials (nu → 0.5: rubber, tissue, plastic flow)
become a target. Until then, do not re-propose it as part of this work.

---

## 1. What the ceiling is

"P1 + closed-form integration" is two coupled assumptions in the code. Nearly every
limitation traces to one of them.

**Fact A — the integrand is constant per element.** `ElementGeometry.grad_phi` is
`(n_elements, N, spatial_dim)` — one gradient per element, no quadrature-point axis. Every
form exploits this: `LaplacianForm.element_matrices` is `einsum('eid,ejd,e->eij', grad_phi,
grad_phi, volumes)` — *integrand × volume*. `MassForm` scales a closed-form
`reference_mass_matrix()` by volume. There is nowhere to sample anything at an interior point.

**Fact B — one DOF per mesh vertex.** `FunctionSpace.n_dofs = n_vertices * n_components`, and
`dof_indices` maps vertex indices straight to DOF slots. `element_to_vertex`,
`FieldSolution.deformed_mesh`, plotting, and `BoundaryConditions.resolve` all assume nodal
values live exactly on vertices.

The symptoms, each tied to its cause:

| Symptom | Root | Where it lives |
|---|---|---|
| Stress/strain constant per element; nodal output needs a volume-weighted projection | A | `ElasticSolution.stress` is `(n_elements,3,3)`; `FunctionSpace.element_to_vertex` |
| O(h²) in L2 (O(h) in stress) — halving the error costs 4× the DOFs | A+B | the accuracy story |
| Volumetric locking as nu → 0.5 | A | `stress_concentration` nu-sweep (`BACKLOG.md` §3) |
| No `∫ κ(x)∇u·∇v`, no source varying *within* an element | A | load is `mass_matrix @ f`, exact only for f's P1 interpolant |

**On locking, the physical heart.** Near-incompressibility imposes `div(u) ≈ 0` pointwise. A
constant-strain triangle contributes *one* constraint per element (its `div u` is constant)
against very few displacement DOFs; the constraint-to-DOF ratio is so high the element has
almost no admissible deformation left, so it over-stiffens. This is why B-bar and
selective/reduced integration "don't apply to a constant-strain triangle" — nothing varies
inside the element to integrate at reduced order. **Once strain is non-constant (P2), those
lighter anti-locking options become available**, which is the partial answer to
incompressibility that stays inside the SPD world, short of the mixed formulation.

---

## 2. The decomposition that makes this phaseable

The two facts fall independently:

- **Variable coefficients need only Fact A.** A source or coefficient sampled at interior
  quadrature points is a change to how forms integrate — nothing about DOF numbering moves.
- **P2 elements need Fact A *and* Fact B.** Quadratic shape functions have non-constant
  gradients (need the quadrature axis) *and* carry edge-midpoint DOFs (need the numbering to
  extend past vertices).

So Fact A can be broken first, delivering variable coefficients on the untouched P1 DOF map,
and validated against the convergence test before Fact B's invasive renumbering lands. This
is the seam that turns one rewrite into two bounded efforts.

---

## 3. Design decisions

### D1 — the shape of `ElementGeometry` (how Fact A falls)

The load-bearing choice: every form and the whole `EnergyForm` chain read `grad_phi`.

- **1a — Unify: add a quadrature-point axis; P1 becomes the 1-point rule. ✅ chosen.**
  `grad_phi → (n_el, n_qp, N, sdim)`, plus shape values `phi (n_qp, N)` and
  `weight_detJ (n_el, n_qp)`; `volumes` becomes derived (`weight_detJ.sum(axis=1)`). Forms
  gain one summed index: `einsum('eqid,eqjd,eq->eij', ...)`. For P1 with a 1-point centroid
  rule, `n_qp = 1` and the sum is a no-op — **byte-identical numerics, new shape.** This is
  the codebase's own aesthetic: the simple thing is a *special case* of the general one, the
  way `Form ⊂ EnergyForm`, `LinearProblem ⊂ EnergyProblem`, `LinearSolve ⊂ NewtonSolve`. The
  batched-einsum performance model survives intact — a quadrature axis is just another
  contraction index, and `n_qp` is tiny (1 for P1, 3–4 for a P2 triangle).
  - *Cost:* the shape change touches every form + the energy path + `derived_fields` in one
    coordinated commit. The MMS test is what makes that safe (§5).
- **1b — Parallel path: keep P1 geometry, add quadrature geometry beside it, forms opt in.**
  Lower blast radius per commit, but carries *two* assembly paths permanently and forfeits
  the "P1 is a special case" structure. Against the grain. Rejected.
- **1c — Caller-supplied `(func, points)`, the shape of today's `fem/quadrature.py`.** Wrong
  layer: a real rule needs *reference-element* points and weights, not a polygon plus a
  function. This is exactly what `ARCHITECTURE.md` §5 flags for deletion. Rejected.

**Chosen: 1a.** Replace `fem/quadrature.py` with a `QuadratureRule(points, weights)` on the
reference simplex, and give `Element` two classmethods: `shape_values(ref_points)` and
`shape_gradients(ref_points)`.

### D2 — keep P1 exact where quadrature would only approximate it

A 1-point rule integrates the P1 **stiffness** exactly (constant integrand) but the
consistent **mass** matrix needs a degree-2 rule to stay exact. Rather than route the P1 mass
through a coarser rule and change its value, let an element advertise a closed-form reference
matrix that a form uses when the integrand is a low-degree polynomial — keeping
`reference_mass_matrix` as the P1 mass path. Quadrature is then used where it is *needed*, not
everywhere for uniformity's sake, and the hot P1 path stays cheap. `MassForm`'s value does not
change.

### D3 — DOF numbering (how Fact B falls, for P2)

P2 adds a node per **edge**, and `mesh.edges` already exists as globally-sorted `(v0, v1)`
pairs — a ready-made global edge numbering. The work:

- `FunctionSpace` numbers DOFs over `vertices ∪ edges`; `n_dofs` and `dof_indices` generalize
  (6 nodes/triangle, 12 for a vector field).
- **BC resolution is the subtle part.** `BoundaryConditions.resolve` works vertex-by-vertex;
  a Dirichlet boundary edge must also pin its midside DOF, and a traction needs a P2 boundary
  mass matrix. Because regions are *geometric callables of position*, evaluating a BC value at
  an edge midpoint is natural — the spec/resolve split carries over cleanly, a real dividend
  of the existing design.
- Plotting and `deformed_mesh` read vertex DOFs and drop the midside term (documented
  limitation); a display-subdivision pass is a later nicety, not a blocker.

### D4 — the elasticity endpoint

Resolved above: **P2 displacement, SPD throughout.** Mixed u–p deferred with its reason
recorded. For locking specifically, P2 plus (optionally) selective integration is the
in-scope mitigation; the full cure is the out-of-scope mixed formulation.

---

## 4. What does *not* change (the dividends)

- **The algebra layer is untouched.** A P2 pure-displacement stiffness is SPD, so
  `DirectBackend` and the AMG-CG `IterativeBackend` both keep working. No saddle-point
  strategy, no indefinite solver.
- **The performance model survives.** Assembly stays one vectorized einsum pass per form; the
  quadrature axis is one more index, not a Python loop.
- **The composition model holds.** P1 is the 1-point special case of the quadrature-aware
  path — no new tier, no dispatch, consistent with the rest of the object model.

One small consequence to fold into Phase 3, so it is not a surprise: the AMG near-kernel
`rigid_body_modes(mesh.vertices, …)` (built in `Solver._backend_for`) reads *vertex*
coordinates, so P2 elasticity on the *iterative* backend needs it generalized to include
edge-node coordinates. A few lines, and it only bites when P2 + `IterativeBackend` +
elasticity coincide — the direct path needs nothing.

---

## 5. The plan

The whole change lands as **one branch, one squash-merge**: the abstraction is justified on
landing because nothing speculative is left unused — the quadrature layer arrives together
with the P2 elements and variable coefficients that exercise it to completion. The two
milestones below are the branch's **internal commit sequence**, not separate merges.

| Milestone | Phases | Delivers | Touches |
|---|---|---|---|
| **M1 — Quadrature + variable coefficients** | 0, 1, 2 | `∫ κ(x)∇u·∇v`, in-element-varying sources, exact quadrature layer | Fact A only — no DOF changes |
| **M2 — Higher-order accuracy** | 3, 4 | P2 elements, O(h³), better stress recovery | Fact B — DOF numbering, BC resolution, plotting |

Every commit in the sequence keeps the full suite green; the **Phase-1 commit additionally
keeps every existing number bit-identical** (P1 through a 1-point rule), the checkpoint that
proves the refactor before any new capability builds on it. The squash-merge collapses the
sequence into one commit on `main`, so the internal phasing is build-time scaffolding for
bisectable regressions, not preserved history.

| Phase | Scope | Proof / guardrail |
|---|---|---|
| **0 · Quadrature primitives** | Replace `quadrature.py` with `QuadratureRule(points, weights)` + Gauss rules on triangle/tet; add `Element.shape_values` / `shape_gradients`. No behavior change. | Unit test: each rule integrates monomials exactly to its advertised degree. |
| **1 · Quadrature-aware geometry (Fact A), P1 value-identical** | Add the quadrature axis to `ElementGeometry`; migrate `LaplacianForm`, `LinearElasticForm`, `MassForm`, the `EnergyForm` chain, and `derived_fields` to sum over quadrature points. P1 uses a 1-point rule ⇒ identical numbers. | **MMS stays green at O(h²)**; full suite green after each form migrates. A refactor with a proof. |
| **2 · Variable coefficients + a real `LinearForm`** | `∫ f(x)·v` sampling f at quadrature points (replaces `mass_matrix @ f` where f varies in-element); `∫ κ(x)∇u·∇v`. Still P1. | New MMS case: manufactured variable-coefficient Poisson, still O(h²). |
| **3 · P2 elements (Fact B)** | `QuadraticTriangleElement` (+ line boundary, then tet); generalize `FunctionSpace` DOF numbering over `vertices ∪ edges`; generalize BC resolution to edge DOFs; generalize `rigid_body_modes` to edge nodes. | New MMS case: **P2 Poisson shows O(h³)** — the payoff and the proof B fell correctly. |
| **4 · Validation** | P2 elasticity convergence + stress-recovery check; optional locking nu-sweep (reuse the `stress_concentration` diagnostic). No new solver. | Convergence rates; nu-sweep flattens relative to P1. |

**Sequencing rationale: Fact A before Fact B.** Variable coefficients exercise the entire
quadrature machinery on the *unchanged* DOF numbering, so by the time P2's renumbering lands
the quadrature layer is already trusted and covered.

**The spine of the effort is `tests/test_convergence.py`** — the MMS test `AGENTS.md`
explicitly protects. It is both the guardrail (P1 must stay O(h²) through the Phase 1
refactor) and the payoff (P2 must show O(h³) in Phase 3). Every phase proves itself against
it, which turns "did I break the physics?" into a green/red signal.

---

## 6. Risks and how they are held

- **Phase 1 is wide.** It changes a shape every form reads. Held by 1a's value-identity: with
  a 1-point P1 rule the numbers do not move, so the full suite — not just MMS — is a
  bit-for-bit guardrail. Migrate one form at a time.
- **Phase 3 is deep.** DOF numbering, BC resolution, and plotting move together. Held by
  landing it *after* the quadrature layer is trusted, and by the O(h³) MMS test as an
  unambiguous correctness signal — a wrong DOF map does not accidentally converge at the right
  rate.
- **Scope creep toward the mixed formulation.** Held by the decision record above: it is
  out of scope until incompressibility is an explicit goal.

---

## References

- `ARCHITECTURE.md` §5 — the vertical-extension framing and the two coupled assumptions.
- `BACKLOG.md` §3 — quadrature, higher-order elements, mixed u–p, `quadrature.py`'s fate.
- `tests/test_convergence.py`, `fem/convergence.py` — the MMS spine.
