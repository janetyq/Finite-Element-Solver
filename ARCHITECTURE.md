# Architecture

An overview of the object model: which concepts exist, which object owns each job, and how they
fit together. Anchored on symbol names rather than line numbers. Open work lives in `BACKLOG.md`.

## The idea

A solve is a composition assembled from parts, not a method looked up by PDE. The package has the
parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedBC`), the object that holds
a composition (`Problem`), the strategies that consume one (`LinearSolve`, `NewtonSolve`, the time
integrators), and the drivers that wrap a strategy to re-solve (`AdaptiveRefinement`,
`DesignOptimizer`). A transient problem is a steady operator paired with a time integrator, not a
PDE type, so `Equation` carries only the identity of a PDE and its physical constants. "What to
solve" is the `Problem`; "how" is the strategy.

## Building a solve

The chain is the same for every problem. Each step has one required object and a few options with
defaults; a facade is this chain with the defaults filled in from an `Equation`.

| Step | Required | Options (default) |
|---|---|---|
| Geometry | `Mesh` | |
| Discretization | `FunctionSpace(mesh, n_components)`, or `Equation.space(mesh)` | `element_type` (linear) |
| Physics | a `Form`, or `Equation.operator` | a `Material` for elasticity; an `EnergyForm` over a strain-energy density for finite strain |
| Statement | `Problem(space, form)`, or `Equation.problem(space, bc)` | `source` (none), `bc` (none); a `LinearProblem` when the form has a constant tangent |
| Solve | `default_strategy(problem)`: `LinearSolve` for a constant tangent, else `NewtonSolve`; or an integrator | `Backend` (direct); Newton: `line_search`, `regularization`; integrator: `dt`, `steps`, `theta` / `beta` |
| Result | `problem.solution(u)` | |
| Outer loop | | `AdaptiveRefinement` over a `problem_for(mesh)` builder; `DesignOptimizer` over a `SIMPModel` |

Composed by hand:

```python
space = FunctionSpace(mesh, n_components=1)
problem = LinearProblem(space, LaplacianForm(), source=1.0, bc=bc)
u = LinearSolve(backend=IterativeBackend()).solve(problem)
solution = problem.solution(u)
```

The same solve through the facade, which builds the space and problem from the equation and
packages the result:

```python
solution = Solver(mesh, Poisson(source=1.0), bc, backend=IterativeBackend()).solve()
```

The two agree exactly: the facade holds no policy of its own, so anything it can do can be composed,
and anything composed (a different form, a hand-built load, a custom strategy) needs no facade.

## Layers

| # | Layer | Question it answers | Varies with |
|---|---|---|---|
| 1 | Geometry / topology | Where are the nodes, what connects to what? | meshing, refinement |
| 2 | Discretization (function space) | What functions can I represent? How are DOFs numbered? | element order, components per node |
| 3 | Physics (forms + materials) | What equation, what constitutive law? | the PDE being solved |
| 4 | Assembly | How do forms become matrices? | quadrature, element type |
| 5 | Constraints | Which DOFs are fixed, to what? | boundary conditions |
| 6 | Algebra | How is `Ax = b` (or `F(x) = 0`) solved? | direct/iterative |
| 7 | Time integration | How does a semi-discrete system advance in `t`? | scheme, step size |
| 8 | Drivers | Outer loops that re-solve: adaptivity, optimization | the study being run |
| 9 | Post-processing | Derived quantities, I/O, plotting | what you want to see |

Each layer can be swapped without touching its neighbours: direct for iterative algebra (6) without
touching physics (3), Crank-Nicolson for backward Euler (7) without touching the operator, a
refined mesh (1) without re-resolving constraints by hand (5).

## Tiers of a solve

| Tier | Role | Objects |
|---|---|---|
| 1. Primitives | the parts a composition is built from | `Form` (the `BilinearForm`s, `EnergyForm`; combinators `ScaledForm`, `MaskedMassForm`, `PrecomputedForm`), `Material`, `FunctionSpace`, `BoundaryConditions` / `ResolvedBC`, `DiscreteSystem` + `Backend` |
| 2. `Problem` | a composition: space + operator + load + constraints | `Problem`, and `LinearProblem` for a constant tangent; `Equation.problem` builds one from a named PDE |
| 3. Solve strategy | consumes a `Problem`, returns the solution | `LinearSolve`, `NewtonSolve`, `EigenSolve`; integrators `ThetaMethod`, `NewmarkMethod` |
| 4. Driver | wraps a strategy, re-solving | `AdaptiveRefinement`, `DesignOptimizer` |

Tier 3 has a second, orthogonal axis: the strategy picks linear vs. Newton, a `Backend` picks
direct vs. iterative. Named PDEs are `Equation`s, not dispatch keys: `Poisson(f).problem(space, bc)`
returns `LinearProblem(space, LaplacianForm(), f, bc)`, `LinearElastic(E, nu,
kinematics=GREEN_LAGRANGE).problem(space, bc)` a `Problem` over an `EnergyForm`, and a PDE with
no name is just a different composition.

A form is one base class. Every `Form` writes `element_residuals` and `element_tangents` at a
state; what else it can answer is a hook with a default of "no": `constant_tangent` (every
`BilinearForm`, whose residual is `K u`), `has_energy` (a bilinear form's `½ uᵀ K u`, an
`EnergyForm`'s density), `derived_field` (the recoverable flux), `near_null_space` (the AMG
near-kernel). A consumer reads the answer it needs: `LinearSolve`, the integrators, the analyses,
and SIMP need a constant tangent; `NewtonSolve` needs nothing more and uses the energy as its
line-search merit when there is one. `RecoversElasticFields` stays a protocol: the interface the
two elastic forms share with `ElasticSolution` and `StressField`.

## Where the classes sit

`█` = owns the layer, `▒` = shares it.

| Class | 1 Geom | 2 Space | 3 Phys | 4 Asm | 5 Cons | 6 Alg | 7 Time | 8 Drive | 9 Post |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Mesh`, `Curve` | █ | | | | | | | | |
| `RuppertsAlgorithm`, `RedGreenRefiner` | █ | | | | | | | | |
| `Element` / `ElementGeometry`, `QuadratureRule` | | ▒ | | ▒ | | | | | |
| `FunctionSpace` | | ▒ | | █ | | | | | ▒ |
| `FieldShape` (`Scalar` / `Vector`) | | | ▒ | | | | | | |
| `Form` (`BilinearForm`s, `EnergyForm`) | | | █ | ▒ | | | | | ▒ |
| `Material` / energy densities | | | █ | | | | | | |
| `Equation` | | | █ | | | | | | |
| `BoundaryConditions` / `ResolvedBC` | | | | | █ | | | | |
| `Problem` / `LinearProblem` | | | ▒ | ▒ | █ | | | | ▒ |
| `DiscreteSystem` | | | | | ▒ | █ | | | |
| `Backend` (`Direct`, `Iterative`, `Minres`) | | | | | | █ | | | |
| `LinearSolve` / `NewtonSolve` / `EigenSolve` | | | | | | ▒ | | | |
| `ThetaMethod` / `NewmarkMethod` | | | | | | ▒ | █ | | |
| `Solver` | | | | | ▒ | ▒ | | | |
| `BucklingAnalysis`, `ModalAnalysis` | | | | | | ▒ | | | ▒ |
| `AdaptiveRefinement`, `SIMPModel` / `DesignOptimizer` | | | | | | | | █ | ▒ |
| Error estimators, `SensitivityAnalysis` | | | | | | | | ▒ | █ |
| `Solution` (typed) | | | | | | | ▒ | | █ |
| `invariants`, `Plotter`, `io` | | | | | | | | | █ |

Constraints (5) have one owner, the `Problem`, whose constructor resolves the boundary conditions
against the space and folds the Dirichlet partition, the Neumann load, and any Robin contribution
into the operator and load. Algebra (6) is split: `DiscreteSystem` owns the Dirichlet elimination
and the `Backend` owns how the free-free block is solved. Physics (3) is owned by the physics
layer alone: an `Equation` answers `operator` itself, for either kinematics, so no facade holds a
mapping from equation to material.

Post-processing (9) is distributed under one rule: a derived quantity lives on the object that
owns the data it needs. `FunctionSpace` owns `integrate`, `recover_nodal`, and `nodal_gradient`;
a `Form` owns `fields_at` / `derived_fields` and `derived_field` (which flux is recoverable, read
by the estimators off `problem.operator`); `Problem.solution(u)` picks the typed `Solution` for its
operator; `Solution` owns the packaging (`ElasticSolution.stress`, `nodal_stress`,
`deformed_mesh`); `invariants` owns the frame-independent reductions.

## Role by role

### `Mesh` / `FunctionSpace`: geometry vs discretization

`Mesh` is geometry: vertices, elements, boundary, topology queries, and optionally the analytic
`Curve` each boundary facet was sampled from. `FunctionSpace` has a mesh and owns the
discretization: element geometry, DOF numbering, cached operators. Two spaces (P1 and P2, scalar
and vector) can share one mesh. `fem/mesh` imports no plot code.

### Elements, quadrature, and assembly

The integration stack is `QuadratureRule -> Element -> ElementGeometry -> Form`. A `QuadratureRule`
is reference-simplex data; an `Element` type is stateless and maps node coordinates to a batched
`ElementGeometry` holding `grad_phi (n_el, n_qp, N, spatial)`, `weight_detJ`, and `points` for
the whole mesh; a `Form` contracts that into element matrices in one vectorized pass. P1 is the
`n_qp == 1` special case, so the same forms serve every order.

DOFs are numbered by the space, never read off the mesh: `element_nodes`, `node_coords`, and
`boundary_nodes` are the mesh's own arrays for P1, and for P2 the `NodeSet` that `p2_connectivity`
builds (vertices, then one edge-midpoint node per edge). `BoundaryConditions.resolve` takes either,
so a geometric condition pins the edge DOFs too. An `IsoparametricTriangleElement` raises the
geometry map to quadratic as well, placing its edge nodes on the mesh's curves so the boundary
follows the true curve.

`FunctionSpace.assemble` is where the two meet:

```
1. degree = form.quadrature_degree (or the element's default)
2. ElementGeometry = element.geometry(node_coords[element_nodes], rule(degree))
3. blocks = form.element_matrices(ElementGeometry)
4. _ScatterPlan(dof_indices(element_nodes)).scatter(blocks)
```

### `Form` / `Material`: the physics

Every assembly path goes through a form. Bilinear forms (`MassForm`, `LaplacianForm`,
`DiffusionForm`, `LinearElasticForm`, `GeometricStiffnessForm`) follow `Gᵀ C G` with the element
supplying `G` and the form `C`; `ScaledForm` and `MaskedMassForm` are combinators (the wave
operator's `c²K`, the Robin boundary integral); `PrecomputedForm` lets a driver reuse element
matrices it can derive more cheaply (SIMP rescales one set by `rho^p`). Each is a `BilinearForm`,
which supplies the residual `K u` and the energy `½ uᵀ K u` from its matrix. `EnergyForm` is the
state-dependent form, mapping an element and a state to energy, residual, and tangent; both kinds
assemble through `assemble_residual` / `assemble_tangent` / `total_energy`, at the larger of the
element's default rule and the one the form asks for.

`Material` owns the constitutive matrix `D`; the strain-displacement matrix `B` sits in
`fem/forms.py` next to the form that contracts it. The physics decomposes as material (the energy
`W`) times kinematics (the strain measure): `SmallStrain` and `StVenantKirchhoff` feed one `W`
either the small-strain `ε` or the Green-Lagrange `S`, chosen on the equation
(`LinearElastic(kinematics=...)`). The linear path accepts only `SMALL`. In 2D the law is plane
strain throughout.

Stress recovery is on the form (`RecoversElasticFields`): `fields_at` gives strain and stress at
every point of a geometry's rule, `derived_fields` reduces that to one tensor per element. Full
`(n_elements, 3, 3)` tensors cross the boundary, never Voigt vectors, and `fem/invariants.py`
reduces them to frame-independent scalars.

### `Problem`: the narrow waist

A `Problem` is the assembly-ready composition for one mesh: space, operator, load, constraints.
Its residual has three terms, each present in `energy`, `residual`, and `tangent` alike: the
form's own (`Π_form`, `R_form`, `∂R_form/∂u`), the Robin boundary term (`½κuᵀRu`, `κRu`, `κR`),
and the load (`−fᵀu`, `−f`, `0`). `internal_residual` is the first two, kept apart from `load` so
a strategy can scale one against the other. `LinearProblem` is the case whose operator has a
constant tangent: `tangent()` with no state is the matrix, assembled once and held, and
`residual(u) = A·u − b`; it is the type every consumer that needs one fixed operator asks for. The
`Problem` owns its constraints, so nothing index-keyed is carried across a mesh change: a driver
that remeshes builds a new one. `bc` is the spec the constraints came from and `resolved` that
resolution on this space. It also answers the two questions that depend on which physics it was
composed from: `solution(u)`
packages a solved vector as the typed `Solution` its operator recovers (`ElasticSolution` for a
form that recovers stress, `ScalarFieldSolution` for one naming a flux, else `FieldSolution`), and
`near_null_space()` is the operator's AMG near-kernel, if it has one. Both delegate to the form,
so a solve composed by hand and one through a facade give the same answer.

### Solve strategies and backends

`LinearSolve` and `NewtonSolve` sit on `DiscreteSystem` (matrix + Dirichlet partition +
factor-once solve) and know nothing about the PDE. `LinearSolve` requires a constant tangent.
`NewtonSolve` takes any `Problem` and an optional `BacktrackingLineSearch` that scales each step
to decrease a merit (the energy `Π(u)` where the problem has one, else `½‖r‖²`).
`default_strategy(problem, backend)` is the one place the choice between them is made for a
caller who names none: `LinearSolve` for a constant tangent, line-searched Newton otherwise,
with `TangentRegularization` added under an iterative backend.

`DiscreteSystem` eliminates the Dirichlet DOFs and hands the free-free block to a `Backend`, which
prepares it into a `LinearSolver` solved against many right-hand sides. `DirectBackend` is sparse
LU; `IterativeBackend` is AMG-preconditioned CG, SPD-only and opt-in; `MinresBackend` handles
symmetric indefinite systems. `backend_for(problem, backend)` hands an `IterativeBackend` the
problem's near-kernel (`LinearElasticForm.near_null_space`, the rigid-body modes, restricted to
the free DOFs) unless the caller set one; `LinearSolve` and `SensitivityAnalysis` go through it.

`EigenSolve` covers the solves that are not `Ax = b`: linearised buckling (`K φ = -λ K_g φ`) and
modal analysis (`K φ = ω² M φ`) share the Dirichlet elimination, the `eigsh` call, and the lift of
each eigenvector back to a full DOF vector. `BucklingAnalysis` and `ModalAnalysis` consume a
`LinearProblem` and return a typed solution, buckling after solving it once for the prestress.

### Time integration

Heat is first order and wave second, so there is one integrator family per order. Each forms a
constant effective operator from the problem's mass and stiffness, factors it once, and steps by
updating the right-hand side. `ThetaMethod` (Crank-Nicolson by default, backward Euler at θ=1)
and `NewmarkMethod` (average acceleration, solving for the acceleration against the SPD
`M + β dt² K`) both take a `Backend`.

### `Equation`

`Equation` is typed data: `Projection`, `Poisson`, `Diffusion`, `Wave`, and `LinearElastic`, each
carrying its physical constants. `operator(space)` returns the form for its physics: the
small-strain stiffness or the St-Venant-Kirchhoff `EnergyForm`, by `LinearElastic.kinematics`.
It refuses rather than approximates when the physics does not apply.
Two more resolve it against a discretization: `space(mesh, element_type)` builds the
`FunctionSpace` with the component count the field implies, and `problem(space, bc)` the
`LinearProblem`. Every facade goes through these two.

### Facades and drivers

`Solver` holds an equation, a BC spec, and the space on a mesh; builds a `Problem` per solve
(`Equation.problem`); hands it to a strategy (the caller's, else `default_strategy` over the
`Backend`); returns `problem.solution(u)`. It fills in defaults and holds no other solve policy
and no state between solves.

Two drivers, each over one spec. `AdaptiveRefinement` owns a mesh and a `problem_for(mesh)`
builder (`equation.problem(equation.space(mesh), bc)`), solves each round's problem with a
strategy (the caller's, else `default_strategy`), hands `(problem, solution)` to the estimator,
and refines. `DesignOptimizer` owns a `SIMPModel` (a space, a `LinearElastic`
equation, and supports; `Equation.problem` resolved once as the template) and each iteration
derives the diluted `LinearProblem` from the current density with `with_operator`, solves it
through `SensitivityAnalysis`, and moves the density by the optimality-criteria update.

### Error estimation and sensitivity

`fem/estimators.py` provides the residual estimator (2D, straight-sided), the Zienkiewicz-Zhu
recovery estimator (dimension-general, curved elements included), and the goal-oriented estimator
(the product of primal and dual recovery indicators). Each takes `(problem, solution)` and reads
the `DerivedField` off the problem's operator, so none needs a physics argument.
`fem/sensitivity.py` computes `dJ/dp` for a `QuantityOfInterest` through one adjoint solve on the
forward factorization; `fem/design.py` drives an optimality-criteria update from that gradient.

### `Solution`

One dataclass per shape, each holding the `FunctionSpace` it was solved on: `FieldSolution` (the
field `u`), `ScalarFieldSolution` (adds the flux), `ElasticSolution` (adds strain, stress,
compliance), `TransientSolution` / `WaveSolution` (time series), `BucklingSolution` (adds the
prestress `reference` solve) / `ModalSolution`. `ElasticSolution` stores the full tensors and derives
`von_mises`, `pressure`, and `principal_stress` on demand; `nodal_stress` re-evaluates the form at
the nodes so a P2 stress keeps its within-element variation. `save` / `load` round-trip any of them
through `fem/io`.

## The recurring pattern

`fem/regions.py` + `fem/boundary.py` is the model: a mesh-independent specification
(`BoundaryConditions`) separated from its resolution against one discretization (`ResolvedBC`,
frozen, keyed by mesh and component count). The same shape recurs: `FunctionSpace` is the resolved
discretization, `Form` the resolved view of an `Equation`'s physics, `Problem` the resolved
composition, and `LinearSolver` a `Backend` resolved against one matrix.
