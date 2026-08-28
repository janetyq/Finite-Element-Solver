# Architecture

An overview of the object model: which concepts exist, which object owns each job, and how they
fit together. Anchored on symbol names rather than line numbers. Open work lives in `BACKLOG.md`.

## The idea

A solve is a composition assembled from parts, not a method looked up by PDE. The package has the
parts (`FunctionSpace`, `Form`, `Material`, `DiscreteSystem`, `ResolvedBC`), the object that holds
a composition (`Problem`), the strategies that consume one (`LinearSolve`, `NewtonSolve`, the time
integrators), and the drivers that wrap a strategy to re-solve (`AdaptiveRefinement`,
`DesignOptimizer`). A transient problem is a steady operator paired with a time integrator, not a
PDE type at the `Problem` level. An `Equation` names the PDE, its physical constants, and the
time-derivative orders it has a meaning for (`time_orders`), which the solves check. "What to
solve" is the `Problem`; "how" is the strategy.

## Building a solve

The chain is the same for every problem. Each step has one required object and a few options with
defaults; a facade is this chain with the defaults filled in from an `Equation`.

| Step | Required | Options (default) |
|---|---|---|
| Geometry | `Mesh` | |
| Discretization | `FunctionSpace(mesh, n_components)`, or `Equation.space(mesh)` | `element_type` (linear) |
| Physics | a `Form`, or `Equation.operator` | a `Material` for elasticity; an `EnergyForm` over a strain-energy density for finite strain |
| Statement | `Problem(space, form)`, or `Equation.problem(mesh, bc)` (which takes the Discretization step too) | `source` (none), `bc` (none), `element_type`; a `LinearProblem` when the form has a constant tangent |
| Solve | `problem.solve()`: `LinearSolve` for a constant tangent, else `NewtonSolve`; or `.solve(problem, ...)` on an integrator or analysis | `strategy`, `Backend` (direct); Newton: `line_search`, `regularization`; integrator: `dt`, `steps`, `theta` / `beta`, initial data from `space.interpolate` |
| Result | a typed `Solution`, returned by every `solve` | |
| Outer loop | | `AdaptiveRefinement` over a `problem_for(mesh)` builder; `DesignOptimizer` over a `SIMPModel` |

Composed by hand:

```python
space = FunctionSpace(mesh, n_components=1)
problem = LinearProblem(space, DiffusionForm(), source=1.0, bc=bc)
solution = problem.solve(backend=IterativeBackend())
```

The same solve from the named equation, which builds the space and the problem:

```python
solution = Poisson(source=1.0).problem(mesh, bc).solve(backend=IterativeBackend())
```

The two agree exactly: `Equation.problem` and `Problem.solve` hold no policy of their own, so
anything they can do can be composed, and anything composed (a different form, a hand-built load,
a custom strategy) needs neither. `Solver(mesh, equation, bc)` is that second line held as an object.

The load is a sum of `Load` terms (`fem/loads.py`), each answering `vector(space, t)`. The volume
source is a `Source`: a constant or a nodal array is integrated exactly through the mass matrix, a
callable is sampled at the quadrature points, which captures variation within an element
(`NodalSource` is the explicit interpolant path); a Neumann value or a Robin value is a
`BoundaryLoad` over its region's facets; a nodal force is a `PointLoad`, passed through `Equation(loads=...)` or `Problem(loads=...)`. A DOF vector built by
hand (an initial condition, a comparison field) is `space.interpolate(value)`, nodal on P2 as well.

Boundary conditions are objects: `Dirichlet(region, value)`, `Neumann(region, value)`, and
`Robin(region, kappa, g)`, collected by `BoundaryConditions(*conditions)` or `bc + condition`,
both frozen. Each condition resolves itself against a node set (`condition.resolve`), and a
`Dirichlet` value may leave a component `None` (free) for a roller.

Operators compose the same way: `a + b` is a `SumForm` and `c * a` a `ScaledForm`, and each form
names the `domain` it integrates over (the elements or the boundary facets), so a Robin condition
is `kappa * BoundaryMassForm(mask)` added to the physics form and assembled beside it.

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
| 1. Primitives | the parts a composition is built from | `Form` (the `BilinearForm`s, `EnergyForm`; combinators `SumForm`, `ScaledForm`; `BoundaryMassForm`, `PrecomputedForm`), `Load` (`Source`, `NodalSource`, `BoundaryLoad`, `PointLoad`), `Material`, `FunctionSpace`, `BoundaryConditions` / `ResolvedBC`, `DiscreteSystem` + `Backend` |
| 2. `Problem` | a composition: space + operator + load + constraints | `Problem`, and `LinearProblem` for a constant tangent; `Equation.problem` builds one from a named PDE |
| 3. Solve strategy | consumes a `Problem`, returns the solution | `LinearSolve`, `NewtonSolve`, `EigenSolve`; integrators `ThetaMethod`, `NewmarkMethod` |
| 4. Driver | wraps a strategy, re-solving | `AdaptiveRefinement`, `DesignOptimizer` |

Tier 3 has a second, orthogonal axis: the strategy picks linear vs. Newton, a `Backend` picks
direct vs. iterative. Named PDEs are `Equation`s, not dispatch keys: `Poisson(f).problem(space, bc)`
returns `LinearProblem(space, DiffusionForm(), f, bc)`, `FiniteStrainElastic(E, nu).problem(space,
bc)` a `Problem` over an `EnergyForm`, and a PDE with no name is just a different composition.

A choice is a parameter when it changes numbers inside one computation (a modulus, a density,
plane stress against plane strain) and a class when it changes what the object is: which solve
runs, what it returns, what it composes with. `LinearElastic` and `FiniteStrainElastic` are two
classes for that reason, as are the backends, the estimators, and `Problem` / `LinearProblem`.

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
| `Solver` | composition only: `Equation.problem` then `Problem.solve` | | | | | | | | |
| `BucklingAnalysis`, `ModalAnalysis` | | | | | | ▒ | | | ▒ |
| `AdaptiveRefinement`, `SIMPModel` / `DesignOptimizer` | | | | | | | | █ | ▒ |
| Error estimators, `SensitivityAnalysis` | | | | | | | | ▒ | █ |
| `Solution` (typed) | | | | | | | ▒ | | █ |
| `invariants`, `Plotter` / `fem/plot`, `io` | | | | | | | | | █ |
| `convergence` (MMS studies), `numerics` | | | | | | | | | ▒ |

Constraints (5) have one owner, the `Problem`, whose constructor resolves the boundary conditions
against the space and turns the Dirichlet partition into its `constraints`, each Neumann and Robin
value into a `BoundaryLoad` load term, and each Robin coefficient into a boundary term of the operator. Algebra (6) is split: `DiscreteSystem` owns the Dirichlet elimination
and the `Backend` owns how the free-free block is solved. Physics (3) is owned by the physics
layer alone: an `Equation` answers `operator` itself, so no facade holds a mapping from equation
to material.

Post-processing (9) is distributed under one rule: a derived quantity lives on the object that
owns the data it needs. `FunctionSpace` owns `integrate`, `recover_nodal`, and `nodal_gradient`;
a `Form` owns `fields_at` / `derived_fields` and `derived_field` (which flux is recoverable, read
by the estimators off `problem.operator`); `Problem.solve` picks the typed `Solution` for its
operator (`solution(u)` packages a vector solved elsewhere); `Solution` owns the packaging (`ElasticSolution.stress`, `nodal_stress`,
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

Every assembly path goes through a form. Bilinear forms (`MassForm`, `DiffusionForm`, `LinearElasticForm`, `GeometricStiffnessForm`) follow `Gᵀ C G` with the element
supplying `G` and the form `C`; `SumForm` and `ScaledForm` are the combinators (`a + b`, `c * a`:
the wave operator's `T K`, the Robin boundary term `kappa * BoundaryMassForm(mask)`, which names the
boundary as its `domain`); `PrecomputedForm` lets a driver reuse element
matrices it can derive more cheaply (SIMP rescales one set by `rho^p`). Each is a `BilinearForm`,
which supplies the residual `K u` and the energy `½ uᵀ K u` from its matrix. `EnergyForm` is the
state-dependent form, mapping an element and a state to energy, residual, and tangent; both kinds
assemble through `assemble_residual` / `assemble_tangent` / `total_energy`, at the larger of the
element's default rule and the one the form asks for.

`Material` owns the constitutive matrix `D`; the strain-displacement matrix `B` sits in
`fem/forms.py` next to the form that contracts it. An `EnergyDensity` returns the one thing an
`EnergyForm` contracts: the energy `W`, the first Piola-Kirchhoff stress `P = dW/dF`, and the
material tangent `A = d²W/dF²`, all in F. How it gets there is the density's own business:
`SmallStrain` and `StVenantKirchhoff` build the chain through a strain measure (small-strain `ε`
or Green-Lagrange `S`), while `NeohookeanEnergyDensity` is written in the invariants of `C = FᵀF`
and writes `P` and `A` directly. The equation names the model: `LinearElastic` gives the constant
stiffness `LinearElasticForm`, `FiniteStrainElastic` the `EnergyForm` over its `law`
(`StVenantKirchhoff` by default). In 2D the law is plane strain throughout.

Stress recovery is on the form (`RecoversElasticFields`): `fields_at` gives strain and stress at
every point of a geometry's rule, `derived_fields` reduces that to one tensor per element. Full
`(n_elements, 3, 3)` tensors cross the boundary, never Voigt vectors, and `fem/invariants.py`
reduces them to frame-independent scalars.

### `Problem`: the narrow waist

A `Problem` is the assembly-ready composition for one mesh: space, operator, load terms,
constraints. `physics` is the form it was stated with and `operator` that form plus one
`kappa * BoundaryMassForm` term per Robin condition; `loads` is the tuple of `Load` terms (the
source, one `BoundaryLoad` per Neumann condition and per Robin value, any `PointLoad`). Its residual
has two terms, each present in `energy`, `residual`, and `tangent` alike: the operator's (`Π`,
`R`, `∂R/∂u`, summed over the operator's terms by the space) and the load's (`−fᵀu`, `−f`, `0`).
`internal_residual` is the first, kept apart from `load` so a strategy can scale one against the
other. `mass` is the problem's mass side, the equation's `density` times the space's consistent
mass matrix, and `damping_matrix` the `RayleighDamping` `αM + βK` when the equation carries one,
both assembled once for the integrators and the modal analysis. `LinearProblem` is the case whose operator has a
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
constant effective operator from `problem.mass`, `problem.tangent()`, and (for `NewmarkMethod`)
`problem.damping_matrix`, factors it once, and steps by updating the right-hand side. A `TimeDependent` source or boundary value (a callable of
position and time) is re-evaluated each step through `problem.load_at(t)`; `ThetaMethod` also
prescribes a time-dependent Dirichlet value per step through `problem.constraints_at(t)`, while
`NewmarkMethod` refuses one (prescribed motion needs its velocity and acceleration too). Both
return a `TransientSolution` whose `at(i)` packages a step as the typed steady solution. A
steady solve or an estimator works on the snapshot `problem.at(t)`; `problem.solve(t=...)`
takes that step itself. `ThetaMethod` (Crank-Nicolson by default, backward Euler at θ=1)
and `NewmarkMethod` (average acceleration, solving for the acceleration against the SPD
`M + β dt² K`) both take a `Backend`.

### `Equation`

`Equation` is typed data: `Projection`, `Poisson`, `Heat`, `Wave`, `LinearElastic`, and
`FiniteStrainElastic` (the last two over the `Elasticity` base), each carrying its physical
constants, a `density` for the time-derivative term where it has one, and `time_orders`, the
time-derivative orders the PDE has a meaning for (`Poisson` {0}, `Heat` {1}, `Wave` {2}, the
elastic equations {0, 2}). The three scalar equations share the `DiffusionForm` operator and
differ in their orders and their constants' names; `Problem.solve`, the integrators, and
`ModalAnalysis` refuse an order the equation lacks, naming the equation to use. `operator(space)`
returns the form for its physics: the small-strain stiffness, or the `EnergyForm` of a
finite-strain law. It refuses rather than approximates when the physics does not apply. The table
in `fem/equations.py` maps each PDE to its class and the solve that steps it.
Two more resolve it against a discretization: `space(mesh, element_type)` builds the
`FunctionSpace` with the component count the field implies, and `problem(mesh_or_space, bc,
element_type)` the `LinearProblem` for a constant tangent, else a `Problem`. Every facade and
driver goes through these two.

### Facades and drivers

`Solver` holds an equation, a BC spec, and the space on a mesh; each `solve` is
`equation.problem(space, bc).solve(strategy, backend)`. It holds no solve policy and no state
between solves.

Two drivers, each over one spec. `AdaptiveRefinement` owns a mesh and a `problem_for(mesh)`
builder (`equation.problem(mesh, bc)`), solves each round's problem with `Problem.solve` (the
caller's strategy, else the default), hands `(problem, solution)` to the estimator,
and refines. `DesignOptimizer` owns a `SIMPModel` (a small-strain elastic `LinearProblem` as
the template, whose material, supports, and load every density shares) and each iteration
derives the diluted `LinearProblem` from the current density with `with_operator`, solves it
through `SensitivityAnalysis`, and moves the density by the optimality-criteria update.

### Extension seams

Each seam is a protocol, exported from `fem`, and the classes beside it are the implementations
to copy: `Form` (`BilinearForm` by `element_matrices`, `EnergyForm` by an `EnergyDensity`);
`SolveStrategy` (`LinearSolve`, `NewtonSolve`); `Backend` and `LinearSolver` (`DirectBackend`,
`IterativeBackend`, `MinresBackend`); `ErrorEstimator` (the three estimators, or any callable of
`(problem, solution)`); `QuantityOfInterest` and `Parameterization` (`Compliance`, `PointValue`,
`DensityField`, `ModulusField`); `DerivedField` (`GradientField`, `StressField`); `FieldShape`
(`Scalar`, `Vector`).

### Error estimation and sensitivity

`fem/estimators.py` provides the residual estimator (2D, straight-sided), the Zienkiewicz-Zhu
recovery estimator (dimension-general, curved elements included), and the goal-oriented estimator
(the product of primal and dual recovery indicators). An estimator has no physics of its own: each
takes `(problem, solution)` and reads the `DerivedField` off the problem's operator and the source
off the problem. A custom estimate is any callable of the same two arguments.
`fem/sensitivity.py` computes `dJ/dp` for a `QuantityOfInterest` through one adjoint solve on the
forward factorization; `fem/design.py` drives an optimality-criteria update from that gradient.

### `Solution`

One dataclass per shape, each holding the `FunctionSpace` it was solved on: `FieldSolution` (the
field `u`), `ScalarFieldSolution` (adds the flux), `ElasticSolution` (adds strain, stress,
compliance), `TransientSolution` / `WaveSolution` (time series; `at(i)` is one step as a steady
solution, flux or stress included), `BucklingSolution` (adds the
prestress `reference` solve) / `ModalSolution`. `ElasticSolution` stores the full tensors and derives
`von_mises`, `pressure`, and `principal_stress` on demand; `nodal_stress` re-evaluates the form at
the nodes so a P2 stress keeps its within-element variation. `save` / `load` round-trip any of them
through `fem/io`.

## The recurring pattern

`fem/regions.py` + `fem/boundary.py` is the model: a mesh-independent specification
(`BoundaryConditions`, a frozen tuple of `Condition`s) separated from its resolution against one
discretization (`ResolvedBC`, frozen, keyed by node set and component count), with the
time-dependent values a second, cheaper step on the resolution (`ResolvedBC.at(t)`). The same
shape recurs: `FunctionSpace` is the resolved
discretization, `Form` the resolved view of an `Equation`'s physics, `Problem` the resolved
composition, and `LinearSolver` a `Backend` resolved against one matrix.
