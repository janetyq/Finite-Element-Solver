# Finite Element Solver

[![CI](https://github.com/janetyq/Finite-Element-Solver/actions/workflows/ci.yml/badge.svg)](https://github.com/janetyq/Finite-Element-Solver/actions/workflows/ci.yml)
[![Demo gallery](https://img.shields.io/badge/demo-gallery-blue)](https://janetyq.github.io/Finite-Element-Solver/)

A finite element method (FEM) solver written from scratch in Python. It meshes a
domain, assembles the discrete system, and solves the Poisson, heat, wave, and both
linear and nonlinear elasticity equations, in 2D and 3D. Boundary conditions can be
Dirichlet, Neumann, or Robin, and are described geometrically so one specification
survives a remesh. On top of the core solves it carries custom meshing (Delaunay,
Ruppert's algorithm, red-green refinement), closed-loop adaptive refinement driven by
an a posteriori error estimator, higher-order (P2) elements, linearised buckling and
free-vibration (modal) analysis, and SIMP topology optimization.

### ▶ [See it running: the demo gallery](https://janetyq.github.io/Finite-Element-Solver/)

The [gallery](https://janetyq.github.io/Finite-Element-Solver/) renders every demo
beside the code that produced it, rebuilt on each push to `main`. The figures below are
a curated subset: a tour of what the solver does, with the full captions in the gallery.

---

## Meshing a domain

The library brings its own meshing. Any outline, traced from an SVG or generated, runs
through three algorithms: Douglas-Peucker to simplify a densely traced outline,
Ruppert's algorithm to triangulate it to a minimum-angle and maximum-area bound, and
adaptive refinement to add triangles where they improve accuracy (shown in later
demos). Below, four outlines carry the same solve, the Poisson "dome" of
$-\nabla^2 u = 1$ with $u = 0$ on the boundary. Each shape makes a different demand:
disconnected islands, true Bezier curves, a hole by the even-odd rule, sharp notches.

<p align="center"><img src="images/outline_to_mesh.png" height="400" alt="Four outlines meshed and Poisson-solved: California with a mesh zoom inset, a cloud, a gear, and a star"></p>

## Solving PDEs

### Poisson's equation

Poisson's equation, $-\nabla^2 u = f$, models heat transfer, electrostatics, and other
steady diffusion. The meshing section above solves its "dome"; the same operator on a
domain with a hole gives a flow. An ideal (incompressible, irrotational) flow has a
velocity potential $\phi$ with $\mathbf{v} = \nabla\phi$, so $\phi$ solves Laplace's
equation. The obstacle is a NACA 2412 airfoil at a 12-degree angle of attack. A
potential difference drives the flow left to right; the wing carries no condition at
all, which is the natural zero-flux condition of the weak form, so it becomes a
streamline the flow parts around. The equipotentials crowd over the upper surface,
where the flow speeds up.

<p align="center"><img src="images/potential_flow.png" width="780" alt="Potential flow: equipotentials and flow speed over a NACA airfoil"></p>

### Heat equation

The heat equation, $\partial u / \partial t = \alpha \nabla^2 u$, describes how
temperature spreads over time. It is integrated with the theta-method, defaulting to
$\theta = \tfrac{1}{2}$ (Crank-Nicolson); $\theta = 1$ is backward Euler. A finned
heatsink is held hot underneath its base, and every other surface sheds heat to ambient
through a convective (Robin) film, $\partial u / \partial n + \kappa (u - u_\infty) = 0$.

Compared against a solid block of the same size: driven by the same chip power, the
block runs about 108&deg;C above ambient while the finned sink runs 58&deg;C, roughly
halving the thermal resistance; held at the same base temperature, the finned sink
sheds about 1.8x the heat on two-thirds the metal. Each fin's efficiency (right), the
heat it sheds against what it would shed at the base temperature throughout, follows
the textbook $\tanh(mL)/(mL)$ law, falling as fins lengthen because a long fin runs
cold toward the tip.

<p align="center"><img src="images/heatsink_comparison.png" height="280" alt="Heatsink vs a solid block: fixed power (the block overheats) and fixed base temperature (the fins shed more)"> <img src="images/heatsink_efficiency.png" height="280" alt="Fin efficiency against the tanh(mL)/(mL) beam-theory law"></p>

### Wave equation

The wave equation, $\partial^2 u / \partial t^2 = c^2 \nabla^2 u$, is second order in
time, so it is integrated with Newmark's average-acceleration method rather than the
theta-method used for the first-order systems. A pulse released from rest spreads out,
reflects off the free boundary the same way up, and interferes with itself.

<p align="center"><img src="images/wave.png" height="400" alt="Wave reflection and interference, second half of the run"></p>

## Solids & structures

### Linear elasticity, in 2D and 3D

The linear elastic solver recovers displacement and a full stress tensor from applied
forces and boundary conditions. A cantilever is clamped on the left and pulled down over
the middle of the right edge; the bending stress is largest at the clamp and splits
tension above the neutral axis from compression below. The same assembly, element
hierarchy, and stress recovery run one dimension up: the 3D panel is a tetrahedral
cantilever under the same clamp-and-load, solved with an AMG-preconditioned
conjugate-gradient backend where a direct factorization's fill-in starts to hurt, and
drawn as its boundary surface.

<p align="center"><img src="images/linear_elastic.png" width="780" alt="Linear elasticity: a 2D cantilever and a 3D tetrahedral one under the same clamp-and-load"></p>

One solve, one stress tensor, several rotation-invariant questions: von Mises,
mean normal stress, the Tresca measure, and the largest tensile principal value are
each a reduction of the same tensor rather than a separate problem.

### Stress at a re-entrant corner, and why fillets exist

An L-bracket clamped at the top and pulled down at the tip concentrates stress at its
inner corner. A sharp re-entrant corner is a stress singularity: the exact elastic
stress there is infinite, so no mesh resolves it, and adaptive refinement into the
corner just keeps the computed peak climbing. A fillet removes the singularity, and
the peak settles on a finite value. Tracking the corner peak against mesh size (right)
shows both: the sharp corner climbs without bound (its "stress" is a property of the
mesh, not the part) and the fillet converges. This is why real parts round their
inner corners.

<p align="center"><img src="images/bracket.png" height="280" alt="L-bracket von Mises stress: sharp corner vs filleted"> <img src="images/bracket_singularity.png" height="280" alt="Corner stress peak vs mesh refinement: sharp climbs, fillet converges"></p>

### Three ways to solve the same stretch

The same clamped block is stretched three ways: a linear solve of $Ku = f$, the same
physics reached by Newton on the elastic energy that system is the stationary point of,
and a finite-strain (Green-Lagrange) solve. The first two agree in displacement to
machine precision; the third stiffens as the stretch grows, which small strain cannot.

<p align="center"><img src="images/elasticity_models.png" width="780" alt="Linear, energy-minimisation, and finite-strain solves of one stretch"></p>

### From an outline to a stress concentration

The one demo that runs the whole pipeline, in one row. A plate with a hole is meshed from
its outline, given roller and traction conditions (the rim is left traction-free, the
natural condition of the weak form), solved on curved quadratic elements, then adaptively
refined toward the stress at the rim. The stress crowds into the material either side of
the hole and relaxes to the applied value within about a diameter. Read at the rim nodes
it peaks at 3.03x the applied stress, against the classic Kirsch factor of 3 for a hole in
an infinite plate and Howland's 3.02 for a hole a tenth of this plate's width.

<p align="center"><img src="images/stress_concentration.png" width="780" alt="Refined mesh with conditions, the stress field, and the peak against the Kirsch factor"></p>

### Buckling analysis

A slender column under compression does not fail by crushing; it snaps sideways once
the load crosses a critical value. `BucklingSolver` finds that value and the shapes by
*linearised (eigenvalue) buckling*: a reference load sets up a prestress, the geometric
stiffness $K_g$ assembled from it competes with the elastic stiffness $K$, and the
generalized eigenproblem $K \phi = -\lambda K_g \phi$ gives the critical load factors
and mode shapes. The column is meshed with P2 elements, which do not lock in bending
the way a constant-strain triangle does.

<p align="center"><img src="images/buckling.png" height="450" alt="Buckling modes of a pinned-pinned column"></p>

### Modal (free-vibration) analysis

The same eigen-machinery, a different pencil. Free vibration solves
$K \phi = \omega^2 M \phi$ (no applied load; the modes are a property of the structure),
using the consistent mass matrix and a shift-invert about zero to pull the lowest
frequencies. A steel tuning fork is meshed from its own outline and held at the stem
base. Its low modes come in pairs; the one whose tines swing oppositely leaves the stem
still and rings, which is "the voice".

<p align="center"><img src="images/modal.png" width="700" alt="A tuning fork's natural modes and their pitches"></p>

### Topology optimization

Topology optimization distributes material to minimize compliance (deformation under
load). Here a simply supported beam carries a central load, and the SIMP (Solid Isotropic
Material with Penalization) method is asked for the stiffest structure using half the
material, penalizing intermediate densities so the design resolves toward solid-or-void.
It finds the classic arch: a compression arch over a tension tie, braced by a diagonal
web. Because compliance is the work the load does, it measures deflection directly, and
the optimized truss comes out only about 1.6x as compliant as the fully solid block on
half the material. What it removed was near the neutral axis, where the material was
barely resisting the bending.

<p align="center"><img src="images/topology_optimization.png" height="400" alt="Solid beam vs the optimized half-material arch, compared by compliance"></p>

The [gallery's topology page](https://janetyq.github.io/Finite-Element-Solver/topology_optimization.html)
plays the SIMP iterations frame by frame, from an even grey to the black-and-white truss.

## Accuracy & performance

### Convergence against manufactured solutions

The one place the solver plots not what it computed but how wrong it was. Against
exactly known (manufactured) solutions, P1 elements are second order in space (halve
$h$, quarter the error) for a scalar unknown and a coupled vector one alike; in time the
order is the theta-method's to choose, first at backward Euler and second at
Crank-Nicolson. Every rate here also runs as an assertion in the test suite.

<p align="center"><img src="images/convergence.png" width="780" alt="Convergence rates in space and time"></p>

### Higher-order elements

P2 (quadratic) triangles carry edge-midpoint DOFs that let the solution curve within an
element. On the same meshes they are third order in $L^2$ where P1 is second, and
reach a given accuracy with fewer degrees of freedom.

<p align="center"><img src="images/higher_order.png" width="780" alt="P1 vs P2 accuracy and cost"></p>

Curved (isoparametric) boundary elements take this a step further. On a boundary that
carries an analytic curve (a `Circle` or `Arc`), an `IsoparametricTriangleElement`
places its edge-midpoint node on the true curve and differentiates the full geometry
map, so the element's boundary edge bends to follow it instead of cutting a chord.
Meshing carries the curve through, so Ruppert's split points and red-green refinement
project onto it and a circular hole stays round under refinement rather than becoming a
finer polygon. On an annulus the meshed area then converges at the element's own order
rather than the polygonal $O(h^2)$, and a coarsely sampled hole, read at its rim node,
already carries most of the Kirsch stress concentration.

### Adaptive refinement

Adaptive refinement re-solves and splits wherever an a posteriori error estimator finds
the most error, keeping triangle quality with red-green refinement. Three estimators
ship: a residual estimator (interior residual, edge flux jump, boundary residual; 2D
only), a Zienkiewicz-Zhu recovery estimator (the gap between the discrete flux and a
recovered continuous one; dimension-general), and a goal-oriented estimator that
refines toward a chosen quantity of interest through an adjoint solve. Below, the
residual estimator on a peaked Poisson source concentrates the mesh where the solution
is hardest to approximate.

<p align="center"><img src="images/refinement.png" width="780" alt="Adaptive refinement on a peaked source"></p>

### Representation error

Before any PDE, there is the question of what the space can represent at all. The target
$\sin(40 r^2)$ has rings that tighten with radius; projected onto a deliberately coarse
P1 mesh, the slow inner rings come through but the fast outer ones break up into the
triangulation. That representation error is the floor every solver on this mesh starts
from, and refining the mesh is what lowers it.

<p align="center"><img src="images/l2_projection.png" width="780" alt="The target sin(40 r^2) beside its L2 projection onto a coarse P1 mesh"></p>

---

## Quick Start

```python
from fem import create_rect_mesh, BoundaryConditions, BCType, Solver, Poisson, Plotter
from fem.regions import everywhere

mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40))

# Conditions are geometric, so the same `bc` is valid on any mesh of this domain.
equation = Poisson(source=1)
bc = BoundaryConditions()
bc.add(BCType.DIRICHLET, everywhere(), 0)

solution = Solver(mesh, equation, bc).solve()

plotter = Plotter(title="Poisson")
plotter.plot(mesh, solution.u, mode="surface")
plotter.show()
```

A solution is a typed dataclass, so its fields are attributes rather than string keys.
An elastic solve returns an `ElasticSolution`, which carries the recovered stress and
strain as full tensors and derives the scalar measures on demand:

```python
solution = Solver(mesh, LinearElastic(E=200, nu=0.3), bc).solve()

solution.u                 # (n_vertices * n_components,) displacement
solution.stress            # (n_elements, 3, 3) Cauchy stress tensors
solution.von_mises         # (n_elements,) equivalent stress, the usual plot
solution.principal_stress  # (n_elements, 3) principal values, ascending
solution.compliance        # (n_elements,) strain energy per element
```

The tensors are stored and the scalars computed on demand. `fem/invariants.py` holds
those reductions; each is rotation-invariant.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. `uv sync` creates a project-local `.venv`, installs the `fem` package in
editable mode, and pins exact versions in `uv.lock`.

```bash
uv sync   # core solver, SVG-outline and 3D tetrahedral meshing, and dev tools (pytest)
```

Prefer plain pip? It is a standard `pyproject.toml` package:
```bash
pip install -e .
```

## Running demos and tests

Runnable demos live in `examples/` (run from the repo root) behind a small CLI:
```bash
uv run python examples/cli.py list          # see every available demo
uv run python examples/cli.py run poisson   # run one by name
uv run python examples/cli.py gallery       # render them all as a browsable site
```

The test suite:
```bash
uv run pytest
```
`uv run ruff check` and `uv run pyright` gate CI alongside the tests.

## Project Structure

The package is grouped by the job each object does. `ARCHITECTURE.md` is the overview;
this is the map.

```
fem/                 # the solver package
├── mesh/            # Mesh geometry, generation, red-green refinement, SVG outlines
├── plot/            # Plotter, 2D drawing helpers, 3D tet rendering
│
│   # discretization: what functions can be represented
├── elements.py      # stateless element types (P1/P2) + batched ElementGeometry
├── quadrature.py    # reference-simplex Gauss rules, wired into assembly
├── space.py         # FunctionSpace: DOF numbering, cached operators, assembly
│
│   # physics: what equation, what material
├── equations.py     # Equation: Projection, Poisson, LinearElastic
├── forms.py         # Form/EnergyForm integrands; derived-field recovery
├── materials.py     # Hooke's law, Lame conversions (2D is plane strain)
├── energies.py      # hyperelastic strain-energy densities and their derivatives
├── fields.py        # Scalar/Vector: components per node, resolved against the mesh
│
│   # constraints
├── boundary.py      # BoundaryConditions spec -> ResolvedBC for one mesh
├── regions.py       # position-based regions and fields (on_plane, in_box, ...)
│
│   # composition and algebra
├── problem.py       # Problem: space + operator + load + constraints
├── solve.py         # LinearSolve / NewtonSolve / EigenSolve strategies
├── system.py        # DiscreteSystem: Dirichlet elimination, factor once
├── backends.py      # DirectBackend (sparse LU) / IterativeBackend (AMG-CG)
├── integrators.py   # ThetaMethod (1st order), NewmarkMethod (2nd order)
│
│   # facades and drivers
├── solver.py        # Solver: the steady linear facade
├── energy_solver.py # EnergySolver: the nonlinear facade
├── buckling.py      # BucklingSolver: linearised (eigenvalue) buckling
├── modal.py         # ModalSolver: free-vibration analysis
├── adaptivity.py    # AdaptiveRefinement driver
├── estimators.py    # residual, recovery, and goal-oriented error estimators
├── sensitivity.py   # adjoint sensitivity: quantities of interest and their gradients
├── topology.py      # SIMP topology optimization driver
├── design.py        # DesignOptimizer: any quantity of interest over a density field
│
│   # results
├── solution.py      # typed Solution hierarchy; ElasticSolution.from_solve
├── invariants.py    # rotation-invariant tensor reductions (von Mises, ...)
├── convergence.py   # manufactured-solution studies and error norms
├── io.py            # mesh JSON, solution npz (no pickle)
│
└── geometry.py, numerics.py, typing.py   # helpers and semantic array aliases
tests/               # pytest suite
examples/            # runnable demo scripts, the CLI, and the gallery builder
files/               # example SVG outlines
```

The figures in this README are committed PNGs, a curated subset of the gallery's
renders, refreshed by hand rather than in CI. After changing a demo, regenerate them
and commit the result:

```bash
uv run python examples/make_readme_figures.py   # rewrites the figures in images/
```

## Methods

- Galerkin finite element method: P1 (linear) basis on triangles and tetrahedra, P2
  (quadratic) and curved isoparametric triangles, over a Gaussian quadrature layer
- Boundary conditions: Dirichlet, Neumann, Robin, and per-component (roller) constraints
- PDEs: L2 projection, Poisson, variable-coefficient diffusion, heat, wave,
  Navier-Cauchy (linear elasticity), St Venant-Kirchhoff hyperelasticity
- Kinematics: infinitesimal strain, or geometrically exact Green-Lagrange
  (2D elasticity is plane strain)
- Time integration: theta-method (backward Euler, Crank-Nicolson) for first-order
  systems; Newmark average-acceleration for second-order
- Linear algebra: sparse throughout; direct (`splu`) by default, or AMG-preconditioned
  CG for large SPD systems, with a rigid-body near-kernel for elasticity
- Eigen-analysis: linearised (eigenvalue) buckling and free-vibration (modal) analysis,
  a geometric-stiffness or mass pencil and a sparse generalized eigensolve
- Derived fields: Cauchy stress and strain tensors, von Mises, principal stresses,
  compliance
- Error estimation: residual, Zienkiewicz-Zhu recovery, and goal-oriented (dual-weighted
  residual) estimators, driving closed-loop adaptive refinement
- Sensitivity and optimization: adjoint gradients of a quantity of interest;
  Newton-Raphson with an optional backtracking line search; optimality criteria (SIMP
  topology and design optimization)
- Mesh algorithms: Delaunay triangulation, Ruppert's algorithm (line segments to
  triangle mesh), red-green refinement

## Roadmap

`BACKLOG.md` tracks the detailed open work; this is the direction.

- Broaden the physics: thermoelasticity, plasticity, fluids (Stokes / Navier-Stokes),
  electrostatics, advection-diffusion, Neo-Hookean
- Inverse problems and shape optimization on the adjoint core
- Standard formats: STL and OBJ meshes, Gmsh `.msh` import, VTK/ParaView `.vtu` export
- Standard benchmark suite: NAFEMS, Cook's membrane, plate-with-hole, L-shaped
  singularity, Euler columns
- Finish the core: 3D P2 with P2-aware plotting and adaptivity, mixed (u-p) for
  near-incompressibility, time-varying loads, two-grid preconditioner

## References

*The Finite Element Method: Theory, Implementation, and Applications* by Mats G. Larson
and Fredrik Bengzon.

[*SIMP Method for Topology Optimization*](https://help.solidworks.com/2019/english/solidworks/cworks/c_simp_method_topology.htm)
by Dassault Systèmes.
