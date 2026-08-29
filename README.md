# Finite Element Solver

[![CI](https://github.com/janetyq/Finite-Element-Solver/actions/workflows/ci.yml/badge.svg)](https://github.com/janetyq/Finite-Element-Solver/actions/workflows/ci.yml)
[![Demo gallery](https://img.shields.io/badge/demo-gallery-blue)](https://janetyq.github.io/Finite-Element-Solver/)

A finite element method (FEM) solver written from scratch in Python. It meshes a
domain, assembles the discrete system, and solves the Poisson, heat, wave, and linear
and nonlinear elasticity equations in 2D and 3D. Boundary conditions can be Dirichlet,
Neumann, or Robin, and are described geometrically so one specification survives a
remesh. It also includes its own meshing (Delaunay, Ruppert's algorithm, red-green
refinement), adaptive refinement driven by a posteriori error estimators, quadratic
and curved elements, buckling and modal analysis, adjoint sensitivities, and SIMP
topology optimization.

### ▶ [See it running: the demo gallery](https://janetyq.github.io/Finite-Element-Solver/)

The [gallery](https://janetyq.github.io/Finite-Element-Solver/) renders every demo
beside the code that produced it and is rebuilt on each push to `main`. The figures
below are a subset of it, with the full captions in the gallery.

---

## Meshing a domain

The library includes its own meshing. An `Outline` of lines, arcs, circles, and Bezier
curves, drawn by hand or traced from an SVG, is simplified with Douglas-Peucker and
triangulated with Ruppert's algorithm (`outline.mesh(min_angle=..., max_area_fraction=...)`)
to a minimum-angle and maximum-area bound, and can then be adaptively refined where the solution needs it (shown in later
demos). Each boundary facet of the result is tagged with the outline it came from, so a
hole can take a condition by `on_tag(1)` rather than by coordinates. Below, the same Poisson problem
($-\nabla^2 u = 1$ with $u = 0$ on the boundary) is solved on four outlines. Each asks
something different of the mesher: disconnected islands, Bezier curves, a hole, and
sharp notches.

<!-- Figures: HTML img inside <p align="center">, not markdown images, so they can be sized and paired.
     Compact figures get height (about 400); wide ones get width (about 780) since a tall height on a wide
     image makes GitHub squish it. Side-by-side pairs share one <p> at a height that fits both in the column.
     PNGs come from examples/make_readme_figures.py. -->

<p align="center"><img src="images/outline_to_mesh.png" height="400" alt="Four outlines meshed and Poisson-solved: California with a mesh zoom inset, a cloud, a gear, and a star"></p>

## Solving PDEs

### Poisson's equation

Poisson's equation, $-\nabla^2 u = f$, models heat transfer, electrostatics, and other
steady diffusion. The meshing section above solves it with a constant source; with
no source it is Laplace's equation, which here gives a flow. An ideal (incompressible,
irrotational) flow has a velocity potential $\phi$ with $\mathbf{v} = \nabla\phi$, and
$\phi$ solves Laplace's equation. The obstacle is a NACA 2412 airfoil at a 12-degree
angle of attack. A potential difference drives the flow left to right. The wing
carries no boundary condition at all, which in the weak form is the natural zero-flux
condition, so it becomes a streamline the flow parts around. The equipotentials crowd
over the upper surface, where the flow speeds up.

<p align="center"><img src="images/poisson.png" width="780" alt="Potential flow: equipotentials and flow speed over a NACA airfoil"></p>

### Heat equation

The heat equation, $\partial u / \partial t = \alpha \nabla^2 u$, describes how
temperature spreads over time. It is integrated with the theta-method, defaulting to
$\theta = \tfrac{1}{2}$ (Crank-Nicolson); $\theta = 1$ is backward Euler. A finned
heatsink is held hot underneath its base, and every other surface sheds heat to ambient
through a convective (Robin) film, $\partial u / \partial n + \kappa (u - u_\infty) = 0$.

The finned sink is compared against a solid block of the same size. Driven by the
same chip power, the block runs about 108&deg;C above ambient while the finned sink
runs 58&deg;C, roughly halving the thermal resistance. Held at the same base
temperature, the finned sink sheds about 1.8x the heat with two-thirds the metal. The
fin efficiency (right), the heat a fin sheds relative to what it would shed at the base
temperature throughout, follows the textbook $\tanh(mL)/(mL)$ law and falls as fins
lengthen, since a long fin runs cold toward the tip.

<p align="center"><img src="images/heatsink_comparison.png" height="280" alt="Heatsink vs a solid block: fixed power (the block overheats) and fixed base temperature (the fins shed more)"> <img src="images/heatsink_efficiency.png" height="280" alt="Fin efficiency against the tanh(mL)/(mL) beam-theory law"></p>

### Wave equation

The wave equation, $\partial^2 u / \partial t^2 = c^2 \nabla^2 u$, is second order in
time and is integrated with Newmark's average-acceleration method. Below, a wave
front meets a harbor breakwater: it reflects off the wall and passes the gap, where it
spreads into the sheltered water as a circular wave centred on the opening.

<p align="center"><img src="images/wave.gif" width="600" alt="A wave front diffracting through a breakwater gap"></p>

## Solids & structures

### Linear elasticity, in 2D and 3D

The linear elastic solver computes the displacement and the full stress tensor from
applied loads and boundary conditions. A cantilever is clamped on the left and pulled
down over the middle of the right edge. The bending stress is largest at the clamp,
with tension above the neutral axis and compression below. The 3D panel is a
tetrahedral cantilever under the same clamp and load, drawn as its boundary surface.
It is solved with AMG-preconditioned conjugate gradients, since in 3D a direct
factorization's fill-in starts to hurt.

<p align="center"><img src="images/linear_elastic.png" width="780" alt="Linear elasticity: a 2D cantilever and a 3D tetrahedral one under the same clamp-and-load"></p>

One solve gives one stress tensor, and the four 2D stress panels are
rotation-invariant reductions of it: von Mises, mean normal stress, the Tresca
measure, and the largest tensile principal value.

### Stress at a re-entrant corner, and why fillets exist

An L-bracket clamped at the top and pulled down at the tip concentrates stress at its
inner corner. A sharp re-entrant corner is a stress singularity, where the exact
elastic stress is infinite, so no mesh resolves it and adaptive refinement into the
corner keeps the computed peak climbing. A fillet removes the singularity and the
peak settles on a finite value. The plot on the right tracks the corner peak against
mesh size for both. The sharp corner climbs without bound, so the stress it reports
is a property of the mesh rather than the part, while the fillet converges. This is
why real parts round their inner corners.

<p align="center"><img src="images/bracket.png" height="280" alt="L-bracket von Mises stress: sharp corner vs filleted"> <img src="images/bracket_singularity.png" height="280" alt="Corner stress peak vs mesh refinement: sharp climbs, fillet converges"></p>

### Linear and nonlinear materials, one stretch

The same clamped block is stretched to the right in four ways, each panel coloured by
how hard the material is working inside. The first two are ordinary small-strain
elasticity, the familiar model where stress grows in proportion to strain, solved two
different ways: directly, and by minimising the block's stored elastic energy. They
settle into the same shape to machine precision, but their colour differs because they
report stress differently: the first uses engineering stress, force over the block's
original cross-section, the second the true stress on the stretched, now-thinner
cross-section, which reads higher. The last two switch to nonlinear elasticity, which
large deformations actually need. St-Venant-Kirchhoff is the simplest such model but
stiffens far too aggressively in tension, while Neo-Hookean, a rubber-like law, stays
realistic out to large stretch.

All four panels share one colour scale. The stresses span a wide range, so the scale is
logarithmic: brighter means more stress, and a whole panel shifting brighter (as
St-Venant-Kirchhoff's does) means that material is carrying more everywhere.

<p align="center"><img src="images/elasticity_models.png" width="780" alt="One stretch, four elasticity models coloured by stress on a shared log scale: two small-strain solves, St-Venant-Kirchhoff, and Neo-Hookean"></p>

### From an outline to a stress concentration

This demo runs the whole pipeline. A plate with a hole is meshed from its outline,
given roller and traction conditions (the rim is left traction-free), solved on curved
quadratic elements, and adaptively refined toward the stress at the rim. The stress
crowds into the material either side of the hole and relaxes to the applied value
within about a diameter. At the rim it peaks at 3.03x the applied stress. The classic
Kirsch factor is 3 for a hole in an infinite plate, and Howland's value for a hole a
tenth of this plate's width is 3.02.

<p align="center"><img src="images/stress_concentration.png" width="780" alt="Refined mesh with conditions, the stress field, and the peak against the Kirsch factor"></p>

### Buckling analysis

A slender column under compression does not fail by crushing. It snaps sideways once
the load crosses a critical value. `BucklingAnalysis` finds that value and the buckled
shapes by linearised (eigenvalue) buckling. A reference load sets up a prestress, the
geometric stiffness $K_g$ is assembled from it, and the generalized eigenproblem
$K \phi = -\lambda K_g \phi$ gives the critical load factors and mode shapes. The
column is meshed with P2 elements, which do not lock in bending the way a
constant-strain triangle does.

<p align="center"><img src="images/buckling.png" height="450" alt="Buckling modes of a pinned-pinned column"></p>

### Modal (free-vibration) analysis

Free vibration solves $K \phi = \omega^2 M \phi$ with the consistent mass matrix, using
shift-invert about zero to find the lowest frequencies. No load is applied, so the
modes are a property of the structure alone. A steel tuning fork is meshed from its
outline and held at the stem base. Its low modes come in pairs, and the one whose
tines swing in opposite directions leaves the stem still and rings; that is the
fork's voice.

<p align="center"><img src="images/modal.png" width="700" alt="A tuning fork's natural modes and their pitches"></p>

### Topology optimization

Topology optimization distributes material to minimize compliance (deformation under
load). Here a simply supported beam carries a central load, and the SIMP (Solid
Isotropic Material with Penalization) method finds the stiffest structure using half
the material, penalizing intermediate densities so the design resolves toward solid or
void. It finds the classic arch, a compression arch over a tension tie braced by a
diagonal web. Compliance is the work the load does, so it measures deflection
directly, and the optimized truss is only about 1.6x as compliant as the fully solid
block on half the material. What it removed was near the neutral axis, where the
material was barely resisting the bending.

<p align="center"><img src="images/topology_optimization.png" height="400" alt="Solid beam vs the optimized half-material arch, compared by compliance"></p>

The [gallery's topology page](https://janetyq.github.io/Finite-Element-Solver/topology_optimization.html)
plays the SIMP iterations frame by frame, from an even grey to the black-and-white truss.

## Accuracy & performance

### Convergence against manufactured solutions

Against exactly known (manufactured) solutions, P1 elements are second order in space
(halve $h$, quarter the error) for both a scalar unknown and a coupled vector one, and
P2 elements third. In time, backward Euler is first order and Crank-Nicolson second.
How the load is built
matters too: sampling the source at the quadrature points instead of reading it at
the vertices keeps the rate and improves the constant about 3x. Every rate here also
runs as an assertion in the test suite.

<p align="center"><img src="images/convergence.png" width="780" alt="Convergence rates in space and time, P1 against P2, and the load built two ways"></p>

### Higher-order elements

P2 (quadratic) triangles carry edge-midpoint DOFs that let the solution curve within an
element. On the same meshes they are third order in $L^2$ where P1 is second (top
left above), and reach a given accuracy with fewer degrees of freedom (top right).

Curved (isoparametric) boundary elements go a step further. On a boundary that
carries an analytic curve (a `Circle` or `Arc`), an `IsoparametricTriangleElement`
places its edge-midpoint node on the true curve, so the element's boundary edge
follows the curve instead of cutting a chord. Meshing carries the curve through, so
Ruppert's split points and red-green refinement project onto it and a circular hole
stays round under refinement. The meshed area then converges at the element's own
order rather than the polygonal $O(h^2)$; the hole-in-plate and bracket demos above
run on these elements.

### Adaptive refinement

Adaptive refinement re-solves and splits wherever an a posteriori error estimator finds
the most error, keeping triangle quality with red-green refinement. There are three
estimators. The residual estimator measures the interior residual, the flux jump
across edges, and the boundary residual (2D only). The Zienkiewicz-Zhu recovery
estimator measures the gap between the discrete flux and a recovered continuous one.
The goal-oriented estimator refines toward a chosen quantity of interest through an
adjoint solve. Below, the residual estimator on a peaked Poisson source concentrates
the mesh where the solution is hardest to approximate, and reaches a given error with
about a third of the unknowns uniform refinement needs (the gallery has the chart).

<p align="center"><img src="images/refinement.png" width="780" alt="Adaptive refinement on a peaked source"></p>

### Representation error

Before any PDE is solved, the mesh already limits what its P1 space can represent.
The target $\sin(40 r^2)$ has rings that tighten with radius. Projected onto a coarse
P1 mesh, the slow inner rings come through but the fast outer ones break up into the
triangulation. This representation error is the floor every solver on this mesh
starts from, and refining the mesh is what lowers it.

<p align="center"><img src="images/l2_projection.png" width="780" alt="The target sin(40 r^2) beside its L2 projection onto a coarse P1 mesh"></p>

---

## Quick Start

```python
from fem import box_mesh, Conditions, Dirichlet, Poisson, Plotter, Source
from fem.regions import everywhere

mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40))

# Conditions are geometric, so the same ones are valid on any mesh of this domain:
# what is applied to the domain (supports, loads, a source) all in one object.
conditions = Conditions(Dirichlet(everywhere(), 0), Source(1))

solution = Poisson().problem(mesh, conditions).solve()

plotter = Plotter(title="Poisson")
plotter.plot(mesh, solution.dofs, mode="surface")
plotter.show()
```

The equation builds a problem from the parts, which compose directly. The same solve, by hand:

```python
from fem import FunctionSpace, LinearProblem, DiffusionForm

space = FunctionSpace(mesh, n_components=1)
problem = LinearProblem(space, DiffusionForm(), conditions)
solution = problem.solve()
```

`problem.solve()` picks `LinearSolve` for a constant tangent and Newton otherwise; `strategy=`
chooses how it iterates and `backend=` how each linear system is solved, independently. The
result is typed by the physics: `Poisson().problem(mesh, conditions).solve()` is a
`DiffusionSolution`, an elastic one an `ElasticSolution`, with no narrowing at the call.

### What you choose at each step

Every step of a solve is one choice among a few named objects, all importable from `fem`.
`ARCHITECTURE.md` explains how they fit; this is the menu.

| Step | Options | Default |
|---|---|---|
| Mesh | `box_mesh`, `annulus_mesh`; an `Outline` of `Line`, `Arc`, `Circle`, `CubicBezier` pieces (`Outline.from_polygons`, `Outline.from_svg`, then `Outline.simplified` and `Outline.mesh`); `Mesh(vertices, elements)`; refine with `RedGreenRefiner` | |
| Element | `LinearTriangleElement`, `LinearTetrahedralElement`, `QuadraticTriangleElement`, `IsoparametricTriangleElement`, via `element_type=` | linear, read off the mesh |
| Equation | `Projection`, `Poisson`, `Heat`, `Wave`, `LinearElastic`, `FiniteStrainElastic` (with `law=` `StVenantKirchhoff` or `NeohookeanEnergyDensity`) | |
| Where | `everywhere`, `on_plane`, `in_box`, `on_tag`, `at_indices`, `union`, `intersect` | |
| Conditions | a `Conditions` of `Dirichlet`, `Neumann`, `Robin` (on regions), a volume `Source`, and `PointLoad`s; a value is a constant, a callable of position, or `TimeDependent` | none |
| Strategy | `LinearSolve`, `NewtonSolve` (with `BacktrackingLineSearch`, `TangentRegularization`), via `strategy=` | `default_strategy`: by the tangent |
| Backend | `DirectBackend`, `IterativeBackend`, `MinresBackend`, via `backend=` | direct |
| In time | `ThetaMethod`, `NewmarkMethod` (with `RayleighDamping`) | |
| Analyses | `BucklingAnalysis`, `ModalAnalysis`, `AdaptiveRefinement` with `ResidualEstimator` / `RecoveryEstimator` / `GoalOrientedEstimator`, `SensitivityAnalysis`, `DesignOptimizer` over a `SIMPModel` | |
| Result | `DiffusionSolution`, `ElasticSolution`, `FieldSolution` (each a `NodalField`), `TransientSolution`, `BucklingSolution`, `ModalSolution` | by the physics |
| Plot | `Plotter.plot(target, values, mode=)` with `mesh`, `boundary`, `colored`, `surface`, `arrows`, `solid`, `bc`, `refinement` | |


A solution is a typed dataclass. An elastic solve returns an `ElasticSolution`, which
carries the stress and strain as full tensors and derives the scalar measures on
demand:

```python
solution = LinearElastic(E=200, nu=0.3).problem(mesh, conditions).solve()

solution.dofs              # (n_nodes * n_components,) the DOF vector
solution.nodal_values      # (n_nodes, n_components) the same by node
solution.evaluate(points)  # (n_points, n_components) the field at any points
solution.deformed_mesh()   # the mesh displaced by the field
solution.stress            # (n_elements, 3, 3) Cauchy stress tensors
solution.von_mises         # (n_elements,) equivalent stress, the usual plot
solution.principal_stress  # (n_elements, 3) principal values, ascending
solution.compliance        # (n_elements,) strain energy per element
```

`fem/post/invariants.py` holds those reductions; each is rotation-invariant.

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

Runnable demos live in `examples/demos/` (run from the repo root) behind a small CLI. Each
demo is a package of two files: `physics.py` poses and solves the problem and is what the
gallery shows as the demo's source, `figures.py` draws it.
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
fem/                 # the solver package; grouped by layer, everything re-exported from `fem`
├── mesh/            # Mesh geometry, Outline pieces sampled to a PSLG, Ruppert meshing, red-green refinement, SVG
│
│   # discretization and constraints
├── elements.py      # stateless element types (P1/P2/curved) + batched ElementGeometry
├── quadrature.py    # reference-simplex Gauss rules, wired into assembly
├── space.py         # FunctionSpace: DOF numbering, geometry, assembly
├── regions.py       # position-based regions and fields (on_plane, in_box, on_tag, ...)
├── boundary.py      # Dirichlet, Neumann, Robin, each resolving itself on a node set
├── loads.py         # Source, PointLoad (and the BoundaryLoad the resolution builds)
├── conditions.py    # Conditions: everything applied to a domain -> ResolvedConditions on a space
├── problem.py       # Problem: space + operator + load + constraints; the narrow waist
│
├── physics/         # what equation, what material
│   ├── equations.py #   Equation: Projection, Poisson, Heat, Wave, LinearElastic, FiniteStrainElastic
│   ├── forms.py     #   Form: bilinear and energy integrands; stress recovery; rigid-body modes
│   ├── materials.py #   Hooke's law, Lame conversions (2D is plane strain)
│   ├── energies.py  #   hyperelastic strain-energy densities and their derivatives
│   ├── fields.py    #   Scalar/Vector: components per node, resolved against the mesh
│   └── derived.py   #   Flux: the flux or stress a form recovers
│
├── algebra/         # how Ax = b and F(x) = 0 are solved
│   ├── system.py    #   DiscreteSystem: Dirichlet elimination, factor once
│   ├── backends.py  #   DirectBackend (sparse LU) / IterativeBackend (AMG-CG) / MinresBackend
│   ├── solve.py     #   LinearSolve / NewtonSolve / EigenSolve strategies
│   └── integrators.py # ThetaMethod (1st order), NewmarkMethod (2nd order)
│
├── analysis/        # analyses and drivers that re-solve
│   ├── buckling.py  #   BucklingAnalysis: linearised (eigenvalue) buckling
│   ├── modal.py     #   ModalAnalysis: free-vibration modes
│   ├── adaptivity.py #  AdaptiveRefinement driver
│   ├── estimators.py #  residual, recovery, and goal-oriented error estimators
│   ├── sensitivity.py # adjoint sensitivity: quantities of interest and their gradients
│   └── design.py    #   SIMPModel + DesignOptimizer: density (topology) design
│
├── post/            # results
│   ├── solution.py  #   typed Solution hierarchy; ElasticSolution.from_solve
│   ├── recovery.py  #   nodal recovery: volume-weighted average and L2 projection
│   ├── invariants.py #  rotation-invariant tensor reductions (von Mises, ...)
│   └── io.py        #   mesh JSON, solution npz (no pickle)
│
├── plot/            # matplotlib only, loaded on first use: Plotter, PanelView tessellation, BC glyphs
└── numerics.py, typing.py   # helpers and semantic array aliases
tests/               # pytest suite (test_layering.py holds the module order)
examples/            # the CLI, the gallery builder, mms.py (manufactured-solution studies),
                     # and demos/<name>/{physics,figures}.py
files/               # example SVG outlines
```

The figures in this README are committed images (PNGs and one GIF), refreshed by
hand. After changing a demo, regenerate them and commit the result:

```bash
uv run python examples/make_readme_figures.py   # rewrites the figures in images/
```

## Methods

- Galerkin finite element method: P1 (linear) basis on triangles and tetrahedra, P2
  (quadratic) and curved isoparametric triangles, over a Gaussian quadrature layer
- Boundary conditions: Dirichlet, Neumann, Robin, and per-component (roller) constraints
- PDEs: L2 projection, Poisson, variable-coefficient diffusion, heat, wave, and two
  elastic models: `LinearElastic` (infinitesimal strain, Navier-Cauchy) and
  `FiniteStrainElastic` (geometrically exact Green-Lagrange strain under a
  hyperelastic law, St Venant-Kirchhoff by default); 2D elasticity is plane strain
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
  electrostatics, advection-diffusion
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
