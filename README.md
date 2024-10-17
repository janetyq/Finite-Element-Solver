# Finite Element Solver

This finite element method (FEM) solver is capable of solving a variety of partial differential equations (PDEs), such as Poisson, heat, wave, and both linear and nonlinear elasticity equations. It supports both Dirichlet and Neumann boundary conditions and can be applied to simulate both 2D and 3D meshes. Interesting features include custom meshing algorithms, adaptive mesh refinement to enhance simulation accuracy and topology optimization for optimizing structural design. (note: this README is a bit behind the current state of the project, reach out for more details)

## Details
This solver uses the Galerkin finite element method with linear basis functions on triangular meshes for 2D problems and tetrahedral meshes for 3D. It is designed to be modular, making it easy to add new PDEs, finite element types, or energy density functions.

## Examples
### L2 Projection
Given a function $f(x, y)$, we can find its best approximation in the finite element space, which is the space of linear functions on the triangular mesh.

![l2_projection](images/l2_projection_demo.png)

### Poisson's Equation
Poisson's equation is a partial differential equation that can be used to model heat transfer, electrostatics, fluid flow, and other phenomena. It is defined as $\Delta u = f$, where $f$ is a given function and $u$ is the unknown function we are trying to solve for. 

Using the finite element method, we can solve for $u$ by finding the weak form of the equation and discretizing it into a linear system. 

![poissons_demo](images/poissons_demo.png)

This example shows the velocity potential $u$ (where gradient of velocity potential = flow velocity) of fluid flow around an obstacle. The Robin boundary conditions are: $u = 0$ on the obstacle, and $n \cdot \Delta u = \frac{du}{dx} = 3$ on left inlet and $n \cdot \Delta u = -\frac{du}{dx} = -1$ on the right outlet. 

### Wave Equation
The wave equation is a partial differential equation that describes waves as they propogate through space and time. It is defined as $\frac{\partial^2 u}{\partial t^2} = c^2 \Delta u$, where $c$ is the wave speed and $u$ is the scalar function describing the wave.

We can simulate the wave propogation over time with Crank-Nicolson integration, solving for $u$ at each timestep.

<div style="display: flex; justify-content: space-between;">
    <img src="images/wave_demo1.png" alt="wave_demo1" width="45%" />
    <img src="images/wave_demo2.png" alt="wave_demo2" width="45%" />
</div>

The wave starts as a single pulse and propogates outwards at a constant speed. When it collides with the boundary, it reflects back and interferes with itself, creating a standing wave pattern.
<!-- TODO: add bc -->

### Heat Equation
The heat equation is a partial differential equation that describes the distribution of heat over time. It is defined as $\frac{\partial u}{\partial t} = \alpha \Delta u$, where $\alpha$ is the thermal diffusivity and $u$ is the temperature.

We can simulate the heat distribution over time with Backwards Euler integration, solving for $u$ at each timestep.

<div style="display: flex; justify-content: space-between;">
    <img src="images/heat_demo1.png" alt="heat_demo1" width="45%" />
    <img src="images/heat_demo2.png" alt="heat_demo2" width="45%" />
</div>

In this example, there is an initial high temperature bump in the corner of the domain. The heat diffuses outwards and eventually will reach a steady state where the temperature is constant. Heat is conserved in this simulation, where the mean temperature of the domain is constant over time.


### Linear Elastic Mechanics
The linear elastic mechanics solver can solve for the displacement and stress field of a solid object given applied forces and boundary conditions. 

![linear_elastic_demo1](images/elastics_demo1.png)

The starting mesh is a supported cantilever beam. We fix the left edge and apply a downward force on the right most edge, and a uniform body force due to gravity.

![linear_elastic_demo2](images/elastics_demo2.png)

The resulting deformed mesh shows the beam bending under the forces with a max stress at the corner of the support. 

Note: This example shows extreme displacement, in reality, the object would no longer be in the linear elastic regime and the solver would not be accurate.

## Adaptive Refinement

The solver can also perform adaptive mesh refinement to increase the accuracy of the solution. It works by calculating the a posteriori error estimate of each element and refining the elements with the largest error. We maintain the triangle quality of the mesh with regular (red-green) refinement.

Here, we show adaptive refinement on solving Poisson's equation.

![adaptive_refinement1](images/poissons_adaptive_refinement1.png)

We can see that the residual error is concentrated near the center of the domain, so the solver refines the mesh in that area. The final mesh has a much higher resolution in the center and much lower residual error.

![adaptive_refinement2](images/poissons_adaptive_refinement2.png)

## Topology Optimization

Topology optimization is a method of structural design where the material distribution of a structure is optimized to minimize some objective function. In this case, we are minimizing the compliance of the structure, which is the amount of deformation under a given load.

The boundary conditions are that the left edge is fixed and a downward force is applied to the right edge. The material distribution is represented by a density field, where 0 is no material and 1 is full material. The solver uses the SIMP (Solid Isotropic Material with Penalization) method to penalize intermediate densities.

<p align="center">
<video width="640" height="360" controls>
  <source src="/images/topopt.mp4" type="video/mp4">
  Video of topology optimization on a cantilevered beam
</video>
</p>

The solver starts with a uniform density field and iteratively updates the density field to minimize the compliance. This image shows the final density field. This structure uses approx 55% of the original material and only deforms slightly more.


## Methods
 - Galerkin Finite Element Method
 - Boundary conditions: Dirichlet, Neumann
 - Partial Differential Equations (PDEs): L2 projection, Poisson's equation, Heat equation, Wave equation, Navier-Cauchy equation (linear elastics), hyperelasticity
 - Integration: Forward/Backward Euler, Crank-Nicolson
 - Energy measures: Dirichlet energy, Kinetic energy
 - Error estimates: A posteriori error residuals
 - Optimization: Gradient descent, Newton-Raphson method, Optimality criteria method (SIMP)
 - Mesh algorithms: Delaunay triangulation, Ruppert's algorithm (line segments -> triangle mesh), Red-Green refinement, half-edge data structure


## Next Steps (in progress)
- Nonlinear elements: quadratic basis functions
- More PDEs: time-dependent dynamics, thermal expansion, transport equations, fluid mechanics, etc.
- Error estimates: a posteriori error estimates for adaptive refinement
- Efficiency: sparse solver
- Interesting Applications: cage-based shape optimization, inverse spring design


### References
*The Finite Element Method: Theory, Implementation, and Applications* by Mats G. Larson and Fredrik Bengzon.

[*SIMP Method for Topology Optimization*](https://help.solidworks.com/2019/english/solidworks/cworks/c_simp_method_topology.htm) by Dassault Systèmes.
