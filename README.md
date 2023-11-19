# Finite Element Solver

Finite element method solver for partial differential equations (PDEs) on arbitrary 2d meshes. 

- Can solve the following equations:
    - L2 projection
    - Poisson equation
    - Heat equation
    - Wave equation
    - **Linear elastic mechanics**
- With customizable boundary conditions (Dirichlet, Neumann, Robin) and arbitrary source functions
- Visualization with colormap and contour plots
- Various quadrature rules
- All meshes used are custom generated from my "Meshing" project

## Examples
### L2 Projection
Given a function $f(x, y)$, we can find its best approximation in the finite element space, which is the space of linear functions on the triangular mesh.

![l2_projection](images/l2_projection_demo.png)

### Poisson Equation
Poisson's equation is a partial differential equation that can be used to model heat transfer, electrostatics, fluid flow, and other phenomena. It is defined as $\Delta u = f$, where $f$ is a given function and $u$ is the unknown function we are trying to solve for. 

Using the finite element method, we can solve for $u$ by finding the weak form of the equation and discretizing it into a linear system. 

![poissons_demo](images/poissons_demo.png)

This example shows the velocity potential $u$ (where gradient of velocity potential = flow velocity) of fluid flow around an obstacle. The Robin boundary conditions are: $u = 0$ on the obstacle, and $n \cdot \Delta u = \frac{du}{dx} = 3$ on left inlet and $n \cdot \Delta u = -\frac{du}{dx} = -1$ on the right outlet. 

### Wave Equation
The wave equation is a partial differential equation that describes waves as they propogate through space and time. It is defined as $\frac{\partial^2 u}{\partial t^2} = c^2 \Delta u$, where $c$ is the wave speed and $u$ is the scalar function describing the wave.

We can simulate the wave propogation over time with Crank-Nicolson integration, solving for $u$ at each timestep.

<div style="display: flex; justify-content: space-between;">
    <img src="images/wave_demo1.png" alt="wave_demo1" width="50%" />
    <img src="images/wave_demo2.png" alt="wave_demo2" width="50%" />
</div>

The wave starts as a single pulse and propogates outwards at a constant speed. When it collides with the boundary, it reflects back and interferes with itself, creating a standing wave pattern.
<!-- TODO: add bc -->

### Heat Equation
The heat equation is a partial differential equation that describes the distribution of heat over time. It is defined as $\frac{\partial u}{\partial t} = \alpha \Delta u$, where $\alpha$ is the thermal diffusivity and $u$ is the temperature.

We can simulate the heat distribution over time with Backwards Euler integration, solving for $u$ at each timestep.

<div style="display: flex; justify-content: space-between;">
    <img src="images/heat_demo1.png" alt="heat_demo1" width="50%" />
    <img src="images/heat_demo2.png" alt="heat_demo2" width="50%" />
</div>

In this example, there is an initial high temperature bump in the corner of the domain. The heat diffuses outwards and eventually will reach a steady state where the temperature is constant. Heat is conserved in this simulation, where the mean temperature of the domain is constant over time.


### Linear Elastic Mechanics
The linear elastic mechanics solver can solve for the displacement and stress field of a solid object given applied forces and boundary conditions. 

![linear_elastic_demo1](images/elastics_demo1.png)

The starting mesh is a supported cantilever beam. We fix the left edge and apply a downward force on the right most edge, and a uniform body force due to gravity.

![linear_elastic_demo2](images/elastics_demo2.png)

The resulting deformed mesh shows the beam bending under the forces with a max stress at the corner of the support. 

Note: This example shows extreme displacement, in reality, the object would no longer be in the linear elastic regime and the solver would not be accurate.

## Todo
    - Add thermal expansion to linear elastics model
    - Solve transport equations, fluid mechanics, etc.
    - Adaptive mesh refinement
    - Expand to a 3d solver


### References
Algorithms are based on the book *The Finite Element Method: Theory, Implementation, and Applications* by Mats G. Larson and Fredrik Bengzon.