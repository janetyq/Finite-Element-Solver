"""Finite Element Solver.

A finite element method (FEM) solver for 2D/3D PDEs (Poisson, heat, wave, linear
and nonlinear elasticity), with custom meshing, adaptive refinement, and topology
optimization.

Common entry points are re-exported here, so typical use is:

    from fem import FEMesh, BoundaryConditions, Solver, Equation

    mesh = FEMesh.load("files/mesh_40x40.json")
    equation = Equation("poisson")
    bc = BoundaryConditions(mesh)
    ...
    solution = Solver(mesh, equation, bc).solve()
"""

__version__ = "0.1.0"

from fem.mesh.mesh import Mesh
from fem.mesh.femesh import FEMesh
from fem.mesh.generation import create_rect_mesh, create_approx_mesh
from fem.elements import (
    LinearLineElement,
    LinearTriangleElement,
    LinearTetrahedralElement,
)
from fem.boundary import BoundaryConditions
from fem.solver import Solver, Equation
from fem.solution import Solution
from fem.energy_solver import EnergySolver
from fem.topology import TopologyOptimizer
from fem.plot.plotter import Plotter

__all__ = [
    "Mesh",
    "FEMesh",
    "create_rect_mesh",
    "create_approx_mesh",
    "LinearLineElement",
    "LinearTriangleElement",
    "LinearTetrahedralElement",
    "BoundaryConditions",
    "Solver",
    "Equation",
    "Solution",
    "EnergySolver",
    "TopologyOptimizer",
    "Plotter",
]
