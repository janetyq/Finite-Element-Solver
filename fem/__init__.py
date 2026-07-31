"""Finite Element Solver.

A finite element method (FEM) solver for 2D/3D PDEs (Poisson, heat, wave, linear
and nonlinear elasticity), with custom meshing, adaptive refinement, and topology
optimization.

Common entry points are re-exported here, so typical use is:

    from fem import Mesh, BoundaryConditions, BCType, Solver, Poisson
    from fem.regions import everywhere

    mesh = Mesh.load("files/mesh_40x40.json")
    equation = Poisson(source=lambda p: 1.0)

    bc = BoundaryConditions()                          # described geometrically,
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)        # so it holds on any mesh

    solution = Solver(mesh, equation, bc).solve()
"""

__version__ = "0.1.0"

import logging

from fem.mesh.mesh import Mesh
from fem.space import FunctionSpace
from fem.mesh.ruppert import create_rect_mesh, create_approx_mesh
from fem.elements import (
    LinearLineElement,
    LinearTriangleElement,
    LinearTetrahedralElement,
)
from fem.boundary import BoundaryConditions, BCType, ResolvedBC
from fem.regions import (
    everywhere,
    on_plane,
    in_box,
    intersect,
    union,
    at_indices,
)
from fem.equations import (
    Equation,
    Projection,
    Poisson,
    LinearElastic,
    StrainMeasure,
)
from fem.solver import Solver
from fem.problem import (
    LinearProblem,
    EnergyProblem,
    projection,
    poisson,
    linear_elastic,
    heat,
    wave,
)
from fem.solve import LinearSolve, NewtonSolve
from fem.backends import DirectBackend, IterativeBackend, rigid_body_modes
from fem.integrators import ThetaMethod, NewmarkMethod
from fem.solution import (
    Solution,
    FieldSolution,
    ElasticSolution,
    TransientSolution,
    WaveSolution,
)
from fem.energy_solver import EnergySolver
from fem.topology import TopologyOptimizer
from fem.adaptivity import AdaptiveRefinement
from fem.plot.plotter import Plotter, PlotMode

# Library-quiet by default: emit nothing unless the application configures a
# handler (e.g. logging.basicConfig(level=logging.INFO)). Standard practice.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Mesh",
    "FunctionSpace",
    "create_rect_mesh",
    "create_approx_mesh",
    "LinearLineElement",
    "LinearTriangleElement",
    "LinearTetrahedralElement",
    "BoundaryConditions",
    "BCType",
    "ResolvedBC",
    "everywhere",
    "on_plane",
    "in_box",
    "intersect",
    "union",
    "at_indices",
    "Solver",
    "Equation",
    "Projection",
    "Poisson",
    "LinearElastic",
    "StrainMeasure",
    "LinearProblem",
    "EnergyProblem",
    "projection",
    "poisson",
    "linear_elastic",
    "heat",
    "wave",
    "LinearSolve",
    "NewtonSolve",
    "DirectBackend",
    "IterativeBackend",
    "rigid_body_modes",
    "ThetaMethod",
    "NewmarkMethod",
    "Solution",
    "FieldSolution",
    "ElasticSolution",
    "TransientSolution",
    "WaveSolution",
    "EnergySolver",
    "TopologyOptimizer",
    "AdaptiveRefinement",
    "Plotter",
    "PlotMode",
]
