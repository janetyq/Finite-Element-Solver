"""Finite Element Solver.

A finite element method (FEM) solver for 2D/3D PDEs (Poisson, heat, wave, linear
and nonlinear elasticity), with custom meshing, adaptive refinement, and topology
optimization.

Common entry points are re-exported here, so typical use is:

    from fem import create_rect_mesh, BoundaryConditions, BCType, Solver, Poisson
    from fem.regions import everywhere

    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40))
    equation = Poisson(source=lambda p: 1.0)

    bc = BoundaryConditions()                          # described geometrically,
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)        # so it holds on any mesh

    solution = Solver(mesh, equation, bc).solve()
"""

__version__ = "0.1.0"

import logging

from fem.mesh.mesh import Mesh
from fem.mesh.curves import Arc, Circle, Curve
from fem.space import FunctionSpace
from fem.mesh.structured import create_rect_mesh
from fem.elements import (
    LinearLineElement,
    LinearTriangleElement,
    LinearTetrahedralElement,
    QuadraticLineElement,
    QuadraticTriangleElement,
    IsoparametricLineElement,
    IsoparametricTriangleElement,
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
    Diffusion,
    Wave,
    LinearElastic,
    StrainMeasure,
)
from fem.solver import Solver
from fem.forms import LinearForm
from fem.problem import LinearProblem, EnergyProblem
from fem.solve import (
    BacktrackingLineSearch,
    LinearSolve,
    NewtonSolve,
    EigenSolve,
    TangentRegularization,
)
from fem.backends import DirectBackend, IterativeBackend, MinresBackend, rigid_body_modes
from fem.integrators import ThetaMethod, NewmarkMethod
from fem.solution import (
    Solution,
    FieldSolution,
    ScalarFieldSolution,
    ElasticSolution,
    BucklingSolution,
    ModalSolution,
    TransientSolution,
    WaveSolution,
)
from fem.buckling import BucklingAnalysis
from fem.modal import ModalAnalysis
from fem.energy_solver import EnergySolver
from fem.sensitivity import (
    SensitivityAnalysis,
    Compliance,
    PointValue,
    MeanStress,
    SoftMaxStress,
    DensityField,
    ModulusField,
)
from fem.design import (
    DesignOptimizer,
    SIMPModel,
    DesignHistory,
    TargetCompliance,
    calculate_smoothing_matrix,
    optimality_criteria_update,
)
from fem.adaptivity import AdaptiveRefinement
from fem.estimators import (
    ErrorEstimator,
    GoalOrientedEstimator,
    RecoveryEstimator,
    ResidualEstimator,
)
from fem.plot.plotter import Plotter, PlotMode

# Library-quiet by default: emit nothing unless the application configures a
# handler (e.g. logging.basicConfig(level=logging.INFO)). Standard practice.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Mesh",
    "Arc",
    "Circle",
    "Curve",
    "FunctionSpace",
    "create_rect_mesh",
    "LinearLineElement",
    "LinearTriangleElement",
    "LinearTetrahedralElement",
    "QuadraticLineElement",
    "QuadraticTriangleElement",
    "IsoparametricLineElement",
    "IsoparametricTriangleElement",
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
    "Diffusion",
    "Wave",
    "LinearElastic",
    "StrainMeasure",
    "LinearProblem",
    "EnergyProblem",
    "LinearForm",
    "LinearSolve",
    "NewtonSolve",
    "BacktrackingLineSearch",
    "TangentRegularization",
    "EigenSolve",
    "DirectBackend",
    "IterativeBackend",
    "MinresBackend",
    "rigid_body_modes",
    "ThetaMethod",
    "NewmarkMethod",
    "Solution",
    "FieldSolution",
    "ScalarFieldSolution",
    "ElasticSolution",
    "BucklingSolution",
    "ModalSolution",
    "TransientSolution",
    "WaveSolution",
    "BucklingAnalysis",
    "ModalAnalysis",
    "EnergySolver",
    "SensitivityAnalysis",
    "Compliance",
    "PointValue",
    "MeanStress",
    "SoftMaxStress",
    "DensityField",
    "ModulusField",
    "DesignOptimizer",
    "SIMPModel",
    "DesignHistory",
    "TargetCompliance",
    "calculate_smoothing_matrix",
    "optimality_criteria_update",
    "AdaptiveRefinement",
    "ErrorEstimator",
    "ResidualEstimator",
    "RecoveryEstimator",
    "GoalOrientedEstimator",
    "Plotter",
    "PlotMode",
]
