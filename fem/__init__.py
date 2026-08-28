"""Finite Element Solver.

A finite element method (FEM) solver for 2D/3D PDEs (Poisson, heat, wave, linear
and nonlinear elasticity), with custom meshing, adaptive refinement, and topology
optimization.

Common entry points are re-exported here, so typical use is:

    from fem import box_mesh, BoundaryConditions, Dirichlet, Poisson
    from fem.regions import everywhere

    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40))
    # Conditions are described geometrically, so one spec holds on any mesh.
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))

    solution = Poisson(source=lambda p: 1.0).problem(mesh, bc).solve()
"""

__version__ = "0.1.0"

import logging

from fem.mesh.mesh import Mesh
from fem.mesh.curves import Arc, Circle, Curve
from fem.space import FunctionSpace
from fem.mesh.structured import annulus_mesh, box_mesh
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.pslg import PSLG
from fem.mesh.svg import read_svg_to_pslg
from fem.elements import (
    LinearLineElement,
    LinearTriangleElement,
    LinearTetrahedralElement,
    QuadraticLineElement,
    QuadraticTriangleElement,
    IsoparametricLineElement,
    IsoparametricTriangleElement,
)
from fem.boundary import BoundaryConditions, Condition, Dirichlet, Neumann, ResolvedBC, Robin
from fem.regions import (
    everywhere,
    on_plane,
    in_box,
    on_tag,
    intersect,
    union,
    at_indices,
    TimeDependent,
)
from fem.equations import (
    Equation,
    Projection,
    Poisson,
    Heat,
    Wave,
    Elasticity,
    LinearElastic,
    FiniteStrainElastic,
)
from fem.solver import Solver
from fem.forms import (
    BilinearForm,
    DiffusionForm,
    EnergyDensity,
    EnergyForm,
    Form,
    BoundaryMassForm,
    LinearElasticForm,
    MassForm,
    ScaledForm,
    SumForm,
)
from fem.loads import BoundaryLoad, Load, NodalSource, PointLoad, Source
from fem.materials import LinearElasticMaterial
from fem.energies import NeohookeanEnergyDensity, SmallStrain, StVenantKirchhoff
from fem.postprocess import DerivedField
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.solve import (
    BacktrackingLineSearch,
    LinearSolve,
    NewtonSolve,
    EigenSolve,
    SolveStrategy,
    TangentRegularization,
    default_strategy,
)
from fem.backends import (
    Backend,
    DirectBackend,
    IterativeBackend,
    LinearSolver,
    MinresBackend,
    rigid_body_modes,
)
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
from fem.sensitivity import (
    SensitivityAnalysis,
    QuantityOfInterest,
    Parameterization,
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
    "box_mesh",
    "annulus_mesh",
    "RuppertsAlgorithm",
    "RedGreenRefiner",
    "PSLG",
    "read_svg_to_pslg",
    "LinearLineElement",
    "LinearTriangleElement",
    "LinearTetrahedralElement",
    "QuadraticLineElement",
    "QuadraticTriangleElement",
    "IsoparametricLineElement",
    "IsoparametricTriangleElement",
    "BoundaryConditions",
    "Condition",
    "Dirichlet",
    "Neumann",
    "Robin",
    "ResolvedBC",
    "everywhere",
    "on_plane",
    "in_box",
    "on_tag",
    "intersect",
    "union",
    "at_indices",
    "TimeDependent",
    "Solver",
    "Equation",
    "Projection",
    "Poisson",
    "Heat",
    "Wave",
    "Elasticity",
    "LinearElastic",
    "FiniteStrainElastic",
    "LinearProblem",
    "Problem",
    "RayleighDamping",
    "Source",
    "NodalSource",
    "Load",
    "BoundaryLoad",
    "PointLoad",
    "Form",
    "BilinearForm",
    "EnergyForm",
    "EnergyDensity",
    "MassForm",
    "BoundaryMassForm",
    "SumForm",
    "DiffusionForm",
    "LinearElasticForm",
    "ScaledForm",
    "LinearElasticMaterial",
    "StVenantKirchhoff",
    "SmallStrain",
    "NeohookeanEnergyDensity",
    "DerivedField",
    "SolveStrategy",
    "LinearSolve",
    "NewtonSolve",
    "default_strategy",
    "BacktrackingLineSearch",
    "TangentRegularization",
    "EigenSolve",
    "Backend",
    "LinearSolver",
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
    "SensitivityAnalysis",
    "QuantityOfInterest",
    "Parameterization",
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
