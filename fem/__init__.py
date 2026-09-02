"""Finite Element Solver.

A finite element method (FEM) solver for 2D/3D PDEs (Poisson, heat, wave, linear
and nonlinear elasticity), with custom meshing, adaptive refinement, and topology
optimization.

Common entry points are re-exported here, so typical use is:

    from fem import box_mesh, Conditions, Dirichlet, Poisson, Source
    from fem.regions import everywhere

    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40))
    # Conditions are described geometrically, so one spec holds on any mesh.
    conditions = Conditions(Dirichlet(everywhere(), 0.0), Source(1.0))

    solution = Poisson().problem(mesh, conditions).solve()
"""

__version__ = "0.1.0"

import logging
from typing import TYPE_CHECKING

from fem.mesh.mesh import Mesh
from fem.mesh.curves import Arc, Circle, CubicBezier, Curve, Line, Piece
from fem.space import FunctionSpace
from fem.field import NodalField
from fem.mesh.structured import box_mesh
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.outline import Outline, douglas_peucker
from fem.elements import (
    LinearLineElement,
    LinearTriangleElement,
    LinearTetrahedralElement,
    QuadraticLineElement,
    QuadraticTriangleElement,
    IsoparametricLineElement,
    IsoparametricTriangleElement,
)
from fem.boundary import Condition, Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial, ResolvedConditions
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
from fem.physics.equations import (
    Equation,
    Projection,
    Poisson,
    Heat,
    Wave,
    Elasticity,
    LinearElastic,
    FiniteStrainElastic,
)
from fem.physics.forms import (
    BilinearForm,
    DiffusionForm,
    Eigenstrain,
    EnergyDensity,
    EnergyForm,
    Form,
    BoundaryMassForm,
    LinearElasticForm,
    MassForm,
    ScaledForm,
    SumForm,
    ThermalStrain,
    rigid_body_modes,
)
from fem.loads import Load, PointLoad, Source
from fem.physics.materials import LinearElasticMaterial
from fem.physics.energies import NeohookeanEnergyDensity, SmallStrain, StVenantKirchhoff
from fem.physics.derived import Flux
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.algebra.solve import (
    BacktrackingLineSearch,
    LinearSolve,
    NewtonDivergence,
    NewtonSolve,
    EigenSolve,
    SolveStrategy,
    TangentRegularization,
    default_strategy,
)
from fem.algebra.backends import (
    Backend,
    DirectBackend,
    IterativeBackend,
    Factorization,
    MinresBackend,
)
from fem.algebra.integrators import ThetaMethod, NewmarkMethod
from fem.post.solution import (
    Solution,
    FieldSolution,
    DiffusionSolution,
    ElasticSolution,
    BucklingSolution,
    ModalSolution,
    TransientSolution,
    WaveSolution,
)
from fem.analysis.buckling import BucklingAnalysis
from fem.analysis.modal import ModalAnalysis
from fem.analysis.sensitivity import (
    SensitivityAnalysis,
    QuantityOfInterest,
    Parameterization,
    Compliance,
    PointValue,
    MeanStress,
    SoftMaxStress,
    DensityParameterization,
    ModulusParameterization,
)
from fem.analysis.design import (
    DesignOptimizer,
    SIMPModel,
    DesignHistory,
    TargetCompliance,
    calculate_smoothing_matrix,
    filter_sensitivity,
    optimality_criteria_update,
)
from fem.analysis.adaptivity import AdaptiveRefinement
from fem.analysis.estimators import (
    ErrorEstimator,
    GoalOrientedEstimator,
    RecoveryEstimator,
    ResidualEstimator,
)

# `Plotter` and `PlotMode` are served lazily so `import fem` does not import matplotlib:
# the solve path never needs it, and a headless run should not pay for it.
_PLOT_EXPORTS = {'Plotter', 'PlotMode'}

if TYPE_CHECKING:
    from fem.plot.plotter import PlotMode, Plotter


def __getattr__(name: str):
    if name in _PLOT_EXPORTS:
        from fem.plot import plotter
        return getattr(plotter, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Library-quiet by default: emit nothing unless the application configures a
# handler (e.g. logging.basicConfig(level=logging.INFO)). Standard practice.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Mesh",
    "Arc",
    "Circle",
    "CubicBezier",
    "Curve",
    "Line",
    "Piece",
    "FunctionSpace",
    "box_mesh",
    "RuppertsAlgorithm",
    "RedGreenRefiner",
    "Outline",
    "douglas_peucker",
    "LinearLineElement",
    "LinearTriangleElement",
    "LinearTetrahedralElement",
    "QuadraticLineElement",
    "QuadraticTriangleElement",
    "IsoparametricLineElement",
    "IsoparametricTriangleElement",
    "Condition",
    "Conditions",
    "Initial",
    "ResolvedConditions",
    "Dirichlet",
    "Neumann",
    "Robin",
    "everywhere",
    "on_plane",
    "in_box",
    "on_tag",
    "intersect",
    "union",
    "at_indices",
    "TimeDependent",
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
    "Load",
    "PointLoad",
    "Form",
    "BilinearForm",
    "EnergyForm",
    "EnergyDensity",
    "Eigenstrain",
    "ThermalStrain",
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
    "Flux",
    "SolveStrategy",
    "LinearSolve",
    "NewtonSolve",
    "NewtonDivergence",
    "default_strategy",
    "BacktrackingLineSearch",
    "TangentRegularization",
    "EigenSolve",
    "Backend",
    "Factorization",
    "DirectBackend",
    "IterativeBackend",
    "MinresBackend",
    "rigid_body_modes",
    "ThetaMethod",
    "NewmarkMethod",
    "NodalField",
    "Solution",
    "FieldSolution",
    "DiffusionSolution",
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
    "DensityParameterization",
    "ModulusParameterization",
    "DesignOptimizer",
    "SIMPModel",
    "DesignHistory",
    "TargetCompliance",
    "calculate_smoothing_matrix",
    "filter_sensitivity",
    "optimality_criteria_update",
    "AdaptiveRefinement",
    "ErrorEstimator",
    "ResidualEstimator",
    "RecoveryEstimator",
    "GoalOrientedEstimator",
    "Plotter",
    "PlotMode",
]
