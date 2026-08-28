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
from typing import TYPE_CHECKING

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
from fem.solver import Solver
from fem.physics.forms import (
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
    rigid_body_modes,
)
from fem.loads import BoundaryLoad, Load, NodalSource, PointLoad, Source
from fem.physics.materials import LinearElasticMaterial
from fem.physics.energies import NeohookeanEnergyDensity, SmallStrain, StVenantKirchhoff
from fem.physics.derived import DerivedField
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.algebra.solve import (
    BacktrackingLineSearch,
    LinearSolve,
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
    LinearSolver,
    MinresBackend,
)
from fem.algebra.integrators import ThetaMethod, NewmarkMethod
from fem.post.solution import (
    Solution,
    FieldSolution,
    ScalarFieldSolution,
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
    DensityField,
    ModulusField,
)
from fem.analysis.design import (
    DesignOptimizer,
    SIMPModel,
    DesignHistory,
    TargetCompliance,
    calculate_smoothing_matrix,
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
