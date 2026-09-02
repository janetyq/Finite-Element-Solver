"""One package per demo, each exporting `DEMO`; `cli.py` collects them.

`DEMO_NAMES` is the order they are registered in, which is the order the gallery lists
them within a section: build a domain first, then solve on it, then the solids, then
the checks on accuracy and speed.
"""
import importlib

from demo_registry import Demo

DEMO_NAMES = [
    'outline_to_mesh',
    'heat',
    'wave',
    'poisson',
    'linear_elastic',
    'elasticity_models',
    'stress_concentration',
    'pressurized_cylinder',
    'bracket',
    'buckling',
    'modal',
    'topology_optimization',
    'l2_projection',
    'convergence',
    'refinement',
    'timing_benchmark',
]


def all_demos() -> list[Demo]:
    """Every demo, in `DEMO_NAMES` order."""
    return [importlib.import_module(f'demos.{name}').DEMO for name in DEMO_NAMES]
