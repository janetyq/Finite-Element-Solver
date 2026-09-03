"""The layer order in ARCHITECTURE.md, as a test.

Every module in `fem/` is placed in `ORDER`, bottom to top. A module's top-level
imports may only name modules at or below its own position: the dependency graph
flows downward, and a cycle cannot appear without one of its edges pointing up.
Function-local imports and `TYPE_CHECKING` blocks are exempt; they are the documented
back-edges (a `Problem` picking its default strategy, a `Mesh` saving itself) and each
names why in its module's docstring.

The list is also the reading order of the package: what a module can assume exists.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

FEM = Path(__file__).resolve().parents[1] / 'fem'

ORDER = [
    # leaves
    'typing', 'numerics', 'quadrature', 'regions',
    'physics.fields', 'physics.materials', 'post.invariants',
    # geometry
    'mesh.curves', 'mesh.mesh', 'mesh.refinement', 'mesh.delaunay', 'mesh.ruppert', 'mesh.structured',
    'mesh.pslg', 'mesh.outline', 'mesh.svg', 'mesh',
    # discretization and constraints
    'elements', 'boundary', 'physics.energies', 'physics.plasticity', 'field',
    # the typed solutions, which everything above the physics packages
    'post.recovery', 'post.solution', 'physics.derived',
    # physics, the space that assembles it, and the problem that composes it
    'physics.forms', 'space', 'loads', 'conditions', 'problem', 'physics.equations', 'physics',
    # algebra
    'algebra.backends', 'algebra.system', 'algebra.solve', 'algebra.integrators',
    'algebra.stepping', 'algebra',
    # analyses and drivers
    'analysis.estimators', 'analysis.sensitivity', 'analysis.design', 'analysis.buckling',
    'analysis.modal', 'analysis.adaptivity', 'analysis',
    # post-processing that reads the whole stack
    'post.io', 'post',
    'plot.tessellation', 'plot.helpers', 'plot.bc', 'plot.plotter', 'plot',
    '',
]
RANK = {name: i for i, name in enumerate(ORDER)}


def _module_name(path: Path) -> str:
    rel = path.relative_to(FEM).with_suffix('')
    parts = list(rel.parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def _top_level_fem_imports(path: Path) -> list[str]:
    '''Modules under `fem` imported at column 0, so neither inside a function nor an
    `if TYPE_CHECKING:` block.'''
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith('fem')]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('fem'):
            module = node.module
            # `from fem.post import invariants` names a module, not an attribute.
            for alias in node.names:
                candidate = f'{module}.{alias.name}'
                if (FEM.parent / Path(*candidate.split('.'))).with_suffix('.py').exists():
                    found.append(candidate)
                else:
                    found.append(module)
    return [name.removeprefix('fem').lstrip('.') for name in found]


MODULES = sorted(FEM.rglob('*.py'), key=_module_name)


def test_every_module_is_placed():
    names = {_module_name(p) for p in MODULES}
    missing = names - set(ORDER)
    extra = set(ORDER) - names
    assert not missing, f'add to ORDER: {sorted(missing)}'
    assert not extra, f'not in fem/: {sorted(extra)}'


@pytest.mark.parametrize('path', MODULES, ids=_module_name)
def test_top_level_imports_flow_downward(path):
    name = _module_name(path)
    upward = [dep for dep in _top_level_fem_imports(path) if RANK[dep] > RANK[name]]
    assert not upward, f'{name} imports {upward} at top level, which sit above it'


def test_importing_fem_does_not_import_the_plot_layer():
    """`import fem` must stay headless: matplotlib and `fem.plot` load only when
    `Plotter` is first touched."""
    code = ('import sys, fem; '
            'print(sorted(m for m in sys.modules if m == "matplotlib" or m.startswith("fem.plot")))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         check=True, cwd=FEM.parent).stdout.strip()
    assert out == '[]', out
