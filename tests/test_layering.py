"""The layer order in ARCHITECTURE.md, as a test.

Every module in `fem/` is placed in `ORDER`, bottom to top. A module's top-level
imports may only name modules at or below its own position: the dependency graph
flows downward, and a cycle cannot appear without one of its edges pointing up.
`TYPE_CHECKING` imports are exempt outright: they are erased at runtime, so types
flow upward freely. A function-local import that points upward is a back-edge (a
`Problem` picking its default strategy, a `Mesh` saving itself); each is named in
its module's docstring and must appear in `BACK_EDGES` below, so a new one is a
deliberate decision, not drift.

The list is also the reading order of the package: what a module can assume exists.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

FEM = Path(__file__).resolve().parents[1] / 'fem'

ORDER = [
    # leaves; the matrix-level algebra (a backend, a factorization, a matrix with its
    # Dirichlet partition) knows nothing about spaces or PDEs, so it sits here, where a
    # space or a solution can hold a factorization of an operator it owns
    'typing', 'numerics', 'quadrature', 'regions', 'algebra.backends', 'algebra.system',
    'physics.fields', 'physics.materials', 'post.invariants',
    # geometry; the mesher sits above the PSLG it consumes, as the refiner does the mesh
    'mesh.curves', 'mesh.mesh', 'mesh.refinement', 'mesh.delaunay', 'mesh.structured',
    'mesh.pslg', 'mesh.ruppert', 'mesh.outline', 'mesh.svg', 'mesh',
    # discretization and constraints
    'elements', 'boundary', 'physics.energies', 'physics.plasticity', 'field',
    # the typed solutions, which everything above the physics packages
    'post.recovery', 'post.solution', 'physics.derived',
    # physics, the space that assembles it, and the problem that composes it
    'physics.forms', 'space', 'loads', 'conditions', 'problem', 'physics.equations', 'physics',
    # algebra over a Problem: the strategies, the integrators, the stepper
    'algebra.solve', 'algebra.integrators',
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


def _fem_modules(node: ast.AST) -> list[str]:
    '''The `fem` modules one import statement names, as `ORDER` keys.'''
    found = []
    if isinstance(node, ast.Import):
        found = [a.name for a in node.names if a.name.startswith('fem')]
    elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('fem'):
        # `from fem.post import invariants` names a module, not an attribute.
        for alias in node.names:
            candidate = f'{node.module}.{alias.name}'
            if (FEM.parent / Path(*candidate.split('.'))).with_suffix('.py').exists():
                found.append(candidate)
            else:
                found.append(node.module)
    return [name.removeprefix('fem').lstrip('.') for name in found]


def _is_type_checking_block(node: ast.AST) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            and node.test.id == 'TYPE_CHECKING')


def _top_level_fem_imports(path: Path) -> list[str]:
    '''Modules under `fem` imported at column 0, so neither inside a function nor an
    `if TYPE_CHECKING:` block.'''
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return [name for node in tree.body for name in _fem_modules(node)]


def _function_local_fem_imports(path: Path) -> list[str]:
    '''Modules under `fem` imported below the top level, outside `TYPE_CHECKING` blocks.'''
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = []
    for top in tree.body:
        if isinstance(top, (ast.Import, ast.ImportFrom)) or _is_type_checking_block(top):
            continue
        for node in ast.walk(top):
            found += _fem_modules(node)
    return found


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


# The documented back-edges: function-local imports that point upward in ORDER, each
# named in the importing module's docstring and listed in ARCHITECTURE.md. A
# function-local import that points downward (`fem`'s lazy `__getattr__` serving the
# plot layer) defers cost, not layering, and is not tracked here.
BACK_EDGES = {
    ('analysis.estimators', 'analysis.sensitivity'),  # goal-oriented estimator solves the dual
    ('field', 'physics.forms'),                       # boundary_integral's boundary mass form
    ('mesh.mesh', 'mesh.refinement'),                 # Mesh.refined
    ('mesh.mesh', 'post.io'),                         # Mesh.save / Mesh.load
    ('mesh.outline', 'mesh.svg'),                     # Outline.from_svg
    ('mesh.pslg', 'mesh.ruppert'),                    # PSLG.mesh runs the mesher
    ('physics.derived', 'physics.forms'),             # stress divergence builds the elastic form
    ('post.solution', 'post.io'),                     # Solution.save / Solution.load
    ('problem', 'algebra.solve'),                     # Problem.solve picks default_strategy
}


def test_function_local_imports_match_the_documented_back_edges():
    found = set()
    for path in MODULES:
        name = _module_name(path)
        for dep in _function_local_fem_imports(path):
            if RANK[dep] > RANK[name]:
                found.add((name, dep))
    undocumented = found - BACK_EDGES
    stale = BACK_EDGES - found
    assert not undocumented, (
        f'upward function-local imports not in BACK_EDGES: {sorted(undocumented)}; '
        'a new back-edge is a design decision: name it in the module docstring, '
        'ARCHITECTURE.md, and BACK_EDGES'
    )
    assert not stale, f'BACK_EDGES entries no longer in the code: {sorted(stale)}'


def test_importing_fem_does_not_import_the_plot_layer():
    """`import fem` must stay headless: matplotlib and `fem.plot` load only when
    `Plotter` is first touched."""
    code = ('import sys, fem; '
            'print(sorted(m for m in sys.modules if m == "matplotlib" or m.startswith("fem.plot")))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         check=True, cwd=FEM.parent).stdout.strip()
    assert out == '[]', out
