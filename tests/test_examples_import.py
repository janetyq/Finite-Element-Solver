"""The demo CLI imports with only the base dependencies.

CI installs no extras, so a module-level import of an optional package in `examples/`
does not degrade one demo -- it takes down `cli.py` entirely, `list` included, and every
test that imports the registry. That has now happened twice, once for `svg.path` and
once for `tetgen`, and neither showed up locally because a developer environment has the
extras installed. This runs the import in a subprocess with the optional packages hidden.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Every distribution named in [project.optional-dependencies], by import name.
OPTIONAL_ROOTS = ('pyvista', 'tetgen')

PROGRAM = textwrap.dedent('''
    import importlib.abc, sys

    BLOCKED = {blocked!r}

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in BLOCKED:
                raise ImportError('blocked ' + name)

    sys.meta_path.insert(0, Block())
    sys.path.insert(0, {examples!r})

    import matplotlib
    matplotlib.use('Agg')

    import cli
    registry = cli.build_registry()
    print('OK', len(registry))
''')


def test_cli_imports_without_the_optional_extras():
    program = PROGRAM.format(blocked=OPTIONAL_ROOTS, examples=str(REPO / 'examples'))
    result = subprocess.run(
        [sys.executable, '-c', program],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, (
        'examples/ does not import without the optional extras, so CI cannot run any '
        f'demo:\n{result.stderr}'
    )
    assert result.stdout.startswith('OK'), result.stdout
