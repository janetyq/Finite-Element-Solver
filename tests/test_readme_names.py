"""The README's "What you choose at each step" table names real things.

Every backticked name in that section must resolve: an attribute chain from `fem`
(`Outline.from_svg`, `Plotter.plot`), a call on a named instance (`problem.solve()`,
where `problem` is a `Problem`), a bare method (`.simplified(tolerance)`, on some
exported class), a `PlotMode` value, or a keyword argument the table quotes. A rename
that forgets the table fails here. The "Where" column names files, which must exist.
"""
import re
from pathlib import Path

import pytest

import fem
from fem.plot.plotter import PlotMode

README = Path(__file__).resolve().parents[1] / 'README.md'
HEADING = '### What you choose at each step'
KEYWORDS = {'element_type=', 'source=', 'law=', 'mode=', 'strategy=', 'backend=', 'initial=', 'dt=', 'steps=', 'v0='}
# The instances the table calls methods on, by the lowercase name it uses.
INSTANCES = {'problem': fem.Problem, 'space': fem.FunctionSpace, 'outline': fem.Outline,
             'solution': fem.Solution, 'equation': fem.Equation}


def _section() -> str:
    text = README.read_text(encoding='utf-8')
    start = text.index(HEADING) + len(HEADING)
    end = re.search(r'^#{2,3} ', text[start:], flags=re.M)
    return text[start:start + end.start()] if end else text[start:]


def _names() -> list[str]:
    rows = [line for line in _section().splitlines() if line.startswith('|')]
    names = re.findall(r'`([^`]+)`', '\n'.join(rows))
    return sorted({n for n in names if n not in KEYWORDS and not _is_path(n)})


def _is_path(name: str) -> bool:
    return name.endswith('.py') or name.endswith('/')


def _paths() -> list[str]:
    '''The files the "Where" column names.'''
    rows = [line for line in _section().splitlines() if line.startswith('|')][2:]
    return [name for row in rows for name in re.findall(r'`([^`]+)`', row.rsplit('|', 2)[-2])
            if _is_path(name)]


def _resolves(name: str) -> bool:
    if name in {m.value for m in PlotMode}:
        return True
    # `.simplified(tolerance)`: a method some exported class has
    bare = re.match(r'\.(\w+)', name)
    if bare:
        return any(hasattr(getattr(fem, export), bare.group(1)) for export in fem.__all__)
    # `Mesh(vertices, elements)` or `Outline.from_polygons`: check the leading path;
    # `problem.solve()` starts from the instance's class
    head = re.match(r'[A-Za-z_][\w.]*', name)
    if not head:
        return False
    parts = head.group(0).split('.')
    obj = fem
    if parts[0] in INSTANCES:
        obj, parts = INSTANCES[parts[0]], parts[1:]
    for part in parts:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def test_the_table_exists():
    assert HEADING in README.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', _names())
def test_every_name_in_the_table_resolves(name):
    assert _resolves(name), f'`{name}` is in the README table but not in fem'


@pytest.mark.parametrize('path', _paths())
def test_every_file_in_the_where_column_exists(path):
    assert (README.parent / path).exists(), f'`{path}` is in the README table but not in the repo'
