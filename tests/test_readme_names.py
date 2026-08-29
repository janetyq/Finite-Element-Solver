"""The README's "What you choose at each step" table names real things.

Every backticked name in that section must resolve: an attribute chain from `fem`
(`Outline.from_svg`, `Plotter.plot`), a `PlotMode` value, or a keyword argument the table
quotes. A rename that forgets the table fails here.
"""
import re
from pathlib import Path

import pytest

import fem
from fem.plot.plotter import PlotMode

README = Path(__file__).resolve().parents[1] / 'README.md'
HEADING = '### What you choose at each step'
KEYWORDS = {'element_type=', 'source=', 'law=', 'mode=', 'strategy=', 'backend='}


def _section() -> str:
    text = README.read_text(encoding='utf-8')
    start = text.index(HEADING) + len(HEADING)
    end = re.search(r'^#{2,3} ', text[start:], flags=re.M)
    return text[start:start + end.start()] if end else text[start:]


def _names() -> list[str]:
    rows = [line for line in _section().splitlines() if line.startswith('|')]
    names = re.findall(r'`([^`]+)`', '\n'.join(rows))
    return sorted({n for n in names if n not in KEYWORDS})


def _resolves(name: str) -> bool:
    if name in {m.value for m in PlotMode}:
        return True
    # `Mesh(vertices, elements)` or `Outline.from_polygons`: check the leading path
    head = re.match(r'[A-Za-z_][\w.]*', name)
    if not head:
        return False
    obj = fem
    for part in head.group(0).split('.'):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def test_the_table_exists():
    assert HEADING in README.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', _names())
def test_every_name_in_the_table_resolves(name):
    assert _resolves(name), f'`{name}` is in the README table but not in fem'
