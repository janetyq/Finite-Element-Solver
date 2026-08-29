"""A demo's source as gallery HTML: highlighted, with a table of its definitions and
links from its `fem` imports to the library on GitHub.

Pure text in, HTML out, with no dependency on the demo being run, so the gallery renders
it in the main process after the workers return. Pygments does the highlighting at build
time; its style sheets are inlined into the page, which keeps the gallery's plain-files
promise (no CDN).
"""
import ast
import html
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pygments import highlight as _pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

REPO = 'https://github.com/janetyq/Finite-Element-Solver'
_ROOT = Path(__file__).resolve().parents[1]

# The light sheet is the default; the dark one is scoped to `prefers-color-scheme: dark`
# by the page, so the code follows the rest of the page's palette.
STYLE_LIGHT = HtmlFormatter(style='friendly').get_style_defs('.source')
STYLE_DARK = HtmlFormatter(style='github-dark').get_style_defs('.source')


@dataclass(frozen=True)
class Definition:
    """One top-level `def` or `class` of a listing: its name, the line it starts on, and
    the first line of its docstring."""
    name: str
    kind: str        # 'def' or 'class'
    lineno: int
    summary: str


def outline(code: str) -> list[Definition]:
    """The top-level definitions of `code`, in order. Private names included: they are
    in the listing, and a reader looking for one wants to jump to it too."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = 'def'
        elif isinstance(node, ast.ClassDef):
            kind = 'class'
        else:
            continue
        doc = ast.get_docstring(node) or ''
        summary = doc.split('\n\n', 1)[0].replace('\n', ' ').strip()
        found.append(Definition(node.name, kind, node.lineno, summary))
    return found


def _module_path(name: str) -> str | None:
    """`fem/mesh/mesh.py` for `fem.mesh.mesh`, `fem/mesh/__init__.py` for a package, or
    None for a name that is neither on this checkout."""
    parts = name.split('.')
    if parts[0] != 'fem':
        return None
    as_module = Path(*parts).with_suffix('.py')
    as_package = Path(*parts) / '__init__.py'
    for candidate in (as_module, as_package):
        if (_ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


def link_imports(highlighted: str, ref: str | None = None) -> str:
    """`highlighted` with every `fem` module name in an import linked to its file on
    GitHub at `ref` (the rendering commit under Actions, `main` otherwise).

    Pygments emits an import's dotted path as one `nn` span, so the span is the unit.
    Other packages are left alone: the reader is here for the library.
    """
    ref = ref or os.environ.get('GITHUB_SHA') or 'main'

    def link(match: re.Match[str]) -> str:
        path = _module_path(match.group(1))
        if path is None:
            return match.group(0)
        return f'<a class="module" href="{REPO}/blob/{ref}/{path}">{match.group(0)}</a>'

    return re.sub(r'<span class="nn">(fem(?:\.\w+)*)</span>', link, highlighted)


def highlight(code: str, anchors: str) -> str:
    """`code` as a `<div class="source"><pre>` listing, every line carrying an anchor
    `{anchors}-{lineno}`, `fem` imports linked."""
    formatter = HtmlFormatter(cssclass='source', lineanchors=anchors)
    return link_imports(_pygments_highlight(code, PythonLexer(), formatter))


def toc_html(code: str, anchors: str) -> str:
    """A list of `code`'s definitions linking to their lines in the `highlight` listing
    with the same `anchors`; empty for a listing too short to need one."""
    definitions = outline(code)
    if len(definitions) < 2:
        return ''
    items = []
    for d in definitions:
        summary = f' <span class="summary">{html.escape(d.summary)}</span>' if d.summary else ''
        items.append(f'<li><a href="#{anchors}-{d.lineno}"><code>{html.escape(d.name)}</code></a>'
                     f'{summary}</li>')
    return '<ul class="toc">\n' + '\n'.join(items) + '\n</ul>'
