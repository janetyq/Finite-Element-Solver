"""A demo's source as gallery HTML: highlighted, with its `fem` imports linked to the
library on GitHub.

Pure text in, HTML out, with no dependency on the demo being run, so the gallery renders
it in the main process after the workers return. Pygments does the highlighting at build
time; its style sheets are inlined into the page, which keeps the gallery's plain-files
promise (no CDN).
"""
import os
import re
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


def highlight(code: str) -> str:
    """`code` as a `<div class="source"><pre>` listing, `fem` imports linked."""
    return link_imports(_pygments_highlight(code, PythonLexer(), HtmlFormatter(cssclass='source')))
