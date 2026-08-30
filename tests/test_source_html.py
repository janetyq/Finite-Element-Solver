"""The gallery's source listings: highlighted and linked to the library."""
import re

from source_html import REPO, highlight, link_imports  # noqa: E402

SNIPPET = '''\
import numpy as np
from fem.mesh.mesh import Mesh
from fem.mesh import box_mesh
from fem.no_such_module import thing


def run():
    return Mesh
'''


def test_fem_imports_link_to_their_file_on_github():
    """A module links to its file, a package to its `__init__`, at the ref built."""
    page = link_imports(highlight(SNIPPET), ref='abc123')
    assert f'href="{REPO}/blob/abc123/fem/mesh/mesh.py"' in page
    assert f'href="{REPO}/blob/abc123/fem/mesh/__init__.py"' in page


def test_other_imports_and_unknown_fem_names_are_left_alone():
    page = highlight(SNIPPET)
    assert 'fem/no_such_module' not in page
    assert re.search(r'<a[^>]*>[^<]*numpy', page) is None


def test_the_ref_defaults_to_main(monkeypatch):
    monkeypatch.delenv('GITHUB_SHA', raising=False)
    assert '/blob/main/fem/mesh/mesh.py' in highlight(SNIPPET)
