"""The gallery's source listings: highlighted, outlined, and linked to the library."""
import re

from demos import modal  # noqa: E402
from source_html import REPO, highlight, link_imports, outline, toc_html  # noqa: E402

SNIPPET = '''\
import numpy as np
from fem.mesh.mesh import Mesh
from fem.mesh import box_mesh
from fem.no_such_module import thing


def _helper(x):
    """Private, but in the listing.

    A second paragraph the summary leaves out.
    """
    return x


class Study:
    pass


def run():
    return Study()
'''


def test_outline_lists_every_top_level_definition_with_its_summary():
    found = outline(SNIPPET)
    assert [(d.kind, d.name) for d in found] == [
        ('def', '_helper'), ('class', 'Study'), ('def', 'run')]
    assert found[0].summary == 'Private, but in the listing.'
    assert found[1].summary == ''
    assert found[0].lineno == 7


def test_outline_of_unparsable_source_is_empty():
    assert outline('def broken(:') == []


def test_fem_imports_link_to_their_file_on_github():
    """A module links to its file, a package to its `__init__`, at the ref built."""
    page = link_imports(highlight(SNIPPET, 'src'), ref='abc123')
    assert f'href="{REPO}/blob/abc123/fem/mesh/mesh.py"' in page
    assert f'href="{REPO}/blob/abc123/fem/mesh/__init__.py"' in page


def test_other_imports_and_unknown_fem_names_are_left_alone():
    page = highlight(SNIPPET, 'src')
    assert 'fem/no_such_module' not in page
    assert re.search(r'<a[^>]*>[^<]*numpy', page) is None


def test_the_ref_defaults_to_main(monkeypatch):
    monkeypatch.delenv('GITHUB_SHA', raising=False)
    assert '/blob/main/fem/mesh/mesh.py' in highlight(SNIPPET, 'src')


def test_toc_links_to_anchors_the_listing_carries():
    toc = toc_html(SNIPPET, 'src')
    listing = highlight(SNIPPET, 'src')
    targets = re.findall(r'href="#([^"]+)"', toc)
    assert targets == ['src-7', 'src-15', 'src-19']
    for target in targets:
        assert f'id="{target}"' in listing
    assert '<code>_helper</code>' in toc
    assert 'Private, but in the listing.' in toc


def test_no_toc_for_a_listing_with_one_definition():
    assert toc_html('def only():\n    pass\n', 'src') == ''


def test_a_module_demo_reads_its_docstring_as_notes_and_not_as_code():
    """The real modal demo: its physics module's docstring becomes prose, once."""
    notes = modal.DEMO.source_notes()
    code = modal.DEMO.source()
    assert notes and notes[0].startswith('A steel tuning fork')
    assert not code.startswith('"""')
    assert code.startswith('from dataclasses import dataclass')
    assert 'def fork_modes' in code
