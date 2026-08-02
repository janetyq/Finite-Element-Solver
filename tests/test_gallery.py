"""The gallery generator produces a browsable directory.

Built on a small hand-made registry rather than the real one: rendering all nineteen
demos takes minutes, and what needs covering here is the generator's four paths -- a
still figure, an animated figure rendered as frames, text output, and a demo skipped
for a missing dependency -- not the demos themselves, which `test_demos.py` runs.
"""
import json
import re
import sys
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

import solver_demos  # noqa: E402
from benchmark_assembly import demo_backends  # noqa: E402
from demo_registry import Demo, DemoResult  # noqa: E402
from gallery import build_gallery  # noqa: E402

from fem.mesh.ruppert import create_rect_mesh  # noqa: E402

# The smallest valid GIF: a 1x1 pixel. Enough to stand in for what PyVista writes.
ONE_PIXEL_GIF = (b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04'
                 b'\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D'
                 b'\x01\x00;')


def _writes_a_gif():
    """A demo whose whole output is a file, the way `3d` hands back a PyVista GIF."""
    Path('animation.gif').write_bytes(ONE_PIXEL_GIF)
    return DemoResult(artifacts=[Path('animation.gif')])


@pytest.fixture(scope='module')
def gallery(tmp_path_factory):
    # The gallery runs demos with no overrides -- neither arguments nor domain -- so
    # the cheap variants are bound here rather than declared on the Demo.
    small = partial(create_rect_mesh, corners=[[0, 0], [1, 1]], resolution=(8, 8))
    registry = {
        'poisson': Demo('poisson', solver_demos.demo_poisson_equation, domain=small),
        'topopt': Demo('topopt', partial(solver_demos.demo_topology_optimization, iters=2),
                       domain=small),
        'backends': Demo('backends', partial(demo_backends, sizes=(5,))),
        'absent': Demo('absent', solver_demos.demo_poisson_equation, domain=small,
                       smoke_requires='a_module_that_is_not_installed'),
        # Defined in this file, so it also stands for a demo from a module the index
        # has no section for.
        'gif_maker': Demo('gif_maker', _writes_a_gif),
    }
    out = tmp_path_factory.mktemp('gallery') / 'out'
    entries = build_gallery(registry, out)
    return out, {entry.name: entry for entry in entries}


def test_writes_an_index_and_a_page_per_demo(gallery):
    out, entries = gallery
    assert (out / 'index.html').exists()
    for name in entries:
        assert (out / f'{name}.html').exists(), f'no page for {name}'


def test_every_local_reference_resolves(gallery):
    """A gallery whose images 404 looks exactly like one that rendered nothing."""
    out, _ = gallery
    missing = []
    for page in out.glob('*.html'):
        text = page.read_text(encoding='utf-8')
        refs = re.findall(r'(?:src|href)="([^"]+)"', text)
        for encoded in re.findall(r'data-frames="([^"]+)"', text):
            refs += json.loads(encoded.replace('&quot;', '"'))
        missing += [f'{page.name} -> {r}' for r in refs
                    if not r.startswith(('http', '#')) and not (out / r).exists()]
    assert not missing, missing


def test_animated_figure_becomes_a_frame_sequence(gallery):
    out, entries = gallery
    animated = [p for p in entries['topopt'].panels if p.frames]
    assert animated, 'the topology optimisation animation produced no frames'
    frames = animated[0].frames
    assert len(frames) > 1
    assert all((out / f).exists() for f in frames)
    # Distinct frames: a stepping bug that redrew frame 0 every time would still
    # write the right number of files.
    assert len({(out / f).read_bytes() for f in frames}) > 1, 'every frame is identical'


def test_text_output_is_carried_onto_the_page(gallery):
    out, entries = gallery
    assert entries['backends'].text
    assert 'amg_cg' in (out / 'backends.html').read_text(encoding='utf-8')


def test_source_is_shown_on_the_page(gallery):
    """The figures are what a demo produced; the code is what a reader came for."""
    out, entries = gallery
    page = (out / 'poisson.html').read_text(encoding='utf-8')
    assert 'def demo_poisson_equation' in page
    assert 'BCType.DIRICHLET' in page


def test_source_survives_a_partial(gallery):
    """A preconfigured demo shows the function that was bound, not partial's own."""
    _out, entries = gallery
    assert 'def demo_topology_optimization' in entries['topopt'].source


def test_a_skipped_demo_still_shows_its_source(gallery):
    """Nothing about a missing optional dependency makes the code less worth reading."""
    out, _entries = gallery
    assert 'def demo_poisson_equation' in (out / 'absent.html').read_text(encoding='utf-8')


def test_index_is_grouped_by_area_not_alphabetically(gallery):
    """Sorted by name, the index opened on the three demos that show it least well."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    headings = re.findall(r'<h2 class="heading">([^<]+)</h2>', index)
    assert headings == ['Solving PDEs', 'Performance', 'Other demos']
    assert index.index('poisson.html') < index.index('Performance')


def test_a_demo_from_an_unlisted_module_still_appears(gallery):
    """Grouping must not be able to drop a demo it has no section for."""
    out, _entries = gallery
    assert 'gif_maker.html' in (out / 'index.html').read_text(encoding='utf-8')


def test_a_gif_artifact_becomes_the_thumbnail(gallery):
    """A demo that draws no figure can still have produced a picture."""
    out, _entries = gallery
    assert '<img src="animation.gif"' in (out / 'index.html').read_text(encoding='utf-8')


def test_a_text_only_demo_shows_its_output_on_its_card(gallery):
    """`backends` has nothing to draw, and an empty tile reads as a broken card."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'thumb-text' in index
    assert 'amg_cg' in index


def test_missing_dependency_is_reported_not_omitted(gallery):
    """A demo that could not run says so on its page, rather than vanishing."""
    out, entries = gallery
    assert entries['absent'].skipped
    assert not entries['absent'].panels
    assert 'Not rendered' in (out / 'absent.html').read_text(encoding='utf-8')
