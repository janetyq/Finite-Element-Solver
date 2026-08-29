"""The gallery generator produces a browsable directory.

Built on a small hand-made registry covering the generator's paths: a still figure, an
animated figure rendered as frames, text output, and a demo skipped for a missing
dependency. `test_demos.py` runs the real demos.
"""
import json
import re
import shutil
from functools import partial
from pathlib import Path

import pytest

from demos import l2_projection, timing_benchmark, topology_optimization  # noqa: E402
from demo_registry import Demo, DemoResult, Figure  # noqa: E402
from gallery import build_gallery  # noqa: E402

from fem.mesh.structured import box_mesh  # noqa: E402
from fem.plot.plotter import Plotter  # noqa: E402
from fem.loads import Source

# The smallest valid GIF: a 1x1 pixel. Enough to stand in for a demo's rendered output.
ONE_PIXEL_GIF = (b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04'
                 b'\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D'
                 b'\x01\x00;')


def _writes_a_gif():
    """A demo whose whole output is a file rather than a figure the gallery renders."""
    Path('animation.gif').write_bytes(ONE_PIXEL_GIF)
    return DemoResult(artifacts=[Path('animation.gif')])


def _text_only():
    """A demo whose whole output is a table of numbers, with nothing for a Plotter to draw."""
    return DemoResult(text='n=5 dofs=375 time=0.01s')


def _poisson(mesh):
    """A cheap Poisson solve with one figure, standing in for a real demo."""
    from fem.conditions import Conditions
    from fem.boundary import Dirichlet
    from fem.physics.equations import Poisson
    from fem.regions import everywhere

    bc = Conditions(Dirichlet(everywhere(), 0))
    solution = Poisson().problem(mesh, bc + Source(1)).solve()
    plotter = Plotter()
    plotter.plot(mesh, solution.dofs, mode='colored')
    return DemoResult([Figure(plotter, 'the dome')])


def _two_figures(mesh, nominate=False):
    """Two figures, optionally with the second nominated as the card image."""
    first, second = Plotter(title='first'), Plotter(title='second')
    first.plot(mesh, mode='mesh')
    second.plot(mesh, mode='mesh')
    return DemoResult([Figure(first, 'how it was built', 'setup'),
                       Figure(second, 'what it says', 'result', thumbnail=nominate)])


def _result_then_setup(mesh):
    """A demo that shows what it found, then how the problem was posed."""
    result, conditions = Plotter(title='result'), Plotter(title='conditions')
    result.plot(mesh, mode='mesh')
    conditions.plot(mesh, mode='mesh')
    return DemoResult([Figure(result, 'what it found', 'fields'),
                       Figure(conditions, 'what was imposed', 'conditions', setup=True)])


def _registry():
    """The hand-made registry the module gallery is built from."""
    # The gallery runs demos with no overrides -- neither arguments nor domain -- so
    # the cheap variants are bound here rather than declared on the Demo.
    small = partial(box_mesh, corners=[[0, 0], [1, 1]], resolution=(8, 8))
    return {
        'poisson': Demo('poisson', _poisson, domain=small,
                        section='Meshing & solving PDEs'),
        'topopt': Demo('topopt', partial(topology_optimization.DEMO.func, iters=2),
                       domain=small, section='Solids & structures'),
        'backends': Demo('backends', partial(timing_benchmark.DEMO.func, sizes=(5,)),
                         section='Accuracy & performance'),
        'text_only': Demo('text_only', _text_only, section='Accuracy & performance'),
        'absent': Demo('absent', _poisson, domain=small,
                       section='Meshing & solving PDEs',
                       smoke_requires='a_module_that_is_not_installed'),
        'pipeline': Demo('pipeline', partial(_two_figures, nominate=True), domain=small,
                         section='Solids & structures'),
        'unnominated': Demo('unnominated', _two_figures, domain=small,
                            section='Solids & structures'),
        'posed': Demo('posed', _result_then_setup, domain=small,
                      section='Solids & structures'),
        # Declares no section, so it also stands for a demo the index has no heading for.
        'gif_maker': Demo('gif_maker', _writes_a_gif),
    }


@pytest.fixture(scope='module')
def gallery(tmp_path_factory):
    registry = _registry()
    out = tmp_path_factory.mktemp('gallery') / 'out'
    # Serial: several of these demos are defined in this test module, which a spawned
    # worker process could not import to unpickle. The parallel path -- which needs
    # importable, picklable demos -- is covered by `test_parallel_build` below.
    entries = build_gallery(registry, out, workers=1)
    return out, {entry.name: entry for entry in entries}


def test_parallel_build_renders_every_demo(tmp_path):
    """Across worker processes, each demo still produces its page and its figures. Only
    module-level demos are used, since a worker unpickles each by importing its module."""
    small = partial(box_mesh, corners=[[0, 0], [1, 1]], resolution=(8, 8))
    registry = {
        'poisson': Demo('poisson', _poisson, domain=small,
                        section='Meshing & solving PDEs'),
        'l2_projection': Demo('l2_projection', l2_projection.DEMO.func, domain=small,
                              section='Accuracy & performance'),
    }
    out = tmp_path / 'out'
    entries = build_gallery(registry, out, workers=2)

    assert {e.name for e in entries} == set(registry)
    assert [e.name for e in entries] == list(registry), 'registry order not preserved'
    for entry in entries:
        assert (out / f'{entry.name}.html').exists()
        assert entry.panels, f'{entry.name} rendered no figures'
        assert all((out / p.src).exists() for p in entry.panels)


def test_only_rebuilds_one_page_and_leaves_the_rest(gallery, tmp_path):
    """`only` is a single-page rebuild: it re-renders the named demo in place without
    disturbing the sibling pages or the index a full build wrote. Runs on a copy of the
    module's full build rather than building another."""
    built, _ = gallery
    out = tmp_path / 'out'
    shutil.copytree(built, out)
    index_before = (out / 'index.html').read_bytes()
    sibling_before = (out / 'text_only.html').read_bytes()

    entries = build_gallery(_registry(), out, workers=1, only=['poisson'])

    assert [e.name for e in entries] == ['poisson']
    assert (out / 'poisson.html').exists()
    assert entries[0].panels and all((out / p.src).exists() for p in entries[0].panels)
    assert (out / 'index.html').read_bytes() == index_before
    assert (out / 'text_only.html').read_bytes() == sibling_before


def test_only_into_a_fresh_dir_writes_the_page_but_no_index(tmp_path):
    """With no prior full build there is nothing to index faithfully, so `only` writes
    the page and leaves the index unwritten rather than one listing a lone demo."""
    registry = {'text_only': Demo('text_only', _text_only, section='Accuracy & performance')}
    out = tmp_path / 'out'
    entries = build_gallery(registry, out, workers=1, only=['text_only'])

    assert [e.name for e in entries] == ['text_only']
    assert (out / 'text_only.html').exists()
    assert not (out / 'index.html').exists()


def test_only_rejects_an_unknown_demo(tmp_path):
    """A mistyped name raises, naming the demo that does not exist."""
    registry = {'poisson': Demo('poisson', _text_only)}
    with pytest.raises(ValueError, match='no such demo'):
        build_gallery(registry, tmp_path / 'out', workers=1, only=['posson'])


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
    assert 'def _poisson' in page
    assert 'Dirichlet(' in page


def test_source_survives_a_partial(gallery):
    """A preconfigured demo shows the function that was bound, not partial's own."""
    _out, entries = gallery
    assert 'def demo(' in entries['topopt'].source


def test_a_skipped_demo_still_shows_its_source(gallery):
    """Nothing about a missing optional dependency makes the code less worth reading."""
    out, _entries = gallery
    assert 'def _poisson' in (out / 'absent.html').read_text(encoding='utf-8')


def test_index_is_grouped_by_section_not_alphabetically(gallery):
    """Sorted by name, the index opened on the demos that show it least well."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    headings = re.findall(r'<h2 class="heading">([^<]+)</h2>', index)
    assert headings == ['Meshing &amp; solving PDEs', 'Solids &amp; structures',
                        'Accuracy &amp; performance', 'Other demos']
    assert index.index('poisson.html') < index.index('topopt.html')


def test_a_demo_declaring_no_section_still_appears(gallery):
    """Grouping must not be able to drop a demo it has no heading for."""
    out, _entries = gallery
    assert 'gif_maker.html' in (out / 'index.html').read_text(encoding='utf-8')


def test_a_demo_can_nominate_which_figure_is_its_card_image(gallery):
    """Read in order, a pipeline demo opens on its setup; a card wants the result."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'pipeline-result.png' in index
    assert 'pipeline-setup.png' not in index


def test_the_first_figure_is_still_the_card_image_by_default(gallery):
    """Nominating is an opt-in; every demo that says nothing keeps what it had."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'unnominated-setup.png' in index
    assert 'unnominated-result.png' not in index


def test_a_setup_figure_gets_its_own_section_below_the_results(gallery):
    """How a problem was posed is the same kind of thing as the source that posed it,
    and belongs beside it rather than among what the demo found."""
    out, _entries = gallery
    page = (out / 'posed.html').read_text(encoding='utf-8')
    assert 'What was imposed' in page
    assert page.index('posed-fields.png') < page.index('What was imposed')
    assert page.index('What was imposed') < page.index('posed-conditions.png')
    assert page.index('posed-conditions.png') < page.index('>Source<')


def test_a_setup_figure_is_not_the_card_image(gallery):
    """A picture of the conditions imposed says nothing about what the demo found."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'posed-fields.png' in index
    assert 'posed-conditions.png' not in index


def test_a_gif_artifact_becomes_the_thumbnail(gallery):
    """A demo that draws no figure can still have produced a picture."""
    out, _entries = gallery
    assert '<img src="animation.gif"' in (out / 'index.html').read_text(encoding='utf-8')


def test_a_text_only_demo_shows_its_output_on_its_card(gallery):
    """A demo with nothing to draw still needs a card, not an empty tile."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'thumb-text' in index
    assert 'n=5 dofs=375' in index


def test_missing_dependency_is_reported_not_omitted(gallery):
    """A demo that could not run says so on its page, rather than vanishing."""
    out, entries = gallery
    assert entries['absent'].skipped
    assert not entries['absent'].panels
    assert 'Not rendered' in (out / 'absent.html').read_text(encoding='utf-8')


def test_a_skipped_demo_gives_its_reason_on_its_card_too(gallery):
    """The deployed gallery installs no extras, so `3d` is skipped there every time."""
    out, _entries = gallery
    index = (out / 'index.html').read_text(encoding='utf-8')
    assert 'thumb-note' in index
    assert 'a_module_that_is_not_installed' in index
