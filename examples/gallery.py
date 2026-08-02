"""Render every registered demo into a static gallery: one page per demo, plus an index.

A page carries what the demo produced -- figures, printed text, files -- and the source
that produced it, so the gallery reads as documentation rather than as a picture book.

Everything is written as plain files with relative links -- no CDN, no embedded data --
so the output directory works the same opened from disk or served over HTTP.

Animated figures become a directory of frames plus a small player, rather than one
animated file. A frame is ~40 KB of PNG where the same animation through matplotlib's
`to_jshtml` is ~150 KB of base64 per frame, and separate files let the browser load
them lazily and cache them.

    uv run python examples/cli.py gallery
"""
import html
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # render to buffers; a gallery run opens no windows

from demo_registry import Demo, DemoResult

IMAGES = 'img'

# The index reads in the order a newcomer should meet the project -- what it solves,
# then how a domain becomes a mesh, then the two supporting pieces. Alphabetical opened
# on `3d`, `adaptive_refinement` and `backends`: a demo needing an optional dependency,
# one whose headline feature is not wired up yet, and a table of timings.
SECTIONS: list[tuple[str, str]] = [
    ('solver_demos', 'Solving PDEs'),
    ('meshing_demos', 'Meshing'),
    ('refinement_demo', 'Adaptive refinement'),
    ('benchmark_assembly', 'Performance'),
]

# Demos from any other module still appear, under this heading, rather than being
# dropped by a grouping that did not know about them.
OTHER_SECTION = 'Other demos'


@dataclass
class Panel:
    """One figure on a demo's page: a still image, or a directory of frames to play."""
    caption: str
    src: str                       # still image, or the first frame of a sequence
    frames: list[str] = field(default_factory=list)


@dataclass
class Entry:
    name: str
    description: str
    panels: list[Panel] = field(default_factory=list)
    text: str | None = None
    artifacts: list[str] = field(default_factory=list)
    skipped: str | None = None
    source: str = ''
    module: str = ''               # which section of the index it belongs under


def _missing_dependency(demo: Demo) -> str | None:
    if demo.smoke_requires is None:
        return None
    try:
        __import__(demo.smoke_requires)
    except ImportError:
        return f'needs the optional dependency {demo.smoke_requires!r}, which is not installed'
    return None


def _render_figures(result: DemoResult, name: str, out_dir: Path) -> list[Panel]:
    images = out_dir / IMAGES
    images.mkdir(parents=True, exist_ok=True)
    panels = []

    for figure in result.figures:
        stem = f'{name}-{figure.slug}' if figure.slug else name

        if figure.animated:
            frame_dir = images / stem
            frame_dir.mkdir(exist_ok=True)
            written = figure.plotter.save_frames(str(frame_dir / '{:03d}.png'))
            frames = [f'{IMAGES}/{stem}/{Path(p).name}' for p in written]
            panels.append(Panel(figure.caption, frames[0], frames))
        else:
            figure.plotter.save(str(images / f'{stem}.png'))
            panels.append(Panel(figure.caption, f'{IMAGES}/{stem}.png'))

        figure.plotter.close()

    return panels


def run_demo(demo: Demo, out_dir: Path) -> Entry:
    """Run one demo and collect everything it produced into an `Entry`.

    Demos write their artifacts relative to the working directory, so this runs with
    `out_dir` as the cwd -- the same arrangement `tests/test_demos.py` uses to keep
    stray files out of the repo.
    """
    entry = Entry(demo.name, demo.description(), source=demo.source(), module=demo.module())

    skip = _missing_dependency(demo)
    if skip is not None:
        entry.skipped = skip
        return entry

    before = set(out_dir.iterdir())
    # No overrides -- not the arguments, and not the domain: the gallery shows what
    # `cli.py run <name>` shows.
    args = [demo.domain()] if demo.domain is not None else []
    result = demo.func(*args)

    entry.panels = _render_figures(result, demo.name, out_dir)
    entry.text = result.text
    # Whatever the demo wrote lands in out_dir; name them relative to the page.
    entry.artifacts = sorted(
        p.name for p in set(out_dir.iterdir()) - before if p.is_file()
    )
    return entry


# --- rendering ------------------------------------------------------------------

STYLE = """
:root { color-scheme: light dark; --fg: #111; --muted: #666; --bg: #fff; --line: #e3e3e3;
        --code: #f6f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8e8; --muted: #9a9a9a; --bg: #16181c; --line: #2c2f36;
          --code: #1d2026; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 2.5rem 1.25rem 4rem; max-width: 62rem; background: var(--bg);
       color: var(--fg); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 0 0 .35rem; }
a { color: inherit; }
.sub { color: var(--muted); margin: 0 0 2rem; }
.grid { display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); }
.card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
        text-decoration: none; display: flex; flex-direction: column; }
/* `contain`, not `cover`: these figures are wide, and cropping one to a 4:3 tile
   showed a fifth of a single panel -- `robin` is four panels across 2000x500. */
.card img, .card .thumb-text, .card .thumb-empty {
  width: 100%; aspect-ratio: 4/3; background: #fff; display: block; }
.card img { object-fit: contain; }
.card .thumb-text { margin: 0; border: 0; border-radius: 0; padding: .7rem .8rem;
                    font-size: .5rem; line-height: 1.5; color: #111; overflow: hidden; }
.card .meta { padding: .7rem .85rem; }
.card .meta p { margin: .2rem 0 0; color: var(--muted); font-size: .82rem; }
.badge { font-size: .72rem; color: var(--muted); border: 1px solid var(--line);
         border-radius: 99px; padding: .05rem .5rem; margin-left: .35rem; vertical-align: middle; }
figure { margin: 0 0 2.5rem; }
figure img { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
figcaption { color: var(--muted); font-size: .9rem; margin-top: .5rem; }
.player { display: flex; align-items: center; gap: .75rem; margin-top: .6rem; }
.player button { font: inherit; padding: .25rem .9rem; border-radius: 6px;
                 border: 1px solid var(--line); background: transparent; color: inherit; cursor: pointer; }
.player input { flex: 1; }
.player .count { color: var(--muted); font-size: .85rem; font-variant-numeric: tabular-nums; }
pre { overflow-x: auto; padding: .9rem 1rem; border: 1px solid var(--line);
      border-radius: 8px; font-size: .85rem; }
pre.source { line-height: 1.45; tab-size: 4; }
.heading { font-size: 1rem; margin: 2.5rem 0 .6rem; text-transform: uppercase;
           letter-spacing: .06em; color: var(--muted); }
.run { margin: 0 0 2rem; }
.run code { font-size: .85rem; background: var(--code); border: 1px solid var(--line);
            border-radius: 6px; padding: .25rem .5rem; }
.note { border-left: 3px solid var(--line); padding-left: .9rem; color: var(--muted); }
"""

PLAYER_JS = """
document.querySelectorAll('[data-frames]').forEach(function (root) {
  var frames = JSON.parse(root.dataset.frames);
  var img = root.querySelector('img');
  var range = root.querySelector('input');
  var button = root.querySelector('button');
  var count = root.querySelector('.count');
  var timer = null;

  function show(i) {
    img.src = frames[i];
    range.value = i;
    count.textContent = (Number(i) + 1) + ' / ' + frames.length;
  }
  function stop() { clearInterval(timer); timer = null; button.textContent = 'Play'; }
  function step() { show((Number(range.value) + 1) % frames.length); }

  range.max = frames.length - 1;
  range.addEventListener('input', function () { stop(); show(range.value); });
  button.addEventListener('click', function () {
    if (timer) { stop(); return; }
    button.textContent = 'Pause';
    timer = setInterval(step, 120);
  });
  show(0);
  frames.forEach(function (src) { new Image().src = src; });  // warm the cache
});
"""


def _page(title: str, body: str, script: str = '') -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{html.escape(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n'
        f'{body}\n'
        + (f'<script>{script}</script>\n' if script else '')
        + '</body>\n</html>\n'
    )


def _panel_html(panel: Panel, index: int) -> str:
    caption = html.escape(panel.caption)
    if not panel.frames:
        return (f'<figure>\n<img src="{panel.src}" alt="{caption}">\n'
                f'<figcaption>{caption}</figcaption>\n</figure>')

    frames = html.escape(str(panel.frames).replace("'", '"'), quote=True)
    return (
        f'<figure data-frames="{frames}" id="player-{index}">\n'
        f'<img src="{panel.src}" alt="{caption}">\n'
        '<div class="player">'
        '<button type="button">Play</button>'
        '<input type="range" min="0" value="0" step="1">'
        '<span class="count"></span>'
        '</div>\n'
        f'<figcaption>{caption}</figcaption>\n</figure>'
    )


def _demo_page(entry: Entry) -> str:
    parts = [
        '<p class="sub"><a href="index.html">&larr; all demos</a></p>',
        f'<h1>{html.escape(entry.name)}</h1>',
        f'<p class="sub">{html.escape(entry.description)}</p>',
        f'<p class="run"><code>uv run python examples/cli.py run {html.escape(entry.name)}</code></p>',
    ]
    if entry.skipped:
        parts.append(f'<p class="note">Not rendered: {html.escape(entry.skipped)}.</p>')

    parts += [_panel_html(panel, i) for i, panel in enumerate(entry.panels)]

    if entry.text:
        parts.append(f'<pre>{html.escape(entry.text)}</pre>')
    for name in entry.artifacts:
        if name.lower().endswith('.gif'):
            parts.append(f'<figure><img src="{name}" alt="{html.escape(name)}">'
                         f'<figcaption>{html.escape(name)}</figcaption></figure>')
        else:
            parts.append(f'<p class="sub">Wrote <a href="{name}">{html.escape(name)}</a>.</p>')

    # The figures are what the demo produced; this is what produced them. Stating a
    # problem to this solver is a dozen readable lines, which is the claim the gallery
    # was otherwise making only in pictures.
    if entry.source:
        parts.append('<h2 class="heading">Source</h2>')
        parts.append(f'<pre class="source">{html.escape(entry.source)}</pre>')

    return _page(f'{entry.name} - FEM demos', '\n'.join(parts), PLAYER_JS)


def _sections(entries: list[Entry]) -> list[tuple[str, list[Entry]]]:
    """Entries under their headings, in `SECTIONS` order, empty sections omitted."""
    grouped = []
    claimed = set()
    for module, title in SECTIONS:
        members = [e for e in entries if e.module == module]
        claimed.update(e.name for e in members)
        if members:
            grouped.append((title, members))

    rest = [e for e in entries if e.name not in claimed]
    if rest:
        grouped.append((OTHER_SECTION, rest))
    return grouped


def _thumbnail(entry: Entry) -> str:
    """The tile at the top of a card.

    A demo with no figure is not a demo with nothing to show: `3d` renders through
    PyVista and hands back a GIF, and `backends` produces a table of timings. Both
    used to get an invisible tile, which read as a broken card.
    """
    src = entry.panels[0].src if entry.panels else next(
        (a for a in entry.artifacts if a.lower().endswith('.gif')), '')
    if src:
        return f'<img src="{src}" alt="" loading="lazy">'
    if entry.text:
        preview = html.escape('\n'.join(entry.text.splitlines()[:5]))
        return f'<pre class="thumb-text">{preview}</pre>'
    return '<span class="thumb-empty"></span>'


def _index_page(entries: list[Entry]) -> str:
    def card(entry: Entry) -> str:
        badge = ''
        if any(p.frames for p in entry.panels):
            badge = '<span class="badge">animated</span>'
        elif entry.skipped:
            badge = '<span class="badge">not rendered</span>'
        return (
            f'<a class="card" href="{entry.name}.html">{_thumbnail(entry)}'
            f'<span class="meta"><h2>{html.escape(entry.name)}{badge}</h2>'
            f'<p>{html.escape(entry.description)}</p></span></a>'
        )

    rendered = sum(1 for e in entries if not e.skipped)
    body = [
        '<h1>Finite Element Solver &mdash; demo gallery</h1>',
        f'<p class="sub">{rendered} of {len(entries)} demos rendered. '
        'Generated by <code>examples/cli.py gallery</code>.</p>',
    ]
    for title, members in _sections(entries):
        body.append(f'<h2 class="heading">{html.escape(title)}</h2>')
        body.append(f'<div class="grid">\n{chr(10).join(card(e) for e in members)}\n</div>')
    return _page('FEM demo gallery', '\n'.join(body))


def build_gallery(registry: dict[str, Demo], out_dir: Path) -> list[Entry]:
    """Render every demo into `out_dir` and write the pages. Returns what was collected."""
    import os

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)          # a stale image is worse than a missing one
    out_dir.mkdir(parents=True)

    entries = []
    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        for name in sorted(registry):
            entry = run_demo(registry[name], out_dir)
            entries.append(entry)
            print(f'  {name}' + (f' - skipped ({entry.skipped})' if entry.skipped else ''))
    finally:
        os.chdir(cwd)

    for entry in entries:
        (out_dir / f'{entry.name}.html').write_text(_demo_page(entry), encoding='utf-8')
    (out_dir / 'index.html').write_text(_index_page(entries), encoding='utf-8')

    return entries
