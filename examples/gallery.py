"""Render every registered demo into a static gallery: one page per demo, plus an index.

A page carries what the demo produced (figures, printed text, files) and the source
that produced it, so the gallery reads as documentation rather than as a picture book.

Everything is written as plain files with relative links (no CDN, no embedded data),
so the output directory works the same opened from disk or served over HTTP.

Animated figures become a directory of frames plus a small player, rather than one
animated file. A frame is ~40 KB of PNG where the same animation through matplotlib's
`to_jshtml` is ~150 KB of base64 per frame, and separate files let the browser load
them lazily and cache them.

    uv run python examples/cli.py gallery
"""
import html
import os
import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # render to buffers; a gallery run opens no windows

from demo_registry import Demo, DemoResult

IMAGES = 'img'

# Frames written per animated figure. Rasterizing frames is the largest single cost of
# a gallery build (the heat demo solves in 0.4s and spends 24s drawing its run), and
# the player steps at 120ms, so twenty images is a two-second loop. The solve keeps
# every step it took; this is only how much of it becomes PNGs.
FRAMES_PER_PLAYER = 20

# The index in the order a newcomer should meet the project: build a domain, solve on
# it, apply that to solids, then ask whether the answer is right and fast. Each demo
# names its own section (`Demo.section`); this is the order they appear in, and within
# a section demos keep the order they were registered in.
SECTIONS: list[str] = [
    'Meshing a domain',
    'Solving PDEs',
    'Solids & structures',
    'Accuracy & performance',
]

# A demo naming no section, or one not listed above, still appears, under this
# heading, rather than being dropped by a grouping that did not know about it.
OTHER_SECTION = 'Other demos'


@dataclass
class Panel:
    """One figure on a demo's page: a still image, or a directory of frames to play."""
    caption: str
    src: str                       # still image, or the first frame of a sequence
    frames: list[str] = field(default_factory=list)
    thumbnail: bool = False        # nominated as the card image; see `Figure.thumbnail`
    setup: bool = False            # how the problem was posed; see `Figure.setup`


@dataclass
class Entry:
    name: str
    description: str
    panels: list[Panel] = field(default_factory=list)
    text: str | None = None
    artifacts: list[str] = field(default_factory=list)
    skipped: str | None = None
    source: str = ''
    section: str = ''              # which heading of the index it belongs under


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
            written = figure.plotter.save_frames(str(frame_dir / '{:03d}.png'),
                                                 max_frames=FRAMES_PER_PLAYER)
            frames = [f'{IMAGES}/{stem}/{Path(p).name}' for p in written]
            panels.append(Panel(figure.caption, frames[0], frames, figure.thumbnail,
                                figure.setup))
        else:
            figure.plotter.save(str(images / f'{stem}.png'))
            panels.append(Panel(figure.caption, f'{IMAGES}/{stem}.png',
                                thumbnail=figure.thumbnail, setup=figure.setup))

        figure.plotter.close()

    return panels


def run_demo(demo: Demo, out_dir: Path) -> Entry:
    """Run one demo and collect everything it produced into an `Entry`.

    Runs with its own temporary directory as the cwd, so a demo that writes a file
    relative to the working directory is detected without racing the other demos a
    parallel build runs alongside it. Figures go straight into the shared images dir
    under `out_dir`, whose names are demo-prefixed and so cannot collide; whatever
    else the demo wrote is then moved out of the temporary directory into `out_dir`.
    Callable in a worker process because it depends on nothing but its arguments.
    """
    out_dir = Path(out_dir)
    entry = Entry(demo.name, demo.description(), source=demo.source(), section=demo.section)

    skip = _missing_dependency(demo)
    if skip is not None:
        entry.skipped = skip
        return entry

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        os.chdir(work_dir)
        try:
            # No overrides, not the arguments and not the domain: the gallery shows
            # what `cli.py run <name>` shows.
            args = [demo.domain()] if demo.domain is not None else []
            result = demo.func(*args)
            entry.panels = _render_figures(result, demo.name, out_dir)
            entry.text = result.text
            written = sorted(p for p in work_dir.iterdir() if p.is_file())
        finally:
            os.chdir(cwd)

        # Whatever the demo wrote lands beside the page; name them relative to it.
        for path in written:
            shutil.move(str(path), str(out_dir / path.name))
        entry.artifacts = [path.name for path in written]
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
.intro { margin: 0 0 .6rem; max-width: 46rem; font-size: 1.05rem; }
.grid { display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); }
.card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
        text-decoration: none; display: flex; flex-direction: column; }
/* `contain`, not `cover`: these figures are wide, and cropping one to a 4:3 tile
   showed a fifth of a single panel; `robin` is four panels across 2000x500. */
.card img, .card .thumb-text, .card .thumb-note, .card .thumb-empty {
  width: 100%; aspect-ratio: 4/3; background: #fff; display: block; }
.card img { object-fit: contain; }
.card .thumb-text { margin: 0; border: 0; border-radius: 0; padding: .7rem .8rem;
                    font-size: .5rem; line-height: 1.5; color: #111; overflow: hidden; }
.card .thumb-note { display: flex; align-items: center; padding: 1.25rem;
                    color: #666; font-size: .85rem; }
.card .meta { padding: .7rem .85rem; }
.card .meta p { margin: .2rem 0 0; color: var(--muted); font-size: .82rem; }
.badge { font-size: .72rem; color: var(--muted); border: 1px solid var(--line);
         border-radius: 99px; padding: .05rem .5rem; margin-left: .35rem; vertical-align: middle; }
figure { margin: 0 0 2.5rem; }
/* Bounded in both directions, not stretched to the column. Every figure is written at
   the same dpi, so `width: 100%` blew a single square panel up to three times the
   height of a three-panel strip, and past its own resolution, so it was soft as well
   as huge. */
figure img { display: block; margin: 0 auto; width: auto; height: auto;
             max-width: 100%; max-height: 30rem;
             border: 1px solid var(--line); border-radius: 8px; background: #fff; }
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

    parts += [_panel_html(panel, i) for i, panel in enumerate(entry.panels)
              if not panel.setup]

    if entry.text:
        parts.append(f'<pre>{html.escape(entry.text)}</pre>')
    for name in entry.artifacts:
        if name.lower().endswith('.gif'):
            parts.append(f'<figure><img src="{name}" alt="{html.escape(name)}">'
                         f'<figcaption>{html.escape(name)}</figcaption></figure>')
        else:
            parts.append(f'<p class="sub">Wrote <a href="{name}">{html.escape(name)}</a>.</p>')

    # Sectioned off rather than left among the results, for the same reason the source
    # is: it answers "how would I state this problem" rather than "what does it show".
    setup = [(i, p) for i, p in enumerate(entry.panels) if p.setup]
    if setup:
        parts.append('<h2 class="heading">What was imposed</h2>')
        parts += [_panel_html(panel, i) for i, panel in setup]

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
    for title in SECTIONS:
        members = [e for e in entries if e.section == title]
        claimed.update(e.name for e in members)
        if members:
            grouped.append((title, members))

    rest = [e for e in entries if e.name not in claimed]
    if rest:
        grouped.append((OTHER_SECTION, rest))
    return grouped


def _thumbnail(entry: Entry) -> str:
    """The tile at the top of a card.

    A demo with no figure is not a demo with nothing to show, and an invisible tile
    reads as a broken card. `backends` produces a table of timings; a demo skipped for
    a missing dependency has at least the reason.

    A demo may nominate which of its figures this is; the first is only the default.
    """
    nominated = next((p for p in entry.panels if p.thumbnail), None)
    chosen = nominated or (entry.panels[0] if entry.panels else None)
    src = chosen.src if chosen else next(
        (a for a in entry.artifacts if a.lower().endswith('.gif')), '')
    if src:
        return f'<img src="{src}" alt="" loading="lazy">'
    if entry.text:
        preview = html.escape('\n'.join(entry.text.splitlines()[:5]))
        return f'<pre class="thumb-text">{preview}</pre>'
    if entry.skipped:
        return f'<span class="thumb-note">Not rendered: {html.escape(entry.skipped)}.</span>'
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
        # What the project is, before what the page is: the count answered a question
        # nobody arrives with.
        '<p class="intro">A finite element solver written from scratch in Python &mdash; '
        'meshing, assembly, and solves for the Poisson, heat, wave and elasticity '
        'equations, in 2D and 3D. Every demo below is rendered beside the code that '
        'produced it.</p>',
        f'<p class="sub">{rendered} of {len(entries)} rendered, by '
        '<code>examples/cli.py gallery</code>, on every push to <code>main</code>.</p>',
    ]
    for title, members in _sections(entries):
        body.append(f'<h2 class="heading">{html.escape(title)}</h2>')
        body.append(f'<div class="grid">\n{chr(10).join(card(e) for e in members)}\n</div>')
    return _page('FEM demo gallery', '\n'.join(body))


def _selected_demos(registry: dict[str, Demo], only: Iterable[str] | None) -> list[Demo]:
    """The demos to render, in registry order.

    Registry order, not alphabetical: it is the order each module lists its demos in,
    which is the order they appear within a section, and `pool.map` preserves it. `only`
    names a subset to render; an unknown name is an error rather than a silent no-op, so
    a mistyped `--only` fails loudly instead of rebuilding nothing.
    """
    if only is None:
        return list(registry.values())
    wanted = set(only)
    unknown = sorted(wanted - registry.keys())
    if unknown:
        raise ValueError(f'no such demo(s): {", ".join(unknown)}')
    return [demo for name, demo in registry.items() if name in wanted]


def build_gallery(registry: dict[str, Demo], out_dir: Path,
                  workers: int | None = None,
                  only: Iterable[str] | None = None) -> list[Entry]:
    """Render demos into `out_dir` and write their pages. Returns what was collected.

    By default every demo is rendered: `out_dir` is rebuilt from scratch and an
    `index.html` linking all of them is written. Pass `only` to render just those demos'
    pages in place: `out_dir` and its other pages are kept, and the index is left
    untouched, because rebuilding it faithfully would mean re-running every demo to
    recover its card. It is the quick single-page rebuild; a full run refreshes the index.

    The demos are independent (each renders its own figures under demo-prefixed names),
    so they run across `workers` processes, bounding the build by the slowest single
    demo rather than the sum. `workers` defaults to the machine's CPU count; `workers=1`
    runs them in this process, which is what the generator's own tests use.
    """
    out_dir = Path(out_dir).resolve()
    demos = _selected_demos(registry, only)

    if only is None:
        if out_dir.exists():
            shutil.rmtree(out_dir)      # a stale image is worse than a missing one
        out_dir.mkdir(parents=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = min(len(demos), os.cpu_count() or 1)

    if workers <= 1:
        entries = [run_demo(demo, out_dir) for demo in demos]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(run_demo, demos, [out_dir] * len(demos)))

    for entry in entries:
        print(f'  {entry.name}' + (f' - skipped ({entry.skipped})' if entry.skipped else ''))

    for entry in entries:
        (out_dir / f'{entry.name}.html').write_text(_demo_page(entry), encoding='utf-8')
    if only is None:
        (out_dir / 'index.html').write_text(_index_page(entries), encoding='utf-8')

    return entries
