"""Render every registered demo into a static gallery: one page per demo, plus an index.

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

from fem.mesh.mesh import Mesh

from demo_registry import Demo, DemoResult

IMAGES = 'img'


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


def run_demo(demo: Demo, mesh: Mesh, out_dir: Path) -> Entry:
    """Run one demo and collect everything it produced into an `Entry`.

    Demos write their artifacts relative to the working directory, so this runs with
    `out_dir` as the cwd -- the same arrangement `tests/test_demos.py` uses to keep
    stray files out of the repo.
    """
    entry = Entry(demo.name, demo.description())

    skip = _missing_dependency(demo)
    if skip is not None:
        entry.skipped = skip
        return entry

    before = set(out_dir.iterdir())
    args = [mesh] if demo.needs_mesh else []
    # No overrides: the gallery shows what `cli.py run <name>` shows.
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
:root { color-scheme: light dark; --fg: #111; --muted: #666; --bg: #fff; --line: #e3e3e3; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8e8; --muted: #9a9a9a; --bg: #16181c; --line: #2c2f36; }
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
.card img { width: 100%; aspect-ratio: 4/3; object-fit: cover; background: #fff; display: block; }
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

    return _page(f'{entry.name} - FEM demos', '\n'.join(parts), PLAYER_JS)


def _index_page(entries: list[Entry]) -> str:
    cards = []
    for entry in entries:
        thumb = entry.panels[0].src if entry.panels else ''
        badge = ''
        if any(p.frames for p in entry.panels):
            badge = '<span class="badge">animated</span>'
        elif entry.skipped:
            badge = '<span class="badge">not rendered</span>'
        image = (f'<img src="{thumb}" alt="" loading="lazy">' if thumb
                 else '<img alt="" style="background:transparent">')
        cards.append(
            f'<a class="card" href="{entry.name}.html">{image}'
            f'<span class="meta"><h2>{html.escape(entry.name)}{badge}</h2>'
            f'<p>{html.escape(entry.description)}</p></span></a>'
        )

    rendered = sum(1 for e in entries if not e.skipped)
    body = (
        '<h1>Finite Element Solver &mdash; demo gallery</h1>\n'
        f'<p class="sub">{rendered} of {len(entries)} demos rendered. '
        'Generated by <code>examples/cli.py gallery</code>.</p>\n'
        f'<div class="grid">\n{chr(10).join(cards)}\n</div>'
    )
    return _page('FEM demo gallery', body)


def build_gallery(registry: dict[str, Demo], out_dir: Path, mesh_file: str) -> list[Entry]:
    """Render every demo into `out_dir` and write the pages. Returns what was collected."""
    import os

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)          # a stale image is worse than a missing one
    out_dir.mkdir(parents=True)

    mesh = Mesh.load(mesh_file)
    entries = []
    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        for name in sorted(registry):
            entry = run_demo(registry[name], mesh, out_dir)
            entries.append(entry)
            print(f'  {name}' + (f' - skipped ({entry.skipped})' if entry.skipped else ''))
    finally:
        os.chdir(cwd)

    for entry in entries:
        (out_dir / f'{entry.name}.html').write_text(_demo_page(entry), encoding='utf-8')
    (out_dir / 'index.html').write_text(_index_page(entries), encoding='utf-8')

    return entries
