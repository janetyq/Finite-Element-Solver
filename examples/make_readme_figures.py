"""Regenerate the figures the README embeds, from the current demos.

    uv run python examples/make_readme_figures.py

The demos are the single source of truth. This renders a curated subset of the
gallery into a temporary directory and copies the chosen figures into `images/` under
README-friendly names, so a figure here is exactly what the gallery renders for the
same demo. Re-run it after a demo changes; the README points at committed files so it
still renders on GitHub, offline, and in a fork.

Only the demos named below are rendered, so this pays for a subset rather than the
whole gallery. Pass `--workers 1` to render them serially in this process (slower, but
sidesteps the multiprocessing import dance on Windows).
"""
import argparse
import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # render to buffers; this opens no windows

# examples/ on the path, so the demo modules and the gallery import the same way the
# CLI runs them.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli import build_registry
from gallery import build_gallery

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES = REPO_ROOT / 'images'

# (gallery figure stem, README file name). The stem is what the gallery writes:
# `<demo>` for a single-figure demo, `<demo>-<slug>` for one figure of several. The
# demo to render is read back off the stem (everything before the first '-'), so each
# named demo runs once even when the README shows two of its figures.
FIGURES: list[tuple[str, str]] = [
    # Meshing a domain
    ('mesh_from_svg-meshed', 'mesh_from_svg.png'),
    ('regions', 'regions.png'),
    # Solving PDEs
    ('poisson-fields', 'poisson.png'),
    ('potential_flow-flow', 'potential_flow.png'),
    ('heat-snapshots', 'heat.png'),
    ('wave-snapshots', 'wave.png'),
    ('robin-sweep', 'robin.png'),
    # Solids & structures
    ('linear_elastic-fields', 'linear_elastic.png'),
    ('bracket-fields', 'bracket.png'),
    ('bracket-singularity', 'bracket_singularity.png'),
    ('elasticity_models-stress', 'elasticity_models.png'),
    ('stress_concentration-built', 'stress_concentration_mesh.png'),
    ('stress_concentration-stress', 'stress_concentration.png'),
    ('elastic_3d', 'elastic_3d.png'),
    ('buckling-modes', 'buckling.png'),
    ('buckling-laws', 'buckling_laws.png'),
    ('modal-modes', 'modal.png'),
    ('modal-law', 'modal_law.png'),
    ('topology_optimization-final', 'topology_optimization.png'),
    # Accuracy & performance
    ('l2_projection', 'l2_projection.png'),
    ('convergence', 'convergence.png'),
    ('higher_order', 'higher_order.png'),
    ('refinement-after', 'refinement.png'),
]


def demo_of(stem: str) -> str:
    """The demo name a gallery stem belongs to: everything before the first '-'."""
    return stem.split('-', 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workers', type=int, default=None,
        help='how many demos to render in parallel (default: one per CPU); 1 renders '
             'them serially in this process')
    parser.add_argument(
        '--render-dir', default=None,
        help='keep the intermediate renders here instead of a self-cleaning temp dir, '
             'so a re-run can copy them again without re-rendering')
    args = parser.parse_args()

    registry = build_registry()
    demos = sorted({demo_of(stem) for stem, _ in FIGURES})
    print(f'Rendering {len(demos)} demos for {len(FIGURES)} README figures ...')

    IMAGES.mkdir(parents=True, exist_ok=True)
    # A named --render-dir persists (build_gallery with `only` keeps what is already
    # there); otherwise a temp dir cleaned on the way out.
    if args.render_dir is not None:
        keep = contextlib.nullcontext(args.render_dir)
    else:
        keep = tempfile.TemporaryDirectory()
    with keep as work:
        out = Path(work)
        build_gallery(registry, out, workers=args.workers, only=demos)

        missing = []
        for stem, name in FIGURES:
            src = out / 'img' / f'{stem}.png'
            if not src.exists():
                missing.append(stem)
                continue
            IMAGES.mkdir(parents=True, exist_ok=True)   # re-create if it was removed mid-run
            shutil.copyfile(src, IMAGES / name)
            print(f'  images/{name}  <-  {stem}')

    if missing:
        raise SystemExit(
            'these figures were not produced (renamed slug, or an animated-only '
            f'figure that saves no still): {", ".join(missing)}')
    print(f'\nWrote {len(FIGURES)} figures to {IMAGES}')


if __name__ == '__main__':
    main()
