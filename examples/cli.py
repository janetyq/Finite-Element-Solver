"""Demo runner: list and run any registered example demo by name.

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
    uv run python examples/cli.py run poisson --save images/poisson.png
    uv run python examples/cli.py run poisson --mesh files/mesh_20x20.json
"""
import argparse
import logging
from pathlib import Path

from fem.mesh.mesh import Mesh

from demo_registry import Demo, DemoResult
import benchmark_assembly
import meshing_demos
import refinement_demo
import solver_demos

# Resolved against the repo, so `run poisson` works from any directory.
DEFAULT_MESH_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'mesh_40x40.json')


def build_registry() -> dict[str, Demo]:
    '''Every registered demo by name. Also the entry point `tests/test_demos.py` uses
    to run them all, so a demo is covered by the smoke test as soon as it is listed.'''
    registry: dict[str, Demo] = {}
    for demo in (solver_demos.DEMOS + meshing_demos.DEMOS + refinement_demo.DEMOS
                 + benchmark_assembly.DEMOS):
        if demo.name in registry:
            raise ValueError(f'duplicate demo name: {demo.name!r}')
        registry[demo.name] = demo
    return registry


def _description(demo: Demo) -> str:
    doc = demo.func.__doc__
    return doc.strip().splitlines()[0] if doc else '(no description)'


def figure_path(save_path: str, figure, only: bool) -> str:
    '''Where one figure of a multi-figure demo is written.

    A single-figure demo lands exactly on `--save`; more than one takes the figure's
    slug as a suffix, so `wave` saves as `wave-animation.png` / `wave-snapshots.png`
    rather than by position.
    '''
    if only:
        return save_path
    stem, dot, ext = save_path.rpartition('.')
    return f'{stem}-{figure.slug}.{ext}' if dot else f'{save_path}-{figure.slug}'


def _show(result: DemoResult) -> None:
    for figure in result.figures:
        figure.plotter.show()


def _save(result: DemoResult, save_path: str, name: str) -> None:
    animated = [f for f in result.figures if f.animated]
    stills = result.still_figures
    if animated and not stills:
        raise NotImplementedError(
            f"{name!r} produces only animated figures, and animation saving isn't "
            "implemented yet under the matplotlib backend (see Plotter.save's TODO) - "
            'rerun without --save to view it interactively.'
        )
    for figure in stills:
        figure.plotter.save(figure_path(save_path, figure, only=len(stills) == 1))


def deliver(result: DemoResult, save_path: str | None, name: str) -> None:
    '''Show or save the figures, print the text, report the files.

    The demo produced all of this and displayed none of it; every choice about where it
    goes is made here.
    '''
    if result.text:
        print(result.text)
    for path in result.artifacts:
        print(f'wrote {path}')

    if save_path is None:
        _show(result)
    else:
        _save(result, save_path, name)


def run_demo(demo: Demo, mesh_file: str, save_path: str | None) -> None:
    args = [Mesh.load(mesh_file)] if demo.needs_mesh else []
    result = demo.func(*args)

    if not isinstance(result, DemoResult):
        raise TypeError(
            f'{demo.name!r} returned {type(result).__name__}; demos return a DemoResult '
            'so the caller decides what to show, save, or print.'
        )
    deliver(result, save_path, demo.name)


def main():
    registry = build_registry()

    parser = argparse.ArgumentParser(description='Run Finite-Element-Solver example demos.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('list', help='list all available demos')

    run_parser = subparsers.add_parser('run', help='run a demo by name')
    run_parser.add_argument('name', choices=sorted(registry), help='demo name')
    run_parser.add_argument(
        '--mesh', default=DEFAULT_MESH_FILE,
        help=f'mesh JSON file to load, for demos that need one (default: {DEFAULT_MESH_FILE})',
    )
    run_parser.add_argument(
        '--save', default=None,
        help='save the plot(s) to this path instead of showing them interactively',
    )

    args = parser.parse_args()

    if args.command == 'list':
        for name in sorted(registry):
            print(f'{name}: {_description(registry[name])}')
        return

    run_demo(registry[args.name], args.mesh, args.save)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # show solver progress when running demos
    main()
