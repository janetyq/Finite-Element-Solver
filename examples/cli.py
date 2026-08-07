"""Demo runner: list, run, or render the whole registry as a browsable gallery.

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
    uv run python examples/cli.py run poisson --save images/poisson.png
    uv run python examples/cli.py run poisson --mesh files/mesh_40x40.json
    uv run python examples/cli.py gallery
"""
import argparse
import inspect
import logging
from pathlib import Path

from fem.mesh.mesh import Mesh

from demo_registry import Demo, DemoResult
import benchmark_assembly
import meshing_demos
import refinement_demo
import solver_demos

DEFAULT_GALLERY_DIR = '.gallery'


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


def _save(result: DemoResult, save_path: str, name: str, dpi: float | None = None) -> None:
    animated = [f for f in result.figures if f.animated]
    stills = result.still_figures
    if animated and not stills:
        raise NotImplementedError(
            f"{name!r} produces only animated figures, and animation saving isn't "
            "implemented yet under the matplotlib backend (see Plotter.save's TODO) - "
            'rerun without --save to view it interactively.'
        )
    for figure in stills:
        figure.plotter.save(figure_path(save_path, figure, only=len(stills) == 1), dpi=dpi)


def deliver(result: DemoResult, save_path: str | None, name: str,
            dpi: float | None = None) -> None:
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
        _save(result, save_path, name, dpi)


def supports_interactive(demo: Demo) -> bool:
    """Whether this demo has a widget-driven mode, declared by taking the parameter."""
    return 'interactive' in inspect.signature(demo.func).parameters


def demo_mesh(demo: Demo, mesh_file: str | None) -> list[Mesh]:
    '''The mesh arguments to call `demo` with: its own domain, or `--mesh` instead.

    Passing `--mesh` to a demo that takes none is a mistake worth naming rather than
    silently ignoring -- it means the caller expected a different demo to be listening.
    '''
    if demo.domain is None:
        if mesh_file is not None:
            raise ValueError(f'{demo.name!r} builds no mesh, so --mesh has nothing to replace')
        return []
    return [Mesh.load(mesh_file) if mesh_file is not None else demo.domain()]


def run_demo(demo: Demo, mesh_file: str | None, save_path: str | None,
             interactive: bool = False, dpi: float | None = None) -> None:
    args = demo_mesh(demo, mesh_file)
    kwargs = {'interactive': True} if interactive else {}
    result = demo.func(*args, **kwargs)

    if not isinstance(result, DemoResult):
        raise TypeError(
            f'{demo.name!r} returned {type(result).__name__}; demos return a DemoResult '
            'so the caller decides what to show, save, or print.'
        )
    deliver(result, save_path, demo.name, dpi)


def main():
    registry = build_registry()

    parser = argparse.ArgumentParser(description='Run Finite-Element-Solver example demos.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('list', help='list all available demos')

    run_parser = subparsers.add_parser('run', help='run a demo by name')
    run_parser.add_argument('name', choices=sorted(registry), help='demo name')
    run_parser.add_argument(
        '--mesh', default=None,
        help='solve on this mesh JSON instead of the domain the demo builds for itself',
    )
    run_parser.add_argument(
        '--save', default=None,
        help='save the plot(s) to this path instead of showing them interactively',
    )
    run_parser.add_argument(
        '--dpi', type=float, default=None,
        help='resolution for --save; the default suits a screen, lower it for a figure '
             'whose file size matters more than its sharpness',
    )
    run_parser.add_argument(
        '--interactive', action='store_true',
        help='run the demo\'s widget-driven mode, where it has one (see `list`)',
    )

    gallery_parser = subparsers.add_parser(
        'gallery', help='render every demo to a browsable static gallery')
    gallery_parser.add_argument(
        '--out', default=DEFAULT_GALLERY_DIR,
        help=f'directory to write (replaced if it exists; default: {DEFAULT_GALLERY_DIR})')
    gallery_parser.add_argument(
        '--workers', type=int, default=None,
        help='how many demos to render in parallel (default: one per CPU); 1 renders '
             'them serially in this process')

    args = parser.parse_args()

    if args.command == 'list':
        for name in sorted(registry):
            print(f'{name}: {registry[name].description()}')
        return

    if args.command == 'gallery':
        from gallery import build_gallery

        print(f'Rendering {len(registry)} demos into {args.out}/ ...')
        build_gallery(registry, Path(args.out), workers=args.workers)
        print(f'\nOpen {Path(args.out).resolve() / "index.html"}')
        return

    demo = registry[args.name]
    if args.interactive and not supports_interactive(demo):
        supported = sorted(n for n, d in registry.items() if supports_interactive(d))
        parser.error(
            f'{args.name!r} has no interactive mode. --interactive applies to: '
            + ', '.join(supported)
        )

    run_demo(demo, args.mesh, args.save, interactive=args.interactive, dpi=args.dpi)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # show solver progress when running demos
    main()
