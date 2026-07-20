"""Demo runner: list and run any registered example demo by name.

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
    uv run python examples/cli.py run poisson --save images/poisson.png
    uv run python examples/cli.py run poisson --mesh files/mesh_20x20.json
"""
import argparse
import logging
from pathlib import Path

from fem.mesh.femesh import FEMesh

from demo_registry import Demo
import meshing_demos
import refinement_demo
import solver_demos

DEFAULT_MESH_FILE = 'files/mesh_40x40.json'


def _build_registry() -> dict[str, Demo]:
    registry: dict[str, Demo] = {}
    for demo in solver_demos.DEMOS + meshing_demos.DEMOS + refinement_demo.DEMOS:
        if demo.name in registry:
            raise ValueError(f'duplicate demo name: {demo.name!r}')
        registry[demo.name] = demo
    return registry


def _description(demo: Demo) -> str:
    doc = demo.func.__doc__
    return doc.strip().splitlines()[0] if doc else '(no description)'


def _show_or_save(result, save_path):
    plotters = result if isinstance(result, list) else [result]

    if save_path is None:
        for plotter in plotters:
            plotter.show()
        return

    is_html = Path(save_path).suffix.lower() in ('.html', '.htm')
    for plotter in plotters:
        if plotter.anims and not is_html:
            raise NotImplementedError(
                f'{save_path!r} would only capture one static frame of an animation - '
                "save to a .html path instead (frames/slider/play all work there), "
                "or rerun without --save to view it interactively."
            )

    if len(plotters) == 1:
        plotters[0].save(save_path)
        return

    stem, dot, ext = save_path.rpartition('.')
    for i, plotter in enumerate(plotters):
        indexed_path = f'{stem}_{i}.{ext}' if dot else f'{save_path}_{i}'
        plotter.save(indexed_path)


def run_demo(demo: Demo, mesh_file: str, save_path: str | None) -> None:
    args = [FEMesh.load(mesh_file)] if demo.needs_mesh else []
    result = demo.func(*args)

    if demo.returns_plotter:
        _show_or_save(result, save_path)
    elif save_path is not None:
        raise NotImplementedError(
            f'{demo.name!r} manages its own display/output and does not support --save'
        )


def main():
    registry = _build_registry()

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
