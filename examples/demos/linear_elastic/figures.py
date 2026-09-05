"""The figures and summary of the cantilever demo, drawn from a `CantileverStudy`."""
from functools import partial

from demo_registry import Demo, DemoResult, Figure

from demos._charts import conditions_figure
from demos.linear_elastic import physics
from demos.linear_elastic.physics import CantileverStudy, run
from fem.mesh.structured import box_mesh
from fem.plot.plotter import Plotter


def _fields_figure(s: CantileverStudy) -> Figure:
    fields = Plotter(1, 2, figsize=(10.5, 4.2), title='Linear elasticity in 2D and 3D')
    fields.plot(s.solution.deformed_mesh(), s.solution.von_mises, mode='colored', idx=(0, 0),
                label='von Mises stress', title=f'2D: {len(s.mesh.elements)} triangles')
    # Only the boundary surface is drawn.
    fields.plot(s.solution_3d.deformed_mesh(), s.solution_3d.von_mises, mode='solid',
                idx=(0, 1), label='von Mises stress',
                title=f'3D: {len(s.box.elements)} tetrahedra')
    return Figure(
        fields,
        'The same clamp and load solved in 2D and 3D. The bending stress is largest '
        'at the clamp, with tension over the neutral axis and compression under it. '
        'The 3D solve carries a three-component displacement and recovers stress the '
        'same way, drawn on its boundary surface.',
        'fields')


def _invariants_figure(s: CantileverStudy) -> Figure:
    deformed = s.solution.deformed_mesh()
    invariants = Plotter(2, 2, title='Stress invariants of the same solve', panel_aspect=4.0)
    for i, (name, values) in enumerate(s.invariants):
        invariants.plot(deformed, values, mode='colored', idx=divmod(i, 2), title=name)
    return Figure(
        invariants,
        'Four rotation-invariant reductions of the same 2D stress tensor: von Mises, '
        'mean normal stress, the Tresca measure, and the largest tensile principal '
        'value.',
        'invariants')


def _conditions_figure(s: CantileverStudy) -> Figure:
    return conditions_figure(
        s.mesh, s.bc,
        'Clamped along the left edge, pulled down over the middle of the right one; '
        'everything between is traction-free. The 3D solve imposes the same clamp '
        'and tip load, one dimension up.',
        panel_aspect=4.0)


def _summary(s: CantileverStudy) -> str:
    return (f'2D triangles           {len(s.mesh.elements)}\n'
            f'3D tetrahedra          {len(s.box.elements)}\n'
            f'3D degrees of freedom  {3 * len(s.box.vertices)}\n'
            f'3D peak deflection     {s.tip_3d:.4f}')


def demo(mesh, **kwargs) -> DemoResult:
    """A cantilever under a tip load in 2D and 3D, with four stress invariants of the 2D
    solve."""
    s = run(mesh, **kwargs)
    return DemoResult([
        _fields_figure(s),
        _invariants_figure(s),
        _conditions_figure(s),
    ], text=_summary(s))


# The 2D cantilever whose domain this is, plus a 3D box the demo builds for itself.
DEMO = Demo('linear_elastic', demo, section='Solids & structures',
            domain=partial(box_mesh, [[0.0, 0.0], [4.0, 1.0]], (140, 35)), smoke_kwargs={'n_3d': 6},
            show_source=physics)
