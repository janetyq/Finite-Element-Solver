"""The figures and summary of the elasticity models demo, drawn from a `StretchStudy`."""
from functools import partial

import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos._charts import conditions_figure
from demos.elasticity_models import physics
from demos.elasticity_models.physics import StretchStudy, run
from domains import square


def _stress_figure(s: StretchStudy) -> Figure:
    # The four fields span roughly a factor of ten (Neo-Hookean's interior against
    # St-VK's), so one shared *log* colour scale carries them all: each panel keeps its
    # own structure while magnitudes still compare across panels. The stress diverges at
    # the clamped corners, where the imposed displacement is singular (it grows without
    # bound under refinement, a mesh artefact, not a material fact), so the scale is
    # capped at the 99th percentile of the pooled field, those corner nodes saturate, and
    # the per-panel median is the exact number.
    vms = s.von_mises
    pooled = np.concatenate([np.ravel(v) for v in vms])
    # A positive floor (log needs one) and a cap that drops the singular corners, both
    # from the pooled field so the four scales are identical.
    lo = float(np.percentile(pooled[pooled > 0], 1))
    hi = float(np.percentile(pooled, 99))

    plotter = Plotter(1, 4, title=f'One {s.stretch:.0%} stretch, four ways to model it')
    for i, ((name, solution), vm) in enumerate(zip(s.solutions, vms)):
        plotter.plot(solution.deformed_mesh(), vm, mode='colored', idx=(0, i),
                     label='von Mises stress (log)', clim=(lo, hi), log_scale=True,
                     colorbar=(i == len(s.solutions) - 1),
                     title=f'{name}\nmedian {np.median(vm):.0f}')
    return Figure(
        plotter,
        'Four solves of one boundary condition: the left edge clamped, the right edge '
        'pulled to a 50% stretch, nothing else loaded. One shared log colour scale '
        'spans all four panels, so magnitudes compare directly across them; the median '
        'under each title is the exact figure, and the singular clamped corners '
        'saturate at the top.',
        'stress',
        body=[
            'Panels 1 and 2 are the same physics, small-strain linear elasticity, '
            'solved two ways: panel 1 as a direct linear solve (K u = f), panel 2 by '
            'minimising the elastic energy with Newton. They reach the same '
            'displacement to machine precision, so the colour difference is not a '
            'physics difference. Panel 1 reports engineering stress sigma = D:eps, '
            'force per unit original area; panel 2 reports true (Cauchy) stress, force '
            'per unit deformed area. Because stretching thins the cross-section, the '
            'same internal force reads higher as true stress, about 50% higher here, '
            'near the 1.5 stretch factor. The two coincide only at small strain, when '
            'the geometry barely moves.',

            'Panels 2, 3 and 4 hold the method fixed and change the material. Panel 2 '
            'is small strain; panels 3 and 4 are nonlinear, finite-strain elasticity. '
            'St-Venant-Kirchhoff (panel 3) is the simplest finite-strain model, '
            'small-strain elasticity rewritten on the Green-Lagrange strain. Its '
            'strength is that it is frame-indifferent, a rigid rotation stores no '
            'energy, which small strain gets wrong; its weakness is that its energy is '
            'polynomial in the stretch, so it over-stiffens steeply in tension and '
            'loses stability in strong compression. In the plot it is by far the most '
            'stressed panel (median 396, several times the interior stress of the '
            'others), the block lit up and fighting the stretch.',

            'Neo-Hookean (panel 4) is a rubber-elasticity model written in the '
            'invariants of C = F^T F with a ln J volumetric term. Its strength is a '
            'physically realistic large-strain response, it does not over-stiffen in '
            'tension and stays stable in compression and near incompressibility; its '
            'cost is a more expensive, non-polynomial evaluation that must guard '
            'against element inversion. In the plot it is the calmest finite-strain '
            'panel, moderate stress everywhere, its median (102) sitting right next to '
            'the linear baseline (107).',
        ])


def _conditions_figure(s: StretchStudy) -> Figure:
    return conditions_figure(
        s.mesh, s.bc,
        'Both ends are Dirichlet. The left is held at zero and the right is displaced '
        f'to {s.stretch:.0%} of the width. Nothing is loaded; the stress above is what '
        'it costs to hold that shape.')


def _summary(s: StretchStudy) -> str:
    return (f'displacement, linear solve vs energy minimisation: '
            f'relative difference {s.drift:.1e}\n'
            f'minimised elastic energy: {s.minimised_energy:.4g}')


def demo(mesh, **kwargs) -> DemoResult:
    """One clamped block stretched by a linear solve, by energy minimisation, and by two
    finite-strain material laws."""
    s = run(mesh, **kwargs)
    return DemoResult([_stress_figure(s), _conditions_figure(s)], text=_summary(s))


# A square keeps the deformed and undeformed shapes comparable at a glance.
DEMO = Demo('elasticity_models', demo, section='Solids & structures',
            domain=partial(square, 60), show_source=physics)
