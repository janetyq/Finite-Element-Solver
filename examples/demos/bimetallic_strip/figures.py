"""The figures and summary of the bimetallic strip demo, drawn from a `BimetalStudy`."""
import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.bimetallic_strip import physics
from demos.bimetallic_strip.physics import BRASS, INVAR, BimetalStudy, run, theory_stress

# The real tip travel is a fraction of a millimetre on a 20 mm strip; the drawings
# scale the displacement so the curl is visible, and every panel says so.
DISPLAY_SCALE = 8.0


def _curl_figure(s: BimetalStudy) -> Figure:
    fractions = (0.25, 0.5, 1.0)
    vm_max = float(s.solution.von_mises.max())
    curl = Plotter(len(fractions), 1, panel_aspect=3.2, axis_labels=False,
                   figsize=(10.5, 7.5),
                   title=f'Heating curls the strip  (displacements drawn x{DISPLAY_SCALE:.0f})')
    for row, f in enumerate(fractions):
        # The problem is linear, so the state at f * dT_design is the design solve
        # scaled: the mesh warp, the stress, and the tip all by f.
        curl.plot(s.solution.deformed_mesh(scale=f * DISPLAY_SCALE),
                  s.solution.von_mises * f, mode='colored', idx=(row, 0),
                  cmap='viridis', clim=(0.0, vm_max), label='von Mises [MPa]',
                  title=f'dT = {f * s.dT_design:.0f} K   '
                        f'tip {1e3 * f * s.tip_fem:.0f} um')
        curl.get_ax((row, 0)).tick_params(left=False, bottom=False,
                                          labelleft=False, labelbottom=False)
    # One view for all three panels, sized by the most-curled strip, so the growing
    # curl is the only thing that changes from row to row.
    warped = s.solution.deformed_mesh(scale=DISPLAY_SCALE).vertices
    pad = 0.4
    for row in range(len(fractions)):
        ax = curl.get_ax((row, 0))
        ax.set_xlim(warped[:, 0].min() - pad, warped[:, 0].max() + pad)
        ax.set_ylim(warped[:, 1].min() - pad, warped[:, 1].max() + pad)
    return Figure(
        curl,
        'The strip at three temperature rises, brass below, invar above. Brass wants '
        'to grow more than the invar it is bonded to, so the mismatch bends the strip '
        'toward the invar side, and the motion is proportional to the temperature: '
        f'{1e3 * s.tip_per_kelvin:.1f} um of tip travel per kelvin. The colour is the '
        'internal stress doing the bending, largest along the bond line, with a spot '
        'at the clamped corner where the rivet also fights the expansion.',
        'curl', thumbnail=True)


def _switching_figure(s: BimetalStudy) -> Figure:
    switching = Plotter(1, 1, title='The thermostat closes its contact')
    ax = switching.chart_ax(xlabel='temperature rise dT [K]', ylabel='tip deflection [mm]')
    dT = np.linspace(0.0, s.dT_design, 200)
    ax.plot(dT, s.tip_theory * dT / s.dT_design, '-', color='tab:red', alpha=0.6,
            label='Timoshenko  kappa L^2 / 2')
    marks = np.linspace(0.0, s.dT_design, 6)
    ax.plot(marks, s.tip_fem * marks / s.dT_design, 'o', color='tab:blue',
            label='computed')
    ax.axhline(s.gap, color='0.4', ls='--', label=f'contact gap  {s.gap:.1f} mm')
    ax.axvline(s.dT_switch, color='tab:green', ls=':',
               label=f'switches at dT = {s.dT_switch:.0f} K')
    ax.grid(alpha=0.3)
    ax.legend()
    return Figure(
        switching,
        'Tip deflection against temperature rise: the computed strip on Timoshenko\'s '
        'line (the response is linear, so the design solve fixes the whole line). '
        f'A contact set {s.gap:.1f} mm away closes when the strip has risen that far, '
        f'at dT = {s.dT_switch:.0f} K: the strip is a thermometer and a switch in one '
        'part, which is the whole trade of a thermostat element.',
        'switching')


def _design_figure(s: BimetalStudy) -> Figure:
    design = Plotter(1, 1, title='How thick to make each layer')
    ax = design.chart_ax(xlabel='brass fraction of the thickness',
                         ylabel='tip travel per kelvin [um/K]')
    per_kelvin = s.length**2 / 2 / s.dT_design * 1e3
    dense = np.linspace(0.05, 0.95, 200) * s.thickness
    theory = [physics.theory_bending(split, s.thickness, s.dT_design)[1] * per_kelvin
              for split in dense]
    ax.plot(dense / s.thickness, theory, '-', color='tab:red', alpha=0.6, label='Timoshenko')
    ax.plot(s.splits / s.thickness, s.split_kappa_fem * per_kelvin, 'o',
            color='tab:blue', label='computed')
    ax.axvline(s.best_split / s.thickness, color='0.6', ls=':',
               label=f'optimum at {s.best_split / s.thickness:.2f}')
    ax.grid(alpha=0.3)
    ax.legend()
    return Figure(
        design,
        'The same strip with the bond line moved: too little of either metal and the '
        'mismatch has nothing to work against, so the sensitivity peaks between. The '
        f'optimum puts slightly more brass than invar ({s.best_split / s.thickness:.2f} '
        'of the thickness, at the thickness ratio sqrt(E_invar / E_brass)), and the '
        'curve is flat around it, which is why commercial bimetals are near-equal '
        'layers.',
        'design')


def _stress_figure(s: BimetalStudy) -> Figure:
    stress = Plotter(1, 1, title='The stress inside the strip')
    ax = stress.chart_ax(xlabel='axial stress sigma_xx [MPa]', ylabel='height y [mm]')
    for y0, y1 in [(0.0, s.split), (s.split, s.thickness)]:
        y = np.linspace(y0, y1, 50)
        ax.plot(theory_stress(y, s.split, s.thickness, s.dT_design), y, '-',
                color='tab:red', alpha=0.6)
    ax.plot(s.profile_stress, s.profile_y, 'o', color='tab:blue', label='computed')
    ax.plot([], [], '-', color='tab:red', alpha=0.6, label='theory')
    ax.axhline(s.split, color='0.4', ls='--', label='bond line')
    ax.axvline(0.0, color='0.8', lw=1)
    ax.grid(alpha=0.3)
    ax.legend()
    return Figure(
        stress,
        'The axial stress through the thickness at mid-span. Each fibre is held away '
        'from its free thermal length, so the profile is linear within a layer and '
        'jumps at the bond, where the two metals disagree most; the whole profile '
        'carries no net force and no net moment, which is what makes the strip bend '
        'rather than stretch. The bond-line stress is what fatigues and delaminates '
        'real bimetal elements.',
        'stress')


def _setup_figure(s: BimetalStudy) -> Figure:
    setup = Plotter(2, 1, panel_aspect=4.0, figsize=(10.5, 4.0),
                    title='How the strip is posed')
    _, alpha = physics.layer_properties(s.mesh, s.split)
    setup.plot(s.mesh, alpha * 1e6, mode='colored', idx=(0, 0), cmap='coolwarm',
               label='alpha [1e-6 / K]',
               title=f'{BRASS.name} below the bond line, {INVAR.name} above')
    setup.plot(s.mesh, mode='bc', conditions=s.bc, idx=(1, 0),
               title='clamped at the root, free everywhere else')
    return Figure(
        setup,
        f'A {s.length:.0f} x {s.thickness:.0f} mm strip, two bonded layers of equal '
        f'thickness: {BRASS.name} (alpha = {BRASS.alpha * 1e6:.0f}e-6/K) below, '
        f'{INVAR.name} (alpha = {INVAR.alpha * 1e6:.1f}e-6/K) above. It is clamped at '
        'the left end and heated uniformly, the element being small and conductive '
        'enough to be isothermal, so everything it does comes from the mismatch. '
        'Plane stress on P2 elements: a strip is thin and free in z, and a thin strip '
        'in bending is where the constant-strain triangle locks.',
        'setup', setup=True)


def _summary(s: BimetalStudy) -> str:
    return (f'curvature at dT = {s.dT_design:.0f} K:\n'
            f'  computed      {s.kappa_fem:.3e} /mm\n'
            f'  Timoshenko    {s.kappa_theory:.3e} /mm   '
            f'({100 * s.kappa_error:.2f}% apart)\n'
            f'tip travel        {1e3 * s.tip_per_kelvin:.1f} um/K  '
            f'({s.tip_fem:.2f} mm at the design rise)\n'
            f'switching rise    {s.dT_switch:.0f} K for a {s.gap:.1f} mm contact gap\n'
            f'best brass share  {s.best_split / s.thickness:.2f} of the thickness')


def demo(**kwargs) -> DemoResult:
    """A bimetallic strip turning heat into motion: the thermostat's element, its
    switching temperature, and its layer design, against Timoshenko's bimetal formula."""
    s = run(**kwargs)
    return DemoResult([
        _curl_figure(s),
        _switching_figure(s),
        _design_figure(s),
        _stress_figure(s),
        _setup_figure(s),
    ], text=_summary(s))


# Builds its own strip: the mesh must carry the bond line, so it takes no domain.
DEMO = Demo('bimetallic_strip', demo, section='Solids & structures',
            show_source=physics,
            smoke_kwargs={'n_length': 48, 'n_thickness': 4, 'splits': (0.25, 0.5, 0.75)})
