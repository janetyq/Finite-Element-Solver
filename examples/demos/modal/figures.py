"""The figures and summary of the tuning fork demo, drawn from a `ForkStudy`."""
import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos._charts import share_panel_limits
from demos.modal import physics
from demos.modal.physics import E, RHO, ForkStudy, cantilever_hz, clamp, run, transverse_motion


def _mode_shape(s: ForkStudy, i):
    """Mode `i` as a deformed mesh, and the signed transverse motion colouring it."""
    transverse = transverse_motion(s.fork, i)
    scale = 0.12 * s.tine_length / np.abs(transverse).max()
    return s.fork.mode_mesh(i, scale), scale * transverse


def _hide_x_ticks(plotter, idx):
    """Drop the x-axis ticks on a tall, thin fork panel, where the millimetre-scale
    labels only collide; the y-axis carries the scale."""
    plotter.get_ax(idx).tick_params(axis='x', labelbottom=False, bottom=False)


def _modes_figure(s: ForkStudy, n_shown) -> Figure:
    modes = Plotter(1, n_shown, figsize=(2.9*n_shown, 6.0), axis_labels=False,
                    title="A tuning fork's natural modes and their pitches")
    for i in range(n_shown):
        shape, colour = _mode_shape(s, i)
        lim = float(np.abs(colour).max())
        tag = '  (the voice)' if i == s.voice else ''
        # No colorbar: the amplitude is arbitrary, and one caption below names the colour.
        # The symmetric clim keeps the still tine white in every panel.
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm',
                   clim=(-lim, lim), colorbar=False,
                   title=f'Mode {i+1}: {s.freqs[i]:.0f} Hz{tag}')
        modes.overlay_supports(s.mesh, clamp, idx=(0, i), coords=shape.vertices)
        _hide_x_ticks(modes, (0, i))
    # One shared vertical scale, so the tines line up across panels like the buckling modes.
    share_panel_limits(modes, n_shown)
    modes.fig.supxlabel(
        'Colour: sideways (transverse) displacement of the mode. Its sign and amplitude '
        'are arbitrary; the pattern of motion is what is physical.', fontsize='medium')
    return Figure(
        modes,
        'The fork rings in these shapes, each at its own pitch. The low modes come '
        'in pairs: the tips swing together (a rocking that shakes the stem, damped '
        'the moment the fork is held there) or oppositely, and the oppositely '
        'moving one, which leaves the stem still, is "the voice" the fork is made '
        'for.',
        'modes', thumbnail=True)


def _swing_figure(s: ForkStudy, n_frames) -> Figure:
    # Not a dynamics simulation: a standing-wave mode is a fixed shape times cos(omega t),
    # evaluated frame by frame. The colour is fixed; only the geometry moves.
    _, colour = _mode_shape(s, s.voice)
    amp = 0.12 * s.tine_length / np.abs(transverse_motion(s.fork, s.voice)).max()
    phases = np.cos(np.linspace(0, 2*np.pi, n_frames, endpoint=False))
    frames = [s.fork.mode_mesh(s.voice, amp*c) for c in phases]
    lim = float(np.abs(colour).max())
    swing = Plotter(1, 1, figsize=(4.6, 6.2),
                    title=f'The voice mode swinging: {s.freqs[s.voice]:.0f} Hz')
    swing.plot_animation(s.mesh, [colour]*n_frames, mode='colored', meshes=frames,
                         cmap='coolwarm', cbar_lims=(-lim, lim), label='sideways motion',
                         titles=['']*n_frames)
    swing.fig.supxlabel(
        "Not a time-stepped simulation: this is the mode's exact\n"
        'motion phi cos(omega t), one undamped, idealized mode\n'
        'at exaggerated amplitude. Only the shape and frequency\n'
        'are physical, not the size; a real fork mixes modes and\n'
        'rings down.', fontsize='small')
    return Figure(
        swing,
        'The voice mode as motion rather than a frozen shape: phi cos(omega t), the '
        'tines flexing apart and together at the natural frequency. Any free '
        'vibration is a sum of the modes, each ringing at its own rate; struck, a '
        'fork sheds the others and settles onto this one, so it sounds a single '
        'clean tone.',
        'swing')


def _tuning_law_figure(s: ForkStudy, n_shown) -> Figure:
    law = Plotter(1, 2, title='Against Euler-Bernoulli beam theory')
    curve = law.chart_ax(idx=(0, 0), xlabel='tine length L (m)', ylabel='voice frequency (Hz)')
    curve.loglog(s.sweep_lengths, s.sweep_freqs, 'o', color='tab:blue',
                 label=f'computed fork (slope {s.tuning_slope:.2f})')
    dense = np.linspace(s.sweep_lengths.min(), s.sweep_lengths.max(), 100)
    curve.loglog(dense, cantilever_hz(dense, s.tine_thickness), '-', color='tab:red',
                 alpha=0.6, label='ideal tine  f ~ 1/L^2')
    curve.axvline(s.tine_length, color='0.6', ls=':',
                  label=f'this fork ({s.tine_length*1000:.0f} mm)')
    curve.set_title('Pitch falls as 1/L^2')
    curve.grid(True, which='both', alpha=0.3)
    curve.legend(fontsize='small')

    bars = law.chart_ax(idx=(0, 1), ylabel='frequency (Hz)')
    x = np.arange(n_shown)
    bars.bar(x, s.freqs[:n_shown],
             color=['tab:red' if i == s.voice else 'tab:blue' for i in range(n_shown)])
    bars.axhline(440.0, color='0.4', ls='--', label='concert A (440 Hz)')
    bars.axhline(s.ideal_hz, color='tab:red', ls=':', alpha=0.6,
                 label=f'ideal tine ({s.ideal_hz:.0f} Hz)')
    bars.set_xticks(x, [str(i + 1) for i in range(n_shown)])
    bars.set_xlabel('mode')
    bars.set_title('First modes (voice in red)')
    bars.grid(True, axis='y', alpha=0.3)
    bars.legend(fontsize='small')
    return Figure(
        law,
        'Left: the fork is a pair of clamped-free tines, so beam theory sets its '
        'voice at f = (1.875)^2 / (2 pi) . (t / L^2) . sqrt(E* / 12 rho), the pitch '
        'falling as 1/L^2. Sweeping the tine length, the computed fork tracks that '
        'slope and sits a little below the ideal-tine line, because a real fork\'s '
        'base yields where beam theory assumes a rigid clamp. Right: this fork\'s '
        'first modes: the voice (red) lands near concert A, a few percent under '
        'the ideal tine for the same base-compliance reason.',
        'law')


def _ring_down_figure(s: ForkStudy) -> Figure:
    t, tip_x = s.ringing.t, s.tip_trace
    after_tap = t > s.tap_length
    envelope = np.abs(tip_x[after_tap]).max() * np.exp(-s.alpha * (t - s.tap_length) / 2)
    # The tip's spectrum: the tap excites every mode, and the peaks sit on the computed
    # frequencies.
    spectrum = np.abs(np.fft.rfft(tip_x[after_tap]))
    spectrum_f = np.fft.rfftfreq(int(after_tap.sum()), d=float(t[1] - t[0]))

    rung = Plotter(1, 2, title='Struck at the tip, ringing down')
    trace = rung.chart_ax(idx=(0, 0), xlabel='time (ms)', ylabel='tip sideways displacement (m)')
    trace.plot(1e3 * t, tip_x, color='tab:blue', lw=0.8, label='right tine tip')
    trace.plot(1e3 * t, envelope, '--', color='tab:red', alpha=0.7,
               label=f'exp(-alpha t / 2), alpha = {s.alpha:.0f} /s')
    trace.plot(1e3 * t, -envelope, '--', color='tab:red', alpha=0.7)
    trace.set_title(f'A {1e3 * s.tap_length:.2f} ms tap, then free vibration')
    trace.grid(True, alpha=0.3)
    trace.legend(fontsize='small')

    peaks = rung.chart_ax(idx=(0, 1), xlabel='frequency (Hz)', ylabel='amplitude')
    shown_f = spectrum_f <= 1.2 * s.freqs[-1]
    peaks.plot(spectrum_f[shown_f], spectrum[shown_f], color='tab:blue', lw=1.0)
    for i, f in enumerate(s.freqs):
        peaks.axvline(f, color='tab:red' if i == s.voice else '0.6', ls=':', alpha=0.8)
    peaks.set_title('Spectrum of the tip motion, computed modes dotted')
    peaks.grid(True, alpha=0.3)
    return Figure(
        rung,
        'The fork struck: a point impulse at one tine tip, then free vibration '
        'stepped by Newmark under mass-proportional damping. Left, the tip trace '
        'rings down inside the exp(-alpha t / 2) envelope every mode shares under '
        'that damping. Right, its spectrum: the tap excites the modes together and '
        'the peaks land on the frequencies the eigensolve found, the voice (red) '
        'among them.',
        'struck')


def _setup_figure(s: ForkStudy) -> Figure:
    built = Plotter(1, 2, figsize=(6.0, 7.0), title='From an outline to a meshed fork')
    built.plot(s.mesh, mode='mesh', idx=(0, 0), title=f'{len(s.mesh.elements)} triangles')
    _hide_x_ticks(built, (0, 0))
    built.plot(s.mesh, mode='bc', bc=clamp, idx=(0, 1), title='Clamped at the stem base')
    return Figure(
        built,
        'The fork is one non-convex outline (stem, base, two tines with a slot) '
        'meshed by Ruppert\'s algorithm, with no structured grid. It is held only at '
        'the stem base: that clamp grounds the structure (a free body has rigid-body '
        'modes the shift-invert eigensolve cannot factor through) and is where a fork '
        'is held, the one place that does not damp the voice.',
        'built', setup=True)


def _summary(s: ForkStudy, n_shown) -> str:
    voice_hz = s.freqs[s.voice]
    period_ms = 1e3 / voice_hz
    return (
        f'A steel tuning fork (E={E:.0e} Pa, rho={RHO:.0f} kg/m^3), meshed from its outline.\n'
        f'tine length x thickness   {s.tine_length*1000:.0f} x {s.tine_thickness*1000:.1f} mm\n'
        f'mesh                      {len(s.mesh.elements)} P2 triangles\n\n'
        f'ideal clamped tine (beam theory)   {s.ideal_hz:.0f} Hz\n'
        f'fork voice (mode {s.voice+1}, computed)      {voice_hz:.0f} Hz   '
        f'({100*(voice_hz/s.ideal_hz - 1):+.0f}%: the base is not a rigid clamp)\n'
        f'first {n_shown} modes (Hz)             '
        + '  '.join(f'{f:.0f}' for f in s.freqs[:n_shown]) + '\n'
        f'tuning law   f ~ L^{s.tuning_slope:.2f}         (beam-theory exponent -2)\n'
        f'struck: mass-proportional damping alpha = {s.alpha:.0f} /s, the voice at 1/e after '
        f'{s.ring_down_periods:.0f} periods ({s.ring_down_periods * period_ms:.0f} ms)'
    )


def demo(n_shown=4, n_frames=24, **kwargs) -> DemoResult:
    """Natural frequencies and modes of a steel tuning fork meshed from its outline,
    against beam theory; then the fork struck and ringing down."""
    s = run(**kwargs)
    return DemoResult([
        _modes_figure(s, n_shown),
        _swing_figure(s, n_frames),
        _tuning_law_figure(s, n_shown),
        _ring_down_figure(s),
        _setup_figure(s),
    ], text=_summary(s, n_shown))


DEMO = Demo('modal', demo, section='Solids & structures',
            show_source=physics,
            smoke_kwargs={'n_across_tine': 3, 'min_angle': 25, 'n_modes': 4, 'n_shown': 3,
                          'sweep_lengths': (0.088, 0.125), 'n_frames': 6,
                          'ring_periods': 3, 'steps_per_period': 12})
