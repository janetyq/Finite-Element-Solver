"""The figures and summary of the tuning fork demo, drawn from a `ForkStudy`."""
import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos._charts import hide_x_ticks, share_panel_limits
from demos.modal import physics
from demos.modal.physics import E, RHO, ForkStudy, cantilever_hz, clamp, run, transverse_motion


def _mode_shape(s: ForkStudy, i):
    """Mode `i` as a deformed mesh, and the signed transverse motion colouring it."""
    transverse = transverse_motion(s.fork, i)
    scale = 0.12 * s.tine_length / np.abs(transverse).max()
    return s.fork.mode_mesh(i, scale), scale * transverse




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
        hide_x_ticks(modes, (0, i))
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


def _struck_figure(s: ForkStudy, shown_periods, frames_per_period) -> Figure:
    """The struck fork in motion: the first `shown_periods` of the Newmark run, at
    `frames_per_period` frames each, so the tines are seen mid-swing rather than at
    one phase every frame."""
    t, tip_x = s.ringing.t, s.tip_trace
    steps_per_period = int(round(1.0 / (s.freqs[s.voice] * (t[1] - t[0]))))
    last = min(len(t) - 1, shown_periods * steps_per_period)
    stride = max(1, steps_per_period // frames_per_period)
    shown = list(range(0, last + 1, stride))
    # The real displacement is microns; one exaggeration scale for every frame, so the
    # motion stays in proportion as it decays. It is set by the slot, not the tine
    # length: the pinch drives the tips toward each other, and any more would draw
    # them passing through one another.
    verts = s.mesh.vertices
    half_slot = float(np.abs(verts[verts[:, 1] > verts[:, 1].max() - 1e-9, 0]).min())
    scale = 0.8 * half_slot / np.abs(tip_x).max()
    n_v = len(verts)
    sideways = [1e9 * s.ringing.u[i].reshape(-1, 2)[:n_v, 0] for i in shown]   # nm
    frames = []
    for i in shown:
        mesh = s.mesh.copy()
        mesh.vertices = mesh.vertices + scale * s.ringing.u[i].reshape(-1, 2)[:n_v]
        frames.append(mesh)
    lim = float(np.abs(sideways).max())
    # The title counts two things that should agree: the periods elapsed at the voice's
    # frequency, and the swings the tip has actually made (its upward zero crossings).
    f_voice = float(s.freqs[s.voice])
    upward = np.flatnonzero((tip_x[:-1] < 0) & (tip_x[1:] >= 0)) + 1
    titles = [f't = {1e3 * t[i]:.2f} ms\n'
              f'{t[i] * f_voice:.1f} periods at {f_voice:.0f} Hz\n'
              f'{int(np.sum(upward <= i))} tip swings counted' for i in shown]
    struck = Plotter(1, 1, figsize=(5.4, 6.4), title='Pinched at the tips and released')
    struck.plot_animation(s.mesh, sideways, mode='colored', meshes=frames, cmap='coolwarm',
                          cbar_lims=(-lim, lim), label='sideways displacement (nm)',
                          titles=titles)
    hide_x_ticks(struck, (0, 0))
    struck.fig.supxlabel(f'Displacement exaggerated {scale:.0f}x; the colour is to scale.\n'
                         f'Played at about one second per period; the real period is '
                         f'{1e3 / f_voice:.2f} ms.', fontsize='small')
    return Figure(
        struck,
        f'The fork struck, the first {shown_periods} periods of the voice: an equal and '
        'opposite impulse at the two tips, then free vibration by Newmark. The tines '
        'start in a mix of modes, shivering with the high ones, and under Rayleigh '
        'damping those die within a few periods, leaving the tines swinging apart and '
        'together in the voice. The pinch is chosen so the rocking mode (tips swinging '
        'the same way) is never excited: it is lower than the voice and would outlast '
        'it here, where nothing models the hand at the stem that damps it in a real '
        'fork.',
        'struck', frames=len(shown))


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
    sigma = s.decay_rate(s.voice)
    envelope = np.abs(tip_x[after_tap]).max() * np.exp(-sigma * (t - s.tap_length))
    # The tip's spectrum: the tap excites every mode, and the peaks sit on the computed
    # frequencies.
    spectrum = np.abs(np.fft.rfft(tip_x[after_tap]))
    spectrum_f = np.fft.rfftfreq(int(after_tap.sum()), d=float(t[1] - t[0]))

    rung = Plotter(1, 2, title='Pinched and released, ringing down')
    trace = rung.chart_ax(idx=(0, 0), xlabel='time (ms)', ylabel='tip sideways displacement (m)')
    trace.plot(1e3 * t, tip_x, color='tab:blue', lw=0.8, label='right tine tip')
    trace.plot(1e3 * t, envelope, '--', color='tab:red', alpha=0.7,
               label=f"the voice's decay, exp(-{sigma:.0f} t)")
    trace.plot(1e3 * t, -envelope, '--', color='tab:red', alpha=0.7)
    trace.set_title(f'A {1e3 * s.tap_length:.2f} ms pinch, then free vibration')
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
        'The same pinch over the whole run. Left, the tip trace rings down inside '
        "the voice's own decay envelope under the Rayleigh damping C = alpha M + "
        'beta K, whose beta term damps each mode in proportion to its frequency '
        'squared. Right, the spectrum of the trace: one peak, on the frequency the '
        'eigensolve found for the voice (red). The overtones the pinch excited '
        '(dotted, grey) have been damped out within the first few periods, and the '
        'rocking mode below the voice is absent because the pinch never excites it.',
        'ring-down')


def _setup_figure(s: ForkStudy) -> Figure:
    built = Plotter(1, 2, figsize=(6.0, 7.0), title='From an outline to a meshed fork')
    built.plot(s.mesh, mode='mesh', idx=(0, 0), title=f'{len(s.mesh.elements)} triangles')
    hide_x_ticks(built, (0, 0))
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
        f'struck: Rayleigh damping alpha = {s.damping.alpha:.0f} /s, '
        f'beta = {s.damping.beta:.2e} s; the voice at 1/e after '
        f'{s.ring_down_periods:.0f} periods ({s.ring_down_periods * period_ms:.0f} ms); '
        f'mode {n_shown} ({s.freqs[n_shown - 1]:.0f} Hz) after '
        f'{1e3 / s.decay_rate(n_shown - 1):.1f} ms'
    )


def demo(n_shown=4, shown_periods=6, frames_per_period=8, **kwargs) -> DemoResult:
    """Natural frequencies and modes of a steel tuning fork meshed from its outline,
    against beam theory; then the fork struck and ringing down."""
    s = run(**kwargs)
    return DemoResult([
        _modes_figure(s, n_shown),
        _struck_figure(s, shown_periods, frames_per_period),
        _ring_down_figure(s),
        _tuning_law_figure(s, n_shown),
        _setup_figure(s),
    ], text=_summary(s, n_shown))


DEMO = Demo('modal', demo, section='Solids & structures',
            show_source=physics,
            smoke_kwargs={'n_across_tine': 3, 'min_angle': 25, 'n_modes': 4, 'n_shown': 3,
                          'sweep_lengths': (0.088, 0.125), 'shown_periods': 2,
                          'ring_periods': 3, 'steps_per_period': 12})
