"""Numerical utilities: source/field functions and a finite-difference order check."""
from collections.abc import Callable

import numpy as np

from fem.typing import FloatArray


def bump_function(
    vertices: FloatArray, center: FloatArray, mag: float = 100, size: float = 0.5
) -> FloatArray:
    '''A radial cosine bump of height `mag` and radius `size`, centered at `center`.

    `mag * cos(pi/2 * r/size)` inside the radius, falling to zero at the rim, and
    flat zero outside it. Seeds smooth initial conditions in the transient demos
    and tests.
    '''
    distances = np.linalg.norm(vertices - center, axis=1)
    return np.where(distances < size, mag * np.cos(np.pi / 2 * distances / size), 0.0)


def central_difference_order(
    function: Callable[[FloatArray], FloatArray | float],
    directional_derivative: Callable[[FloatArray], FloatArray | float],
    u: FloatArray,
    *,
    eps: FloatArray | None = None,
    n_directions: int = 4,
    seed: int = 0,
) -> float:
    '''Fitted order of a central-difference check of `directional_derivative` at `u`.

    For each of a few random directions `d`, compares the central difference
        (function(u + eps*d) - function(u - eps*d)) / (2*eps)
    against `directional_derivative(d)` across a sweep of `eps`, and returns the slope
    of log(error) vs log(eps). A correct derivative makes the error O(eps^2), so the
    slope is ~2; the default `eps` stays in that regime, above the roundoff floor.

    `directional_derivative(d)` is the derivative at `u` along `d`: `gradient @ d` for
    a scalar `function`, `hessian @ d` for a vector one.
    '''
    rng = np.random.default_rng(seed)
    if eps is None:
        eps = np.logspace(-4, -1, 8)
    directions = rng.standard_normal((n_directions, *np.shape(u)))
    per_direction = tuple(range(1, directions.ndim))  # every axis but the direction index
    directions /= np.linalg.norm(directions, axis=per_direction, keepdims=True)
    exact = [np.asarray(directional_derivative(d), dtype=float) for d in directions]

    errors = []
    for step in eps:
        squared = 0.0
        for d, exact_d in zip(directions, exact):
            approx = (function(u + step * d) - function(u - step * d)) / (2 * step)
            squared += float(np.sum((np.asarray(approx, dtype=float) - exact_d) ** 2))
        errors.append(np.sqrt(squared))
    return float(np.polyfit(np.log(eps), np.log(errors), 1)[0])
