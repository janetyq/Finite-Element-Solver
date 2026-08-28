"""Scalar reductions of a tensor must be rotation invariant: these rotate the input and
assert the output holds still. A norm over Voigt components is not invariant (it
counts an off-diagonal term once where the tensor holds it twice), which
`test_voigt_norm_is_not_invariant_but_frobenius_is` pins as a contrast.
"""
import numpy as np
import pytest

from fem.physics.forms import voigt_to_tensor
from fem.post.invariants import (
    deviatoric,
    frobenius,
    max_shear,
    principal,
    pressure,
    trace,
    von_mises,
)


def rotation_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_3d(a: float, b: float, c: float) -> np.ndarray:
    """An arbitrary 3D rotation, as a product of three axis rotations."""
    def rx(t):
        return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    return rx(a) @ ry(b) @ rz(c)


def rotate(tensor: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The same tensor read in a rotated frame: R A R^T, batched."""
    return np.einsum('ij,ejk,lk->eil', R, tensor, R)


def symmetric_batch(n: int = 12, d: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, d, d))
    return A + np.swapaxes(A, -2, -1)


INVARIANTS = [frobenius, trace, pressure, von_mises, max_shear]


@pytest.mark.parametrize('reduce', INVARIANTS, ids=lambda f: f.__name__)
def test_reductions_are_invariant_under_rotation(reduce):
    """The defining property: turning the coordinate frame must not move the number."""
    tensor = symmetric_batch()
    R = rotation_3d(0.3, -1.1, 2.7)

    np.testing.assert_allclose(reduce(rotate(tensor, R)), reduce(tensor), atol=1e-12)


def test_principal_values_are_invariant_under_rotation():
    """Eigenvalues are frame-independent by construction; this pins that the
    batched implementation preserves it (and the ascending order)."""
    tensor = symmetric_batch()
    R = rotation_3d(-0.7, 0.4, 1.9)

    np.testing.assert_allclose(principal(rotate(tensor, R)), principal(tensor), atol=1e-12)


def test_voigt_norm_is_not_invariant_but_frobenius_is():
    """Pure shear in 2D, and the same state in a frame rotated 45 degrees: a Voigt norm
    reports tau and then tau*sqrt(2); the Frobenius norm reports tau*sqrt(2) both times."""
    tau = 3.0
    shear = np.array([[0.0, 0.0, tau]])          # Voigt [sxx, syy, sxy]
    rotated = np.array([[tau, -tau, 0.0]])       # the same state at 45 degrees

    voigt_norm = np.linalg.norm(shear, axis=-1), np.linalg.norm(rotated, axis=-1)
    assert not np.isclose(voigt_norm[0], voigt_norm[1])
    np.testing.assert_allclose(voigt_norm[0], tau)
    np.testing.assert_allclose(voigt_norm[1], tau * np.sqrt(2))

    tensors = [voigt_to_tensor(v, shear_factor=1.0) for v in (shear, rotated)]
    np.testing.assert_allclose(frobenius(tensors[0]), tau * np.sqrt(2))
    np.testing.assert_allclose(frobenius(tensors[1]), tau * np.sqrt(2))


def test_von_mises_ignores_hydrostatic_pressure():
    """Uniform pressure causes no distortion, so it must not register as equivalent stress."""
    tensor = symmetric_batch()
    shifted = tensor + 100.0 * np.eye(3)

    np.testing.assert_allclose(von_mises(shifted), von_mises(tensor), atol=1e-10)


def test_von_mises_of_uniaxial_tension_is_the_axial_stress():
    """The calibration case: a bar pulled along one axis yields when the axial
    stress reaches the tensile strength, so von Mises must return exactly it."""
    sigma = 250.0
    tensor = np.zeros((1, 3, 3))
    tensor[0, 0, 0] = sigma

    np.testing.assert_allclose(von_mises(tensor), sigma)


def test_von_mises_of_pure_shear_is_sqrt_three_tau():
    """The other textbook case, and the one that distinguishes von Mises from a
    plain norm: pure shear of magnitude tau gives sqrt(3)*tau."""
    tau = 5.0
    tensor = np.zeros((1, 3, 3))
    tensor[0, 0, 1] = tensor[0, 1, 0] = tau

    np.testing.assert_allclose(von_mises(tensor), np.sqrt(3) * tau)


def test_deviatoric_part_is_trace_free():
    np.testing.assert_allclose(trace(deviatoric(symmetric_batch())), 0.0, atol=1e-12)


def test_pressure_is_positive_in_compression():
    """Sign convention: a body squeezed from all sides is under positive pressure."""
    tensor = -2.0 * np.eye(3)[None]
    np.testing.assert_allclose(pressure(tensor), 2.0)


# -- the Voigt unpacking the invariants sit on top of -------------------------


def test_voigt_to_tensor_places_2d_components():
    """[sxx, syy, sxy] -> the symmetric 2x2, with the shear in both slots."""
    got = voigt_to_tensor(np.array([[1.0, 2.0, 3.0]]), shear_factor=1.0)
    np.testing.assert_allclose(got[0], [[1.0, 3.0], [3.0, 2.0]])


def test_voigt_to_tensor_halves_engineering_shear():
    """Strain packs gamma = 2*eps, so the tensor entry is half the packed one."""
    got = voigt_to_tensor(np.array([[0.0, 0.0, 4.0]]), shear_factor=2.0)
    np.testing.assert_allclose(got[0], [[0.0, 2.0], [2.0, 0.0]])


def test_voigt_to_tensor_places_3d_components_in_assembly_order():
    """3D Voigt rows are ordered [xx, yy, zz, xy, yz, xz], as `strain_displacement` writes them."""
    got = voigt_to_tensor(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), shear_factor=1.0)
    np.testing.assert_allclose(
        got[0], [[1.0, 4.0, 6.0], [4.0, 2.0, 5.0], [6.0, 5.0, 3.0]]
    )


def test_voigt_to_tensor_rejects_an_unknown_component_count():
    with pytest.raises(ValueError, match='3 .2D. or 6'):
        voigt_to_tensor(np.zeros((2, 4)))
