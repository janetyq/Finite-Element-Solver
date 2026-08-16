"""Scalar measures of a stress or strain tensor.

A solve produces tensors; a plot or a failure criterion wants one number per
element. Reducing a tensor to a scalar is a choice, with one hard constraint: the
result must be **rotation invariant**. A material does not know which way the
axes point, so a scalar that changes when the frame turns describes the
bookkeeping, not the state.

Everything here takes full `(n_elements, d, d)` tensors, never the Voigt vectors
assembly packs; a norm or eigenvalue of the packed form is not the tensor's
(see `fem.forms.voigt_to_tensor`). `tests/test_invariants.py` checks invariance
by rotating the input.
"""
import numpy as np

from fem.typing import ElementField, FloatArray


def frobenius(tensor: FloatArray) -> ElementField:
    '''The Frobenius norm sqrt(A:A), batched over `(n_elements, d, d)`.

    Counts each off-diagonal entry the two times it appears in a symmetric tensor.
    '''
    return np.sqrt(np.einsum('eij,eij->e', tensor, tensor))


def trace(tensor: FloatArray) -> ElementField:
    '''The first invariant tr(A), batched.'''
    return np.einsum('eii->e', tensor)


def deviatoric(tensor: FloatArray) -> FloatArray:
    '''The trace-free part A - tr(A)/d * I, batched.

    Splits a tensor into the volume change it describes (the trace) and the shape
    change (what is left). Metal plasticity depends on the second alone, which von
    Mises measures.
    '''
    d = tensor.shape[-1]
    return tensor - (trace(tensor) / d)[:, None, None] * np.eye(d)


def pressure(stress: FloatArray) -> ElementField:
    '''Hydrostatic pressure -tr(sigma)/d: positive in compression, batched.'''
    return -trace(stress) / stress.shape[-1]


def von_mises(stress: FloatArray) -> ElementField:
    '''Von Mises equivalent stress sqrt(3/2 s:s), with s the deviatoric stress.

    The scalar a yield criterion compares against a material's tensile strength,
    and the usual thing to colour a stress plot by. Built from the deviator, so
    it ignores hydrostatic pressure: a body under uniform compression is not
    yielding, however large the pressure.

    Takes the **full 3x3 tensor**. A 2D solve must supply the out-of-plane
    component before calling this (see `LinearElasticMaterial.out_of_plane_stress`):
    under plane strain `sigma_zz` is nonzero, and dropping it does not give the
    2D von Mises stress, it gives a different number entirely.
    '''
    s = deviatoric(stress)
    return np.sqrt(1.5 * np.einsum('eij,eij->e', s, s))


def principal(tensor: FloatArray) -> FloatArray:
    '''Principal values, ascending, batched: `(n_elements, d)`.

    The eigenvalues of the tensor: the normal components in the frame where the
    shear terms vanish. `eigvalsh` assumes symmetry, which stress and strain both
    have, and returns them already sorted.
    '''
    # eigvalsh is typed as returning any float width; ours are float64 throughout.
    return np.asarray(np.linalg.eigvalsh(tensor), dtype=np.float64)


def max_shear(tensor: FloatArray) -> ElementField:
    '''Maximum shear stress (s_max - s_min)/2, from the principal values.'''
    values = principal(tensor)
    return (values[:, -1] - values[:, 0]) / 2.0
