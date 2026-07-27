"""Scalar measures of a stress or strain tensor.

A solve produces tensors; a plot, a failure criterion, or an error indicator
wants one number per element. Reducing a tensor to a scalar is a *choice*, and
the choice has a hard constraint: the result must be **rotation invariant**. A
material does not know which way the axes point, so a scalar that changes when
the coordinate frame turns is not a property of the state -- it is an artifact of
how the state was written down.

That constraint is the reason this module exists rather than the reductions being
inlined where they are needed. The functions here take **full tensors**
`(n_elements, d, d)`, never Voigt vectors. Voigt packing is an assembly-internal
optimization that stores a symmetric tensor as a vector and -- for strain -- folds
a factor of two into the shear terms so that a dot product reproduces the tensor
contraction. That packing is correct for the contraction it was designed for and
wrong for everything else: `norm` of a Voigt vector counts the off-diagonal terms
once where the tensor holds them twice, and for strain the engineering shear
double-counts them instead. Both errors are invisible until you rotate the frame.

So the boundary is: Voigt lives inside `fem.forms`, tensors come out, and nothing
here needs to know the convention. `tests/test_invariants.py` checks invariance
directly, by rotating the input.
"""
import numpy as np

from fem.typing import ElementField, FloatArray


def frobenius(tensor: FloatArray) -> ElementField:
    '''The Frobenius norm sqrt(A:A), batched over `(n_elements, d, d)`.

    The most basic invariant -- the root of the sum of every squared component,
    which counts each off-diagonal entry the two times it appears in a symmetric
    tensor. Unlike a norm taken over Voigt components it is genuinely invariant.
    '''
    return np.sqrt(np.einsum('eij,eij->e', tensor, tensor))


def trace(tensor: FloatArray) -> ElementField:
    '''The first invariant tr(A), batched.'''
    return np.einsum('eii->e', tensor)


def deviatoric(tensor: FloatArray) -> FloatArray:
    '''The trace-free part A - tr(A)/d * I, batched.

    Splits a tensor into the volume change it describes (the trace) and the shape
    change (what is left). Metal plasticity depends on the second alone, which is
    what von Mises measures.
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
    it ignores hydrostatic pressure -- a body under uniform compression is not
    yielding, however large the pressure.

    Takes the **full 3x3 tensor**. A 2D solve must supply the out-of-plane
    component before calling this (see `LinearElasticMaterial.out_of_plane_stress`):
    under plane strain `sigma_zz` is nonzero, and dropping it does not give the
    2D von Mises stress, it gives a different number entirely.
    '''
    s = deviatoric(stress)
    return np.sqrt(1.5 * np.einsum('eij,eij->e', s, s))


def principal(tensor: FloatArray) -> FloatArray:
    '''Principal values, ascending, batched -- `(n_elements, d)`.

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
