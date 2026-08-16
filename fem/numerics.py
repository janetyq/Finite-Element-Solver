"""Numerical utilities: source/field functions, the SIMP smoothing matrix, and
finite-difference gradient/Hessian checks.
"""
import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import KDTree

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray, SparseMatrix


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


def calculate_smoothing_matrix(mesh: Mesh, r: float) -> SparseMatrix:
    '''Row-normalized cone weights over the element centers within radius `r`.

    The SIMP sensitivity filter: an element's smoothed sensitivity is a weighted
    mean of the sensitivities within `r` of it, under the weight `r - distance`
    falling linearly to zero at the radius. Filtering the sensitivity is what keeps
    the optimizer off checkerboard designs, and `r` sets the design's feature size.

    Sparse, off a KD-tree neighbour query: an element couples only to the ones
    inside its radius, so only those pairs are stored. Under the usual choice of a
    radius tracking the element size, that is a bounded number of neighbours each
    and the filter costs O(n_elements); hold `r` fixed while refining and the
    neighbour count grows with the mesh, though the stored pairs stay a small
    fraction of all n^2 of them.

    Rows sum to 1, except at `r = 0`, where every weight is zero and the row is too.
    '''
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    n_elements = len(centers)

    # Distinct pairs (i < j) within the radius, so each coupling is found once and
    # mirrored below; the self-pairs query_pairs omits are the diagonal, at distance
    # zero and hence full weight r.
    pairs = KDTree(centers).query_pairs(r, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    off_diagonal = r - np.linalg.norm(centers[i] - centers[j], axis=1)

    diagonal = np.arange(n_elements)
    rows = np.concatenate([i, j, diagonal])
    cols = np.concatenate([j, i, diagonal])
    weights = np.concatenate([off_diagonal, off_diagonal, np.full(n_elements, float(r))])

    # The 1e-16 keeps a weightless row (only reachable at r = 0) at zero rather than
    # dividing by it.
    row_sums = np.bincount(rows, weights=weights, minlength=n_elements)
    return csr_array(
        (weights / (row_sums[rows] + 1e-16), (rows, cols)),
        shape=(n_elements, n_elements),
    )


# Gradient checking - TODO: make faster
def check_gradient(function, gradient, input_shape):
    import matplotlib.pyplot as plt  # local: keeps matplotlib off the core import path

    u = np.random.random(input_shape)
    computed_gradient = gradient(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_gradient = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = function(u + eps * direction)
            eval_m = function(u - eps * direction)
            numerical_gradient.append((eval_p - eval_m) / (2 * eps))
        numerical_gradient = np.array(numerical_gradient).reshape(computed_gradient.shape)
        errors_list.append(np.linalg.norm(numerical_gradient - computed_gradient))

    plt.title('Gradient check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()


def check_hessian(gradient, hessian, input_shape):
    import matplotlib.pyplot as plt  # local: keeps matplotlib off the core import path

    u = np.random.random(input_shape)
    computed_hessian = hessian(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_hessian = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = gradient(u + eps * direction)
            eval_m = gradient(u - eps * direction)
            numerical_hessian.append((eval_p - eval_m) / (2 * eps))
        numerical_hessian = np.array(numerical_hessian).reshape(computed_hessian.shape)
        errors_list.append(np.linalg.norm(numerical_hessian - computed_hessian))

    plt.title('Hessian check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()
