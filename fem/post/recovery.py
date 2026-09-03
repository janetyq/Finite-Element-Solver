"""Nodal recovery: continuous per-node fields from per-element or per-point ones.

A derived field (a flux, a stress) is computed where the solve knows it, at element
means, element nodes, or quadrature points. Smooth output, P2 plotting, and the
recovery error estimator want it once per node instead. Two recoveries do that:

- `'average'`: the volume-weighted average of the readings of the elements sharing a
  node. Local and cheap; weighted so that on a graded mesh a sliver does not count as
  much as the large element beside it.
- `'l2'`: the global L2 projection onto the nodal space, `M q = ∫ f φ`. A mass solve,
  more accurate on a graded mesh, and it conserves the field's integral.

Every function takes the `FunctionSpace` whose nodes the field is recovered onto; the
space contributes its numbering, volumes, geometry, and `nodal_mass_matrix`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from scipy.sparse.linalg import spsolve

from fem.typing import ElementValues, FloatArray, NodalValues

if TYPE_CHECKING:
    from fem.elements import ElementGeometry
    from fem.space import FunctionSpace

__all__ = ['RecoveryMethod', 'average_to_nodal', 'nodal_gradient', 'project_to_nodal',
           'recover_nodal']

# The two recoveries of the module docstring; every `method` parameter takes one.
RecoveryMethod: TypeAlias = Literal['average', 'l2']


def recover_nodal(space: FunctionSpace, values: ElementValues,
                  method: RecoveryMethod = 'average') -> NodalValues:
    '''Recover a continuous nodal field from a per-element one.

    Takes `(n_elements,)` or `(n_elements, *component_shape)` and returns `(n_nodes,)`
    or `(n_nodes, *component_shape)`, each component recovered independently.
    `method` is `'average'` or `'l2'` (see the module docstring).
    '''
    values = np.asarray(values, dtype=float)
    if len(values) != len(space.element_nodes):
        raise ValueError(
            f'expected one value per element ({len(space.element_nodes)}), '
            f'got {len(values)}'
        )
    if method == 'average':
        n_local = space.element_nodes.shape[1]
        per_node = np.repeat(values[:, None, ...], n_local, axis=1)
        return average_to_nodal(space, per_node)
    if method == 'l2':
        return _recover_nodal_l2(space, values)
    raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")


def average_to_nodal(space: FunctionSpace, values_at_nodes: FloatArray) -> NodalValues:
    '''Volume-weighted nodal average of a field sampled at each element's nodes.

    `values_at_nodes` is `(n_elements, N, *component_shape)`: element e's reading of
    the field at its N nodes, as `form.sample(space.geometry_at_nodes, ...)` produces.
    A node shared by several elements gets their readings averaged, weighted by element
    volume. This is `recover_nodal('average')` for a field that varies within the
    element; for an element-constant field the two agree.
    '''
    values_at_nodes = np.asarray(values_at_nodes, dtype=float)
    nodes = space.element_nodes
    if values_at_nodes.shape[:2] != nodes.shape:
        raise ValueError(
            f'expected one value per element node {nodes.shape}, '
            f'got {values_at_nodes.shape[:2]}'
        )
    weights = space.element_volumes
    n_local = nodes.shape[1]
    flat = nodes.ravel()
    trailing = values_at_nodes.shape[2:]

    # Scatter each element's volume-weighted readings onto its own nodes; `add.at`
    # accumulates, so a shared node sums its elements' contributions.
    w = weights.reshape((-1, 1) + (1,) * len(trailing))
    weighted = (values_at_nodes * w).reshape(len(flat), *trailing)
    sums = np.zeros((space.n_nodes, *trailing))
    np.add.at(sums, flat, weighted)
    norms = np.bincount(flat, weights=np.repeat(weights, n_local), minlength=space.n_nodes)
    # Every referenced node belongs to at least one element; an unreferenced one would
    # divide by zero, so it keeps 0 instead.
    norms = np.where(norms > 0, norms, 1.0).reshape((-1,) + (1,) * len(trailing))
    return sums / norms


def _recover_nodal_l2(space: FunctionSpace, values: FloatArray) -> NodalValues:
    '''The L2 projection of a per-element field onto the nodal space: solve M q = b.

    `b_i = ∫ f φ_i`, and with `f` element-constant that is `Σ_e f_e ∫_e φ_i`, built from
    the same rule the mass matrix integrates with so `M⁻¹ b` is the exact projection.
    Each trailing component is one right-hand side against the shared scalar mass matrix.
    '''
    geometry = space.geometry
    # The integral of each shape function over each element: (n_elements, N).
    shape_integral = np.einsum('eq,qn->en', geometry.weight_detJ, geometry.shape)
    nodes = space.element_nodes
    n_local = nodes.shape[1]
    trailing = values.shape[1:]

    contrib = (shape_integral.reshape(len(values), n_local, *((1,) * len(trailing)))
               * values[:, None, ...])
    load = np.zeros((space.n_nodes, *trailing))
    np.add.at(load, nodes.ravel(), contrib.reshape(len(values) * n_local, *trailing))

    projected = spsolve(space.nodal_mass_matrix, load.reshape(space.n_nodes, -1))
    return np.asarray(projected).reshape(space.n_nodes, *trailing)


def project_to_nodal(space: FunctionSpace, values_qp: FloatArray,
                     geometry: ElementGeometry) -> NodalValues:
    '''L2-project a per-quadrature-point field onto the continuous nodal space.

    `values_qp` is `(n_elements, n_qp, *component_shape)`, a field sampled at
    `geometry`'s quadrature points. Solves `M q = b` with `b_i = Σ_e Σ_q w_eq φ_i(x_eq)
    f_eq`, the L2 projection of that field, recovering each trailing component against
    the shared scalar mass matrix.

    This generalizes `recover_nodal('l2')` from an element-constant field to one that
    varies within the element, as a P2 derived field does. `geometry` must be the
    space's own geometry so its shape functions and node numbering line up with the
    nodal space `M` is built on.
    '''
    values_qp = np.asarray(values_qp, dtype=float)
    trailing = values_qp.shape[2:]
    nodes = space.element_nodes
    n_local = nodes.shape[1]
    # b[e, n, ...] = Σ_q weight_detJ[e,q] shape[q,n] f[e,q,...]
    contrib = np.einsum('eq,qn,eq...->en...', geometry.weight_detJ, geometry.shape, values_qp)
    load = np.zeros((space.n_nodes, *trailing))
    np.add.at(load, nodes.ravel(), contrib.reshape(len(nodes) * n_local, *trailing))
    projected = spsolve(space.nodal_mass_matrix, load.reshape(space.n_nodes, -1))
    return np.asarray(projected).reshape(space.n_nodes, *trailing)


def nodal_gradient(space: FunctionSpace, u: NodalValues,
                   method: RecoveryMethod = 'average') -> NodalValues:
    '''(n_nodes, spatial_dim) continuous gradient of a nodal field.

    `'average'` evaluates each element's gradient at its own nodes and volume-averages
    the elements sharing a node; `'l2'` projects the gradient sampled at quadrature
    points onto the nodal space. Both read a P2 gradient's variation within the
    element, so a boundary node gets the boundary value rather than an interior one.
    For P1 both agree with `recover_nodal(space, space.gradient(u), method)`.
    '''
    u_elements = np.asarray(u)[space.element_nodes]
    if method == 'average':
        return average_to_nodal(space, space.geometry_at_nodes.gradients(u_elements))
    if method == 'l2':
        geometry = space.geometry_at(2 * space.element_type.SHAPE_DEGREE)
        return project_to_nodal(space, geometry.gradients(u_elements), geometry)
    raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")
