"""Semantic type aliases for the arrays this package passes around.

Nearly every signature here takes or returns an `np.ndarray`, which says almost
nothing: the interesting distinctions are *which quantity* an array holds and
*what shape* it has. `ElementField` and `VertexField` are the same runtime type
and confusing them is a real bug -- `Solution.get_values` exists largely to
convert between them -- so the names carry the meaning the dtype cannot.

These are aliases, not `NewType`s. A checker will not stop you passing a
`VertexField` where an `ElementField` is wanted; enforcing that would mean
wrapping every array at construction, which is not worth it in numerical code.
They document intent for the reader and for autocomplete, and the shape comments
are the contract.
"""
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, Union

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
# Any integer width: scipy hands back `intc` from Delaunay while numpy's own
# constructors give `int_`, and those are different types on Windows.
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# (spatial_dim,) coordinates of a single point.
Point: TypeAlias = FloatArray

# (n_vertices, spatial_dim) node coordinates.
Vertices: TypeAlias = FloatArray

# (n_elements, n_nodes) vertex indices per element; (n_boundary, n_nodes) for a
# boundary facet array.
Elements: TypeAlias = IntArray

# (n_vertices,) for a scalar PDE, or (n_vertices, n_components) for a vector one.
VertexField: TypeAlias = FloatArray

# (n_elements,) one value per element -- stress, density, an error estimate.
ElementField: TypeAlias = FloatArray

# (n_vertices * n_components,) the flat unknown a solve works in, ordered so that
# node v component d lives at index n_components*v + d. `fem.space.dof_indices` builds
# these from element node indices.
DofVector: TypeAlias = FloatArray

# (k,) indices into a DofVector.
DofIndices: TypeAlias = IntArray

# (n_vertices,) indices into the vertex array. Distinct from DofIndices: equal
# only when n_components == 1.
VertexIndices: TypeAlias = IntArray

# (n, n) dense system matrix -- mass, stiffness, or a Crank-Nicolson block.
Matrix: TypeAlias = FloatArray

# (free_idxs, fixed_idxs, fixed_values) -- the DOF partition a solve works in.
# Passed explicitly where the unknown is not one value per node, as in the wave
# solver's stacked [u; du/dt] block.
Constraints: TypeAlias = tuple[DofIndices, DofIndices, FloatArray]

# A region: (n_vertices, spatial_dim) coordinates -> (n_vertices,) membership mask.
# Any callable of that shape qualifies; `fem.regions` names the recurring cases.
Region: TypeAlias = Callable[[Vertices], BoolArray]

# A field value: a constant, a per-component constant, or a function of position.
# `fem.regions.evaluate_field` normalizes all three to (n_points, n_components).
FieldValue: TypeAlias = Union[
    float,
    Sequence[float],
    FloatArray,
    Callable[[Point], Union[float, Sequence[float], FloatArray]],
    None,
]
