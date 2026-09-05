"""Semantic type aliases for the arrays this package passes around.

Nearly every signature here takes or returns an `np.ndarray`, which says almost
nothing: the interesting distinctions are which quantity an array holds and what
shape it has. `ElementValues` and `NodalValues` are the same runtime type, and
confusing them is a real bug, so the names carry the meaning the dtype cannot.

These are aliases, not `NewType`s. A checker will not stop you passing a
`NodalValues` where an `ElementValues` is wanted; enforcing that would mean
wrapping every array at construction, which is not worth it in numerical code.
They document intent for the reader and for autocomplete, and the shape comments
are the contract.
"""
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

FloatArray: TypeAlias = npt.NDArray[np.float64]
# Any integer width: scipy hands back `intc` from Delaunay while numpy's own
# constructors give `int_`, and those are different types on Windows.
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# (spatial_dim,) coordinates of a single point.
Point: TypeAlias = FloatArray

# (n_vertices, spatial_dim) node coordinates.
Vertices: TypeAlias = FloatArray

# (n_elements, n_nodes, spatial_dim) node coordinates gathered per element: the
# input to batched element geometry. `vertices[elements]` builds one.
ElementVertices: TypeAlias = FloatArray

# (n_elements, n_nodes) vertex indices per element; (n_boundary, n_nodes) for a
# boundary facet array.
Elements: TypeAlias = IntArray

# (n_nodes,) for a scalar PDE, or (n_nodes, n_components) for a vector one: one row per
# node of the space (vertices first, then any edge nodes).
NodalValues: TypeAlias = FloatArray

# (n_elements,) one value per element: stress, density, an error estimate.
ElementValues: TypeAlias = FloatArray

# (n_vertices * n_components,) the flat unknown a solve works in, ordered so that
# node v component d lives at index n_components*v + d. `fem.space.dof_indices` builds
# these from element node indices.
DofVector: TypeAlias = FloatArray

# (k,) indices into a DofVector.
DofIndices: TypeAlias = IntArray

# (n_vertices,) indices into the vertex array. Distinct from DofIndices: equal
# only when n_components == 1.
VertexIndices: TypeAlias = IntArray

# (n, n) dense matrix: a per-element block (grad_phi grad_phi^T, B^T D B), or a
# small dense system in a test.
Matrix: TypeAlias = FloatArray

# An assembled global operator: mass, stiffness, or tangent. Sparse (CSR): FEM
# matrices have a handful of nonzeros per row, so the dense (n_dofs, n_dofs) form
# is O(N^2) memory and never assembled.
SparseMatrix: TypeAlias = csr_array

# What a DiscreteSystem factors: a sparse assembled operator in production, or a
# small dense one in a test (csc_array converts either). Kept loose:
# scipy.sparse ships no stubs, so sparse subscripting fights a precise type more
# than it documents.
Operator: TypeAlias = Any

# A region: (n_vertices, spatial_dim) coordinates -> (n_vertices,) membership mask.
# Any callable of that shape qualifies; `fem.regions` names the recurring cases.
Region: TypeAlias = Callable[[Vertices], BoolArray]

if TYPE_CHECKING:
    from fem.regions import TimeDependent

# A field value: a constant, a per-component constant, a function of position, or a
# `TimeDependent` function of position and time (fixed at a time by
# `fem.regions.field_at` before use). `fem.regions.evaluate_field` normalizes the
# first three to (n_points, n_components). A function of position is given every
# point at once, an (N, d) array like a `Region` is, and returns its components over
# them: `lambda p: np.sin(p[:, 0])`, `lambda p: [p[:, 1], 0.0]`. A component may itself be `None`: a
# Dirichlet value leaves it free rather than pinned, a Neumann value leaves it
# undriven (zero in the integral); `evaluate_field` rejects it for every other use.
FieldValue: TypeAlias = Union[
    float,
    Sequence[float | None],
    FloatArray,
    Callable[[Vertices], float | Sequence[float | None | FloatArray] | FloatArray],
    'TimeDependent',
    None,
]
