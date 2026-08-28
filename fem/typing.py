"""Semantic type aliases for the arrays this package passes around.

Nearly every signature here takes or returns an `np.ndarray`, which says almost
nothing: the interesting distinctions are which quantity an array holds and what
shape it has. `ElementField` and `VertexField` are the same runtime type, and
confusing them is a real bug, so the names carry the meaning the dtype cannot.

These are aliases, not `NewType`s. A checker will not stop you passing a
`VertexField` where an `ElementField` is wanted; enforcing that would mean
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

# (n_vertices,) for a scalar PDE, or (n_vertices, n_components) for a vector one.
VertexField: TypeAlias = FloatArray

# (n_elements,) one value per element: stress, density, an error estimate.
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

# (free_idxs, fixed_idxs, fixed_values): the DOF partition a solve works in.
# Passed explicitly when the fixed values differ from the field's own, as in the
# Newton increment and the Newmark acceleration solve (both pinned to zero there).
Constraints: TypeAlias = tuple[DofIndices, DofIndices, FloatArray]

# A region: (n_vertices, spatial_dim) coordinates -> (n_vertices,) membership mask.
# Any callable of that shape qualifies (a `fem.regions.Region` object is one through
# its `__call__`); `fem.regions` names the recurring cases and adds `&`, `|`, `~`.
Region: TypeAlias = Callable[[Vertices], BoolArray]

if TYPE_CHECKING:
    from fem.regions import Field, TimeDependent

# A field value: a constant, a per-component constant, a function of position, a
# `TimeDependent` function of position and time (fixed at a time by
# `fem.regions.field_at` before use), or an already-normalized `Field`.
# `fem.regions.as_field` normalizes any of these to a `Field`, which `sample`s to
# (n_points, n_components). A component may itself be `None`, meaningful only for a
# Dirichlet value (BoundaryConditions' own resolver leaves it free rather than
# pinned); every other use rejects it (a load has no free component).
FieldValue: TypeAlias = Union[
    float,
    Sequence[Union[float, None]],
    FloatArray,
    Callable[[Point], Union[float, Sequence[Union[float, None]], FloatArray]],
    'TimeDependent',
    'Field',
    None,
]
