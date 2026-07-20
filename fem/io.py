"""Persistence for meshes and solutions.

Both live here rather than on the classes themselves so there is a single place
that knows the on-disk formats:

- Meshes: JSON. Small, portable, human-readable.
- Solutions: a single ``.npz`` archive holding the value arrays alongside the
  mesh geometry and ``dim`` needed to rebuild the object.

Solutions deliberately avoid ``pickle``. Pickle executes arbitrary code on load
and is fragile across refactors, since it stores the class path -- moving or
renaming a class breaks every file previously written. The npz path reads plain
numeric arrays and reconstructs a known mesh class by name, so loading a file
can never run code from it. ``np.load`` is called with the default
``allow_pickle=False``, which enforces that.

Value arrays must be numeric and non-ragged (lists of equal-length arrays, such
as the per-timestep ``u_values``, are fine -- they stack). A ragged value fails
at save time rather than silently becoming a pickled object array.
"""
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# npz key namespacing: solution values are user-named, so they get a prefix to
# keep them from colliding with the mesh/dim metadata stored in the same archive.
_VALUE_PREFIX = 'value.'
_MESH_CLASS = '__mesh_class__'
_MESH_ELEMENT_TYPE = '__mesh_element_type__'
_MESH_VERTICES = '__mesh_vertices__'
_MESH_ELEMENTS = '__mesh_elements__'
_MESH_BOUNDARY = '__mesh_boundary__'
_DIM = '__dim__'


# --- meshes -----------------------------------------------------------------

def save_mesh(mesh, path='test_mesh.json'):
    '''Write a mesh to JSON.'''
    with open(path, 'w') as f:
        json.dump({
            'vertices': mesh.vertices.tolist(),
            'elements': mesh.elements.tolist(),
            'boundary': mesh.boundary.tolist(),
        }, f)
    logger.info('Saved mesh to %s', path)


def load_mesh(path='test_mesh.json', cls=None):
    '''Read a mesh from JSON. `cls` selects the class to rebuild (default Mesh),
    which is how `FEMesh.load` returns an FEMesh rather than a bare Mesh.'''
    from fem.mesh.mesh import Mesh

    if cls is None:
        cls = Mesh
    with open(path, 'r') as f:
        data = json.load(f)
    return cls(data['vertices'], data['elements'], data['boundary'])


# --- solutions --------------------------------------------------------------

def _mesh_to_arrays(mesh):
    arrays = {
        _MESH_VERTICES: np.asarray(mesh.vertices),
        _MESH_ELEMENTS: np.asarray(mesh.elements),
        _MESH_BOUNDARY: np.asarray(mesh.boundary),
        _MESH_CLASS: np.array(type(mesh).__name__),
    }
    element_type = getattr(mesh, 'element_type', None)  # FEMesh only
    if element_type is not None:
        arrays[_MESH_ELEMENT_TYPE] = np.array(element_type.__name__)
    return arrays


def _mesh_from_arrays(data):
    import fem.elements
    from fem.mesh.femesh import FEMesh
    from fem.mesh.mesh import Mesh

    geometry = (data[_MESH_VERTICES], data[_MESH_ELEMENTS], data[_MESH_BOUNDARY])
    mesh_class = str(data[_MESH_CLASS])

    if mesh_class == 'Mesh':
        return Mesh(*geometry)
    if mesh_class == 'FEMesh':
        element_type_name = str(data[_MESH_ELEMENT_TYPE])
        element_type = getattr(fem.elements, element_type_name, None)
        if element_type is None:
            raise ValueError(f'Unknown element type in saved solution: {element_type_name}')
        return FEMesh(*geometry, element_type=element_type)
    raise ValueError(f'Unknown mesh class in saved solution: {mesh_class}')


def save_solution(solution, path='solution.npz'):
    '''Write a solution (values + mesh + dim) to a single npz archive.'''
    arrays = _mesh_to_arrays(solution.femesh)
    arrays[_DIM] = np.asarray(solution.dim)
    for name, value in solution.values.items():
        arrays[_VALUE_PREFIX + name] = np.asarray(value)
    # Pass a handle rather than the path so numpy doesn't append its own .npz.
    with open(path, 'wb') as f:
        np.savez(f, **arrays)
    logger.info('Saved solution to %s', path)


def load_solution(path='solution.npz'):
    '''Read a solution written by `save_solution`.'''
    from fem.mesh.femesh import FEMesh
    from fem.solution import Solution

    with np.load(path) as data:
        # A Solution always comes from a solve, so its mesh is an FEMesh and that
        # is what round-trips. A bare Mesh here means the file was hand-built,
        # and the missing element_objs/M/K would only surface much later.
        mesh = _mesh_from_arrays(data)
        if not isinstance(mesh, FEMesh):
            raise ValueError(
                f'solution at {path} stores a {type(mesh).__name__}; a Solution is '
                f'defined over an FEMesh'
            )
        solution = Solution(mesh, int(data[_DIM]))
        for key in data.files:
            if key.startswith(_VALUE_PREFIX):
                solution.values[key[len(_VALUE_PREFIX):]] = data[key]
    return solution
