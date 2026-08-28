"""Persistence for meshes and solutions.

- Meshes: JSON. Small, portable, human-readable.
- Solutions: a single ``.npz`` archive holding the value arrays alongside the mesh
  geometry and ``n_components`` needed to rebuild the object.

Solutions avoid ``pickle``, which executes arbitrary code on load and breaks when a
class moves. The npz path reads plain numeric arrays (``allow_pickle=False``) and
reconstructs a known class by name. Value arrays must be numeric and non-ragged; a
ragged value fails at save time. The archive stores the solution's class name and
reflects over its dataclass fields, with ``mesh`` and ``n_components`` as the header.
"""
import dataclasses
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Header fields are handled by name rather than reflected into value arrays: `mesh` and
# `n_components` reconstruct the geometry, and `element_type` is a class stored by name
# (an array field cannot hold one), resolved back through `fem.elements` on load.
_SOLUTION_HEADER = ('space',)

# npz key namespacing: solution values are user-named, so they get a prefix to
# keep them from colliding with the mesh/component-count metadata in the same archive.
_VALUE_PREFIX = 'value.'
_MESH_CLASS = '__mesh_class__'
_ELEMENT_TYPE = '__element_type__'
_MESH_VERTICES = '__mesh_vertices__'
_MESH_ELEMENTS = '__mesh_elements__'
_MESH_BOUNDARY = '__mesh_boundary__'
# The on-disk key keeps its original spelling: renaming the Python name is free,
# renaming the archive key would strand every .npz already written.
_N_COMPONENTS = '__dim__'
_SOLUTION_CLASS = '__solution_class__'


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


def load_mesh(path='test_mesh.json'):
    '''Read a mesh from JSON.'''
    from fem.mesh.mesh import Mesh

    with open(path, 'r') as f:
        data = json.load(f)
    return Mesh(data['vertices'], data['elements'], data['boundary'])


# --- solutions --------------------------------------------------------------

def _mesh_to_arrays(mesh):
    arrays = {
        _MESH_VERTICES: np.asarray(mesh.vertices),
        _MESH_ELEMENTS: np.asarray(mesh.elements),
        _MESH_BOUNDARY: np.asarray(mesh.boundary),
        _MESH_CLASS: np.array(type(mesh).__name__),
    }
    return arrays


def _mesh_from_arrays(data):
    from fem.mesh.mesh import Mesh

    geometry = (data[_MESH_VERTICES], data[_MESH_ELEMENTS], data[_MESH_BOUNDARY])
    mesh_class = str(data[_MESH_CLASS])
    # 'FEMesh' still appears in archives written before the element data moved to
    # FunctionSpace. Geometry is all that was ever stored, so those load as a Mesh
    # and a solve rebuilds the rest.
    if mesh_class not in ('Mesh', 'FEMesh'):
        raise ValueError(f'Unknown mesh class in saved solution: {mesh_class}')
    return Mesh(*geometry)


def _persisted(f: dataclasses.Field) -> bool:
    '''Whether a solution field is stored: the space is written as its mesh and
    parameters, and a field marked `metadata={'persist': False}` (a form, not an
    array) is dropped.'''
    return f.name not in _SOLUTION_HEADER and f.metadata.get('persist', True)


def save_solution(solution, path='solution.npz'):
    '''Write a typed solution (its fields + mesh + component count) to one npz archive.'''
    arrays = _mesh_to_arrays(solution.mesh)
    arrays[_N_COMPONENTS] = np.asarray(solution.n_components)
    arrays[_SOLUTION_CLASS] = np.array(type(solution).__name__)
    # A class cannot live in an array, so the element type is stored by name and
    # resolved back through `fem.elements` on load; '' means the linear default.
    arrays[_ELEMENT_TYPE] = np.array(
        solution.element_type.__name__ if solution.element_type is not None else '')
    for f in dataclasses.fields(solution):
        if not _persisted(f):
            continue
        value = np.asarray(getattr(solution, f.name))
        if value.dtype == object:
            # A ragged field (e.g. unequal-length time steps) can only be stored as
            # an object array, which means pickle; refuse it rather than degrade.
            raise ValueError(
                f'solution field {f.name!r} is ragged and cannot be saved without pickle'
            )
        arrays[_VALUE_PREFIX + f.name] = value
    # Pass a handle rather than the path so numpy doesn't append its own .npz.
    with open(path, 'wb') as f:
        np.savez(f, **arrays)
    logger.info('Saved solution to %s', path)


def load_solution(path='solution.npz'):
    '''Read a solution written by `save_solution`, reconstructing its dataclass.'''
    import fem.elements as elements_module
    import fem.post.solution as solution_module
    from fem.space import FunctionSpace

    with np.load(path) as data:
        mesh = _mesh_from_arrays(data)
        n_components = int(data[_N_COMPONENTS])
        cls = getattr(solution_module, str(data[_SOLUTION_CLASS]))
        # Resolve the element type stored by name (older archives predate it, so a
        # missing key means the linear default).
        type_name = str(data[_ELEMENT_TYPE]) if _ELEMENT_TYPE in data else ''
        element_type = getattr(elements_module, type_name) if type_name else None
        # The space's node numbering is deterministic, so the rebuilt one matches the
        # space the solve used.
        space = FunctionSpace(mesh, element_type, n_components=n_components)
        fields = {
            f.name: data[_VALUE_PREFIX + f.name]
            for f in dataclasses.fields(cls)
            if _persisted(f)
        }
        return cls(space, **fields)
