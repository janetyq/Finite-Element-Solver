"""Persistence for meshes and solutions.

Both live here rather than on the classes themselves so there is a single place
that knows the on-disk formats:

- Meshes: JSON. Small, portable, human-readable.
- Solutions: a single ``.npz`` archive holding the value arrays alongside the
  mesh geometry and ``n_components`` needed to rebuild the object.

Solutions deliberately avoid ``pickle``. Pickle executes arbitrary code on load
and is fragile across refactors, since it stores the class path -- moving or
renaming a class breaks every file previously written. The npz path reads plain
numeric arrays and reconstructs a known mesh class by name, so loading a file
can never run code from it. ``np.load`` is called with the default
``allow_pickle=False``, which enforces that.

Value arrays must be numeric and non-ragged (lists of equal-length arrays, such
as a per-timestep ``u`` series, are fine -- they stack). A ragged value fails at
save time rather than silently becoming a pickled object array.

Solutions are typed dataclasses (:mod:`fem.solution`), so the archive stores the
class name and reflects over the dataclass fields; ``load`` reconstructs the same
class. ``mesh`` and ``n_components`` are handled separately as the shared header.
"""
import dataclasses
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

_SOLUTION_HEADER = ('mesh', 'n_components')

# npz key namespacing: solution values are user-named, so they get a prefix to
# keep them from colliding with the mesh/component-count metadata in the same archive.
_VALUE_PREFIX = 'value.'
_MESH_CLASS = '__mesh_class__'
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


def save_solution(solution, path='solution.npz'):
    '''Write a typed solution (its fields + mesh + component count) to one npz archive.'''
    arrays = _mesh_to_arrays(solution.mesh)
    arrays[_N_COMPONENTS] = np.asarray(solution.n_components)
    arrays[_SOLUTION_CLASS] = np.array(type(solution).__name__)
    for f in dataclasses.fields(solution):
        if f.name in _SOLUTION_HEADER:
            continue
        value = np.asarray(getattr(solution, f.name))
        if value.dtype == object:
            # A ragged field (e.g. unequal-length time steps) can only be stored as
            # an object array, which means pickle -- refuse it rather than degrade.
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
    import fem.solution as solution_module

    with np.load(path) as data:
        # A Solution is defined over geometry alone -- it reads vertices and
        # elements and nothing else -- so whichever mesh class was stored is
        # enough. Element geometry and operators belong to a FunctionSpace,
        # which a solve builds for itself.
        mesh = _mesh_from_arrays(data)
        n_components = int(data[_N_COMPONENTS])
        cls = getattr(solution_module, str(data[_SOLUTION_CLASS]))
        fields = {
            f.name: data[_VALUE_PREFIX + f.name]
            for f in dataclasses.fields(cls)
            if f.name not in _SOLUTION_HEADER
        }
        return cls(mesh, n_components, **fields)
