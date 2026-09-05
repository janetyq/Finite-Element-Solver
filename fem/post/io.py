"""Persistence for meshes and solutions.

- Meshes: JSON. Small, portable, human-readable. A curved or tagged mesh carries its
  ``boundary_curves`` (each an analytic `Curve` written as a tagged dict) and its
  ``boundary_tags`` alongside the geometry, so it reloads rounded and tagged rather than
  as a straight, untagged mesh.
- Solutions: a single ``.npz`` archive holding the value arrays alongside the mesh
  geometry and ``n_components`` needed to rebuild the object.

Neither path unpickles. The npz path reads plain arrays (``allow_pickle=False``) and
reconstructs a known class by name, checked against its base class rather than resolved
to any attribute; boundary curves ride along as one JSON string. The archive stores the
solution's class name and reflects over its dataclass fields, with ``mesh`` and
``n_components`` as the header.
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
_MESH_TAGS = '__mesh_tags__'
_MESH_CURVES = '__mesh_curves__'
_N_COMPONENTS = '__n_components__'
_SOLUTION_CLASS = '__solution_class__'


# --- meshes -----------------------------------------------------------------

def save_mesh(mesh, path='test_mesh.json'):
    '''Write a mesh to JSON, boundary curves and tags included when it carries them.'''
    from fem.mesh.curves import curve_to_dict

    data = {
        'vertices': mesh.vertices.tolist(),
        'elements': mesh.elements.tolist(),
        # The facet order is written so reloaded curves and tags stay aligned with it.
        'boundary': mesh.boundary.tolist(),
    }
    if mesh.boundary_tags is not None:
        data['boundary_tags'] = mesh.boundary_tags.tolist()
    if mesh.boundary_curves is not None:
        data['boundary_curves'] = [None if c is None else curve_to_dict(c)
                                   for c in mesh.boundary_curves]
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.info('Saved mesh to %s', path)


def load_mesh(path='test_mesh.json'):
    '''Read a mesh from JSON, restoring boundary curves and tags when present.'''
    from fem.mesh.curves import curve_from_dict
    from fem.mesh.mesh import Mesh

    with open(path, 'r') as f:
        data = json.load(f)
    raw_curves = data.get('boundary_curves')
    boundary_curves = (None if raw_curves is None else
                       [None if c is None else curve_from_dict(c) for c in raw_curves])
    return Mesh(data['vertices'], data['elements'], data['boundary'],
                boundary_curves=boundary_curves, boundary_tags=data.get('boundary_tags'))


# --- solutions --------------------------------------------------------------

def _mesh_to_arrays(mesh):
    from fem.mesh.curves import curve_to_dict

    arrays = {
        _MESH_VERTICES: np.asarray(mesh.vertices),
        _MESH_ELEMENTS: np.asarray(mesh.elements),
        _MESH_BOUNDARY: np.asarray(mesh.boundary),
        _MESH_CLASS: np.array(type(mesh).__name__),
    }
    if mesh.boundary_tags is not None:
        arrays[_MESH_TAGS] = np.asarray(mesh.boundary_tags)
    if mesh.boundary_curves is not None:
        # A curve is an object, not an array; the whole list rides along as one JSON
        # string so the archive stays free of pickled objects.
        arrays[_MESH_CURVES] = np.array(json.dumps(
            [None if c is None else curve_to_dict(c) for c in mesh.boundary_curves]))
    return arrays


def _mesh_from_arrays(data):
    from fem.mesh.curves import curve_from_dict
    from fem.mesh.mesh import Mesh

    geometry = (data[_MESH_VERTICES], data[_MESH_ELEMENTS], data[_MESH_BOUNDARY])
    mesh_class = str(data[_MESH_CLASS])
    if mesh_class != 'Mesh':
        raise ValueError(f'Unknown mesh class in saved solution: {mesh_class}')
    tags = np.asarray(data[_MESH_TAGS]) if _MESH_TAGS in data.files else None
    curves = None
    if _MESH_CURVES in data.files:
        curves = [None if c is None else curve_from_dict(c)
                  for c in json.loads(str(data[_MESH_CURVES]))]
    return Mesh(*geometry, boundary_curves=curves, boundary_tags=tags)


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
    # resolved back through `fem.elements` on load.
    arrays[_ELEMENT_TYPE] = np.array(solution.element_type.__name__)
    for f in dataclasses.fields(solution):
        if not _persisted(f):
            continue
        arrays[_VALUE_PREFIX + f.name] = np.asarray(getattr(solution, f.name))
    # Pass a handle rather than the path so numpy doesn't append its own .npz.
    with open(path, 'wb') as f:
        np.savez(f, **arrays)
    logger.info('Saved solution to %s', path)


def _resolve(module, name, base, what):
    '''The class `name` in `module`, required to be a subclass of `base`.

    A saved archive names its solution and element classes as strings; resolving one to
    any module attribute would let a tampered file point at an arbitrary callable, so the
    result is checked to be a class deriving from the expected base before it is used.
    '''
    cls = getattr(module, name, None)
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ValueError(f'unknown {what} in saved solution: {name!r}')
    return cls


def load_solution(path='solution.npz'):
    '''Read a solution written by `save_solution`, reconstructing its dataclass.'''
    import fem.elements as elements_module
    import fem.post.solution as solution_module
    from fem.space import FunctionSpace

    with np.load(path) as data:
        mesh = _mesh_from_arrays(data)
        n_components = int(data[_N_COMPONENTS])
        # Resolve the stored names against their base class, not to any module attribute,
        # so a tampered archive cannot name an arbitrary callable to instantiate.
        cls = _resolve(solution_module, str(data[_SOLUTION_CLASS]),
                       solution_module.Solution, 'solution')
        element_type = _resolve(elements_module, str(data[_ELEMENT_TYPE]),
                                elements_module.Element, 'element type')
        # The space's node numbering is deterministic, so the rebuilt one matches the
        # space the solve used.
        space = FunctionSpace(mesh, element_type, n_components=n_components)
        fields = {
            f.name: data[_VALUE_PREFIX + f.name]
            for f in dataclasses.fields(cls)
            if _persisted(f)
        }
        return cls(space, **fields)
