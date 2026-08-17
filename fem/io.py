"""Persistence for meshes and solutions.

Both live here rather than on the classes themselves so there is a single place
that knows the on-disk formats:

- Meshes: JSON by default, or any format meshio handles when the path carries a
  non-``.json`` suffix. The native JSON is small, portable, and human-readable;
  the meshio path reads Gmsh ``.msh`` and writes VTK ``.vtu`` (openable in
  ParaView), among ~40 formats, through one Mesh-to-meshio adapter. Physical-group
  tags (named boundaries and subdomains) carry both ways: an import fills a mesh's
  ``cell_tags`` / ``facet_tags`` / ``tag_names``, and an export writes them back.
- Solutions: a single ``.npz`` archive holding the value arrays alongside the
  mesh geometry and ``n_components`` needed to rebuild the object.

Standard formats carry only straight-sided geometry, so a mesh's analytic
``boundary_curves`` do not survive a meshio round-trip; use the native JSON when
that association must be kept. Standard formats also cannot express the P2
midside nodes a FunctionSpace adds, and a Mesh is P1 by construction, so a
higher-order file imports at its corner nodes.

Solutions deliberately avoid ``pickle``. Pickle executes arbitrary code on load
and is fragile across refactors, since it stores the class path: moving or
renaming a class breaks every file previously written. The npz path reads plain
numeric arrays and reconstructs a known mesh class by name, so loading a file
can never run code from it. ``np.load`` is called with the default
``allow_pickle=False``, which enforces that.

Value arrays must be numeric and non-ragged (lists of equal-length arrays, such
as a per-timestep ``u`` series, are fine; they stack). A ragged value fails at
save time rather than silently becoming a pickled object array.

Solutions are typed dataclasses (:mod:`fem.solution`), so the archive stores the
class name and reflects over the dataclass fields; ``load`` reconstructs the same
class. ``mesh`` and ``n_components`` are handled separately as the shared header.
"""
import dataclasses
import json
import logging
import pathlib

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

# meshio maps some extensions to a surprising default format; `.msh` is shared by
# Gmsh and Ansys and defaults to Ansys, which cannot even write boundary lines. Pin
# the FEM-standard meaning so a `.msh` is always Gmsh. Writing uses the MSH2 form
# (`gmsh22`), whose physical groups need no per-node entity bookkeeping the MSH4
# writer demands; reading lets the Gmsh reader detect either version.
_FORCED_WRITE_FORMAT = {'.msh': 'gmsh22'}
_FORCED_READ_FORMAT = {'.msh': 'gmsh'}


def save_mesh(mesh, path='test_mesh.json'):
    '''Write a mesh. A `.json` suffix uses the native format; any other suffix
    (e.g. `.vtu`, `.msh`, `.obj`) is written through meshio.'''
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == '.json':
        with open(path, 'w') as f:
            json.dump({
                'vertices': mesh.vertices.tolist(),
                'elements': mesh.elements.tolist(),
                'boundary': mesh.boundary.tolist(),
            }, f)
    else:
        _mesh_to_meshio(mesh).write(path, file_format=_FORCED_WRITE_FORMAT.get(suffix))
    logger.info('Saved mesh to %s', path)


def load_mesh(path='test_mesh.json'):
    '''Read a mesh. A `.json` suffix uses the native format; any other suffix
    (e.g. `.msh`, `.vtu`, `.obj`) is read through meshio.'''
    from fem.mesh.mesh import Mesh

    suffix = pathlib.Path(path).suffix.lower()
    if suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        return Mesh(data['vertices'], data['elements'], data['boundary'])

    import meshio
    return _mesh_from_meshio(meshio.read(path, file_format=_FORCED_READ_FORMAT.get(suffix)))


# --- meshio adapter ---------------------------------------------------------

# Node count of a linear simplex -> meshio cell type. These are the shapes a
# Mesh holds: a line (2), a triangle (3), a tet (4).
_MESH_CELL_TYPE = {2: 'line', 3: 'triangle', 4: 'tetra'}

# meshio cell type -> (corner-node count, element dimension). The higher-order
# variants list their corner nodes first, so slicing to the corner count drops
# the midside nodes a P1 Mesh does not carry.
_SIMPLEX_FROM_CELL = {
    'line': (2, 1), 'line3': (2, 1),
    'triangle': (3, 2), 'triangle6': (3, 2),
    'tetra': (4, 3), 'tetra10': (4, 3),
}

# meshio's key for the integer physical-group id of each cell. The Gmsh reader and
# writer both use it; we read it into a Mesh's tags and write a Mesh's tags back to it.
_PHYSICAL = 'gmsh:physical'


def _mesh_to_meshio(mesh):
    '''A meshio.Mesh over the volume/area cells, ready for `.write`.

    Physical-group tags, if the mesh carries them, ride along as `gmsh:physical` cell
    data (with the facet tags on an added boundary block) plus `field_data` for the
    names, so a Gmsh `.msh` round-trips the groups and a `.vtu` shows them as a cell
    field. An untagged mesh writes only its volume cells, exactly as before.
    '''
    import meshio

    points = np.asarray(mesh.vertices, dtype=float)
    # VTK-family formats store 3D points; a padded z=0 column reduces back
    # cleanly on read, so pad rather than emit a format-specific 2D file.
    if points.shape[1] < 3:
        points = np.hstack([points, np.zeros((len(points), 3 - points.shape[1]))])

    cells: list = [(_MESH_CELL_TYPE[mesh.elements.shape[1]], np.asarray(mesh.elements))]
    cell_data: dict = {}
    field_data: dict = {}
    if mesh.cell_tags is not None or mesh.facet_tags is not None:
        dim = mesh.elements.shape[1] - 1
        physical = [
            np.asarray(mesh.cell_tags) if mesh.cell_tags is not None
            else np.zeros(len(mesh.elements), dtype=int)
        ]
        if mesh.facet_tags is not None:
            cells.append((_MESH_CELL_TYPE[mesh.boundary.shape[1]], np.asarray(mesh.boundary)))
            physical.append(np.asarray(mesh.facet_tags))
        # meshio's Gmsh writer expects a geometrical id beside the physical one; reuse
        # the physical id, which is all that is needed to round-trip the groups.
        cell_data = {_PHYSICAL: physical, 'gmsh:geometrical': [p.copy() for p in physical]}
        field_data = _tag_field_data(mesh, dim)

    return meshio.Mesh(points, cells, cell_data=cell_data, field_data=field_data)


def _tag_field_data(mesh, cell_dim):
    '''meshio `field_data` naming each tag id: {name: [id, dimension]}.

    A cell tag has the element dimension; a facet tag one less. Gmsh keys names by both,
    so the dimension is recorded alongside the id.
    '''
    cell_ids = set(np.unique(mesh.cell_tags).tolist()) if mesh.cell_tags is not None else set()
    facet_ids = set(np.unique(mesh.facet_tags).tolist()) if mesh.facet_tags is not None else set()
    field_data = {}
    for tag_id, name in mesh.tag_names.items():
        if tag_id in cell_ids:
            field_data[name] = np.array([tag_id, cell_dim])
        elif tag_id in facet_ids:
            field_data[name] = np.array([tag_id, cell_dim - 1])
    return field_data


def _mesh_from_meshio(m):
    '''Build a Mesh from a meshio.Mesh, keeping the highest-dimensional simplex
    cells as the elements and re-deriving the boundary from them.

    Physical-group tags are carried through: the top block's ids become `cell_tags`,
    the boundary blocks' ids are matched onto the derived boundary by vertex set to
    become `facet_tags`, and `field_data` supplies the names.
    '''
    from fem.geometry import get_boundary_from_vertices_elements
    from fem.mesh.mesh import Mesh

    # A .msh file often carries boundary lines/triangles alongside the volume
    # cells; the top-dimensional block is the mesh, the rest are its facets.
    best = None  # (dim, block_index, corner_elements)
    for idx, block in enumerate(m.cells):
        info = _SIMPLEX_FROM_CELL.get(block.type)
        if info is None:
            continue
        corners, dim = info
        if best is None or dim > best[0]:
            best = (dim, idx, np.asarray(block.data)[:, :corners])
    if best is None:
        raise NotImplementedError(
            'no line, triangle, or tetra cells found; only linear-simplex meshes are supported'
        )
    dim, top_idx, elements = best

    points = np.asarray(m.points, dtype=float)
    # The solver assumes spatial_dim equals the element dimension, so drop the
    # ambient coordinates a 3D-padded file adds. A genuine surface mesh (nonzero
    # trailing coordinates) is out of scope, so reject it with a clear message
    # rather than silently flattening it.
    if points.shape[1] > dim:
        if not np.allclose(points[:, dim:], 0.0):
            raise NotImplementedError(
                f'{dim}D elements embedded in {points.shape[1]}D space are not supported; '
                'the solver requires spatial_dim to equal the element dimension'
            )
        points = points[:, :dim]

    # Keep only referenced vertices and renumber: isolated points and dropped
    # midside nodes would otherwise be empty rows that make assembly singular.
    used = np.unique(elements)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    points = points[used]
    elements = remap[elements]

    boundary = get_boundary_from_vertices_elements(elements.tolist())
    cell_tags, facet_tags, tag_names = _read_meshio_tags(m, top_idx, dim, remap, boundary)
    return Mesh(points, elements, boundary,
                cell_tags=cell_tags, facet_tags=facet_tags, tag_names=tag_names)


def _read_meshio_tags(m, top_idx, dim, remap, boundary):
    '''(cell_tags, facet_tags, tag_names) from a meshio.Mesh's physical groups.

    `cell_tags` come straight from the top block's per-cell ids. `facet_tags` are keyed
    by sorted vertex set (remapped to the pruned numbering) and looked up per derived
    boundary facet, since the boundary is re-derived from the volume cells rather than
    taken from meshio's boundary blocks, which a Gmsh file only partly populates. An
    untagged facet gets 0, the id Gmsh reserves for "no group".
    '''
    physical = m.cell_data.get(_PHYSICAL)
    tag_names = {int(spec[0]): str(name) for name, spec in m.field_data.items()}
    if physical is None:
        return None, None, {}

    # Gmsh reserves id 0 for "ungrouped", and its writer fills an untagged mesh with
    # zeros; an all-zero array therefore means no groups, which reads back as None.
    cell_tags = np.asarray(physical[top_idx], dtype=int)
    cell_tags = cell_tags if cell_tags.any() else None

    facet_tag_by_vertices: dict[tuple[int, ...], int] = {}
    for idx, block in enumerate(m.cells):
        info = _SIMPLEX_FROM_CELL.get(block.type)
        if info is None or info[1] != dim - 1:
            continue
        for facet, tag in zip(np.asarray(block.data)[:, :info[0]], np.asarray(physical[idx])):
            mapped = remap[facet]
            if np.any(mapped < 0):
                continue  # a facet on a pruned vertex is not a facet of the volume mesh
            facet_tag_by_vertices[tuple(sorted(int(v) for v in mapped))] = int(tag)

    facet_tags = np.array(
        [facet_tag_by_vertices.get(tuple(sorted(int(v) for v in facet)), 0) for facet in boundary],
        dtype=int,
    )
    facet_tags = facet_tags if facet_tags.any() else None

    if cell_tags is None and facet_tags is None:
        return None, None, {}
    return cell_tags, facet_tags, tag_names


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
    import fem.solution as solution_module

    with np.load(path) as data:
        # A Solution is defined over geometry alone (it reads vertices and
        # elements and nothing else), so whichever mesh class was stored is
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
