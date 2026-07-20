"""Plotly trace builders used by the Plotter class: mesh, boundary, highlights,
colored fields, surfaces, arrows, and boundary conditions.

Every mesh here is treated as 3D (`z=0` for genuinely-2D meshes) so a single
`go.Mesh3d`/`go.Scatter3d`-based code path covers both -- a flat mesh is just a
3D mesh viewed from above. That's also what lets a 3D tet mesh's boundary
surface (extracted via `_surface_faces`) render through the exact same builders
as a 2D triangle mesh.
"""
import numpy as np
import plotly.graph_objects as go


def _to_3d(points):
    """Pad a (N, 2) point array to (N, 3) with z=0; pass (N, 3) through unchanged."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] >= 3:
        return points[:, :3]
    return np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])


def _surface_faces(mesh):
    """Triangle faces to render `mesh`'s surface: itself if already triangles
    (2D), or its boundary faces if tetrahedra (a 3D volume mesh has no single
    surface of its own -- a Mesh3d needs the triangulated boundary).
    """
    if mesh.elements.shape[1] == 3:
        return mesh.elements
    if mesh.elements.shape[1] == 4:
        # Deferred: only tet meshes need this, and it pulls in the optional
        # mesh3d extra (pyvista/tetgen) that a 2D-only install won't have.
        from fem.mesh.tetmesh import find_boundary_faces
        return find_boundary_faces(mesh.elements)
    raise ValueError(f'Unsupported element shape for plotting: {mesh.elements.shape}')


def _edges_from_faces(faces):
    """Every edge of every face, as (i, j) pairs. Duplicate edges (shared
    between adjacent triangles) are harmless for a wireframe."""
    edges = []
    for face in faces:
        n = len(face)
        edges.extend((face[k], face[(k + 1) % n]) for k in range(n))
    return edges


def _line_trace(points3d, edges, **kwargs):
    """One Scatter3d line trace covering every edge, with a gap (`None`)
    between segments so they don't connect to each other."""
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [points3d[i, 0], points3d[j, 0], None]
        ys += [points3d[i, 1], points3d[j, 1], None]
        zs += [points3d[i, 2], points3d[j, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', **kwargs)


def _auto_scale(vertices, vectors, target_fraction=0.08):
    """Scale factor so the longest vector spans `target_fraction` of the mesh's
    bounding-box diagonal. Vector magnitudes here range from O(0.1) gradients to
    O(50) tractions, so a fixed scale can't work for both -- this is what
    actually fixes the "hard to see scale" complaint the matplotlib version had,
    rather than picking a different fixed guess.
    """
    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    max_mag = np.max(np.linalg.norm(vectors, axis=1))
    if max_mag < 1e-12 or extent < 1e-12:
        return 1.0
    return target_fraction * extent / max_mag


def _vector_field_trace(origins, vectors, scale, color='black'):
    """Tail->head line segments plus a small two-stroke arrowhead, for an
    explicit set of 2D origin points and vectors. Hand-rolled rather than
    `plotly.figure_factory.create_quiver`, which returns a standalone 2D-only
    `Figure` -- not a trace this can compose into a shared `scene` subplot.
    """
    vectors = np.asarray(vectors)
    origins3d = _to_3d(origins)
    heads3d = _to_3d(origins3d[:, :2] + scale * vectors)

    xs, ys, zs = [], [], []
    for tail, head in zip(origins3d, heads3d):
        xs += [tail[0], head[0], None]
        ys += [tail[1], head[1], None]
        zs += [tail[2], head[2], None]

        direction = head[:2] - tail[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        barb_len = 0.2 * norm
        unit = direction / norm
        for angle in (2.8, -2.8):  # radians back from the shaft direction: a narrow ~160 degree sweep
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.array([c*unit[0] - s*unit[1], s*unit[0] + c*unit[1]]) * barb_len
            barb = head[:2] + rotated
            xs += [head[0], barb[0], None]
            ys += [head[1], barb[1], None]
            zs += [head[2], head[2], None]

    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=2),
                         opacity=0.6, showlegend=False, hoverinfo='skip')


def plot_mesh(mesh, color='black', width=0.5):
    points3d = _to_3d(mesh.vertices)
    edges = _edges_from_faces(_surface_faces(mesh))
    return _line_trace(points3d, edges, line=dict(color=color, width=width),
                        showlegend=False, hoverinfo='skip')


def plot_boundary(mesh, color='black', width=3):
    points3d = _to_3d(mesh.vertices)
    return _line_trace(points3d, mesh.boundary, line=dict(color=color, width=width),
                        showlegend=False, hoverinfo='skip')


def plot_highlight(mesh, idxs_list, color_list, label_list, mode='vertices'):
    points3d = _to_3d(mesh.vertices)
    traces = []
    if mode == 'vertices':
        for idxs, color, label in zip(idxs_list, color_list, label_list):
            pts = points3d[idxs]
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(size=4, color=color), name=label,
            ))
    elif mode == 'elements':
        faces = _surface_faces(mesh)
        for idxs, color, label in zip(idxs_list, color_list, label_list):
            selected = faces[list(idxs)]
            traces.append(go.Mesh3d(
                x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2],
                i=selected[:, 0], j=selected[:, 1], k=selected[:, 2],
                color=color, opacity=0.2, name=label, showlegend=True,
            ))
    return traces


def plot_colored(mesh, values, cmin=None, cmax=None):
    points3d = _to_3d(mesh.vertices)
    faces = _surface_faces(mesh)
    values = np.asarray(values)

    if len(values) == len(mesh.vertices):
        intensitymode = 'vertex'
    elif len(values) == len(faces):
        intensitymode = 'cell'
    else:
        raise ValueError(
            f'values length {len(values)} matches neither vertex count '
            f'({len(mesh.vertices)}) nor face count ({len(faces)})'
        )

    return go.Mesh3d(
        x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=values, intensitymode=intensitymode,
        colorscale='Viridis', cmin=cmin, cmax=cmax, showscale=True,
    )


def plot_surface(mesh, values, cmin=None, cmax=None):
    """A scalar field as height over a 2D domain. Only meaningful for a
    genuinely-2D mesh -- a mesh with real 3D geometry has no spare axis to
    turn the field into a height; use `plot_colored` for that instead.
    """
    if mesh.vertices.shape[1] != 2:
        raise ValueError(
            'surface mode turns a scalar field into height over a 2D domain; '
            'use colored mode for a mesh with real 3D geometry'
        )

    values = np.asarray(values)
    if values.shape == (len(mesh.vertices),):
        pass
    elif values.shape == (len(mesh.elements),):
        values = mesh.convert_element_values_to_vertex_values(values)
    else:
        raise ValueError(f'Invalid values shape: {values.shape}')

    faces = _surface_faces(mesh)
    return go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=values,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=values, intensitymode='vertex',
        colorscale='Viridis', cmin=cmin, cmax=cmax, showscale=True,
    )


def plot_arrows(mesh, values):
    centroids = np.mean(mesh.vertices[mesh.elements], axis=1)
    scale = _auto_scale(mesh.vertices, values)
    return _vector_field_trace(centroids, values, scale)


def plot_bc(mesh, bc):
    from fem.boundary import BCType

    traces = [plot_mesh(mesh)]
    points3d = _to_3d(mesh.vertices)

    neumann_idxs, neumann_values = [], []
    for bc_type, idxs, values in bc.entries(mesh):
        if not len(idxs):
            continue
        if bc_type is BCType.DIRICHLET:
            pts = points3d[idxs]
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(size=5, color='red'),
                name='dirichlet', showlegend=False,
            ))
        elif bc_type is BCType.NEUMANN:
            neumann_idxs.extend(idxs)
            neumann_values.append(values)

    if neumann_idxs:
        neumann_values = np.concatenate(neumann_values, axis=0)
        scale = _auto_scale(mesh.vertices, neumann_values)
        traces.append(_vector_field_trace(mesh.vertices[neumann_idxs], neumann_values, scale))

    return traces
