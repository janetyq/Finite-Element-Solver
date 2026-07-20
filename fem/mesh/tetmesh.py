"""3D tetrahedral mesh generation, via tetgen (pyvista is tetgen's data
interchange format here, not a rendering dependency -- 3D meshes render through
fem.plot like everything else, via their boundary faces).
"""
import tetgen
import pyvista as pv
import numpy as np

from fem.mesh.mesh import Mesh

def find_boundary_faces(tetrahedrons):
    # This will store the faces and the number of times they appear
    face_count = {}

    # Loop through each tetrahedron
    for tet in tetrahedrons:
        # Extract the four faces from the tetrahedron (combinations of 3 vertices)
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]

        # Count each face's occurrence
        for face in faces:
            if face not in face_count:
                face_count[face] = 0
            face_count[face] += 1

    # Boundary faces are those that appear only once
    boundary_faces = [list(face) for face, count in face_count.items() if count == 1]

    return np.array(boundary_faces)


def create_rect_tetmesh(x_lim, y_lim, z_lim, subdividisions=2):
    points = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    points[:, 0] = points[:, 0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    points[:, 1] = points[:, 1] * (y_lim[1] - y_lim[0]) + y_lim[0]
    points[:, 2] = points[:, 2] * (z_lim[1] - z_lim[0]) + z_lim[0]

    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [0, 2, 4],
        [2, 6, 4],
        [0, 4, 1],
        [1, 4, 5],
        [2, 3, 6],
        [3, 7, 6],
        [1, 5, 3],
        [3, 5, 7],
        [4, 6, 5],
        [5, 6, 7]
    ])

    for _ in range(subdividisions):
        points, faces = subdivide_triangle_mesh(points, faces)

    surface_mesh = pv.PolyData()
    surface_mesh.points = points
    surface_mesh.faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).ravel()

    tet = tetgen.TetGen(surface_mesh)
    tet.tetrahedralize(order=1, mindihedral=10, minratio=1.2)
    grid = tet.grid

    mesh = grid_to_mesh(grid)
    return mesh


def grid_to_mesh(grid):
    vertices = grid.points
    elements = grid.cells.reshape(-1, 5)[:, 1:]
    boundary = find_boundary_faces(elements)
    return Mesh(vertices, elements, boundary)

def mesh_to_grid(mesh):
    cells = np.hstack([np.full((mesh.elements.shape[0], 1), 4), mesh.elements]).ravel()
    celltypes = np.full((mesh.elements.shape[0], 1), pv.CellType.TETRA)
    points = mesh.vertices
    tet_grid = pv.UnstructuredGrid(cells, celltypes, points)
    return tet_grid

def subdivide_triangle_mesh(vertices, faces):
    # divides each triangle face into 4 smaller triangles
    new_vertices = vertices.tolist()
    new_faces = []

    # add midpoints of each edge and create new faces
    for face in faces:
        points = vertices[face]
        midpoints = (np.roll(points, -1, axis=0) + points) / 2

        n = len(new_vertices)
        new_vertices.extend(midpoints)
        new_faces.extend([
            [face[0], n, n + 2],
            [n, face[1], n + 1],
            [n + 1, face[2], n + 2],
            [n, n + 1, n + 2]
        ])

    # remove duplicate vertices and reindex faces
    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)
    final_vertices, indices_map = np.unique(new_vertices, axis=0, return_inverse=True)
    final_faces = indices_map[new_faces]

    return final_vertices, final_faces
