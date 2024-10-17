import sys

import tetgen
import pyvista as pv
import numpy as np

sys.path.append('..')
from Mesh import Mesh

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


def plot_tet(tet_grid, surface):
    # plots tetrahedral mesh cut midway with the surface mesh outline

    # get cell centroids
    cells = tet_grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = tet_grid.points[cells].mean(axis=1)

    # extract cells below the midway
    mask = cell_center[:, 2] < np.median(cell_center[:, 2])
    cell_ind = mask.nonzero()[0]
    subgrid = tet_grid.extract_cells(cell_ind)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
    plotter.add_mesh(surface, 'r', 'wireframe')
    plotter.add_legend([[' Surface Mesh ', 'r'],
                        [' Tet Mesh ', 'black']])
    plotter.show()

def plot_tetmesh(mesh):
    tet_grid = mesh_to_grid(mesh)
    tet_grid.plot(show_edges=True)

def plot_tetmesh_values(mesh, values, clim=None):
    tet_grid = mesh_to_grid(mesh)
    if clim is None:
        clim = [values.min(), values.max()]
    tet_grid.plot(scalars=values, cmap='bwr', clim=clim, flip_scalars=True, show_edges=True)

def plot_tetmesh_animation(mesh, values_array, save_file='tetmesh_animation.gif', title=None):
    tet_grid = mesh_to_grid(mesh)
    tet_grid['v'] = values_array[0]
    plotter = pv.Plotter()
    clim = [min(values_array.flatten()), max(values_array.flatten())]

    if title is not None:
        plotter.add_text(title, font_size=18, color='black', position='upper_edge')
    text = plotter.add_text("", font_size=12, color='black')
    actor = plotter.add_mesh(tet_grid, scalars="v", clim=clim, cmap='bwr', flip_scalars=True, show_edges=True)

    plotter.open_gif(save_file)

    for i in range(len(values_array)):
        text.SetText(0, f"Iter {i}")
        tet_grid["v"] = values_array[i]
        plotter.write_frame()
    
    for i in range(5):
        plotter.write_frame()

    plotter.show()

def create_rect_tetmesh(x_lim, y_lim, z_lim, subdividisions=2, plot=True):
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

    if plot:
        plot_tet(grid, surface_mesh)

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