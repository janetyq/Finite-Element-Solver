import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.colorbar import Colorbar
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# random useful functions
def bump_function(vertices, center, mag=100, size=0.5):
    return np.array([mag*cos(pi/2*np.linalg.norm(point - center)/size) if np.linalg.norm(point - center) < size else 0 for point in vertices])

def calculate_polygon_area(polygon):
    x, y = polygon.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_tetrahedron_volume(tetrahedron): # TODO: similar for triangle?
    a, b, c = tetrahedron[1:] - tetrahedron[0]
    return np.abs(np.dot(a, np.cross(b, c)) / 6)

def point_in_polygon(point, polygon):
    x, y = point
    x_coords, y_coords = polygon.T
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i+1) % n], y_coords[(i+1) % n]
        if y1 < y <= y2 or y2 < y <= y1:
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                inside = not inside
    return inside

def calculate_circumcenter(triangle_points):
    edge_vectors = [triangle_points[(i+1)%3] - triangle_points[i] for i in range(3)]
    edge_midpoints = [0.5 * (triangle_points[i] + triangle_points[(i+1)%3]) for i in range(3)]
    edge_perps = [[vec[1], -vec[0]] for vec in edge_vectors]

    # remove bisectors with 0 slope
    for i in range(3):
        if edge_perps[i][0] == 0:
            edge_perps.pop(i)
            edge_midpoints.pop(i)
            break

    # calculate center using intersection of bisectors
    s1, s2 = [perp[1] / perp[0] for perp in edge_perps[:2]]
    m1, m2 = edge_midpoints[:2]
    x = (m2[1] - m1[1] + m1[0]*s1 - m2[0]*s2) / (s1 - s2)
    y = m1[1] + s1*(x - m1[0])
    center = [x, y]

    return center

def calculate_triangle_min_angle(triangle):
    # returns the smallest angle (degrees) in the triangle
    lengths = np.linalg.norm([triangle[i] - triangle[(i+1)%3] for i in range(3)], axis=1)
    angles = np.arccos([
        (lengths[1]**2 + lengths[2]**2 - lengths[0]**2) / (2 * lengths[1] * lengths[2]),
        (lengths[2]**2 + lengths[0]**2 - lengths[1]**2) / (2 * lengths[2] * lengths[0]),
        (lengths[0]**2 + lengths[1]**2 - lengths[2]**2) / (2 * lengths[0] * lengths[1])
    ])
    return np.min(angles) * 180 / np.pi

def get_boundary_from_vertices_elements(vertices, elements):
    edges = set()
    boundary_edges = set()

    # Step 1: Convert elements to edges
    for element in elements:
        for i in range(3):  # Each element is a triangle (3 vertices)
            edge = tuple(sorted([element[i], element[(i + 1) % 3]]))  # Edges are represented by sorted vertex indices
            if edge in edges:
                # If edge is already in set, it's an interior edge, remove it from edges
                edges.remove(edge)
            else:
                # If edge is not in set, it's a new edge, add it to edges
                edges.add(edge)

    # Step 2: Identify boundary edges
    for edge in edges:
        count = 0
        for element in elements:
            if edge[0] in element and edge[1] in element:
                count += 1
        if count == 1:
            boundary_edges.add(edge)

    boundary_edges = [list(edge) for edge in boundary_edges]

    return boundary_edges

# Material properties
def Enu_to_Lame(E, nu):
    # mu - shear modulus, lambda - Lame constant
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lamb

def Lame_to_Enu(mu, lamb):
    # E - Young's modulus, nu - Poisson's ratio
    E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
    nu = lamb / (2 * (lamb + mu))
    return E, nu

# Printing
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Gradient checking - TODO: make faster
def check_gradient(function, gradient, input_shape):
    u = np.random.random(input_shape)
    computed_gradient = gradient(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_gradient = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = function(u + eps * direction)
            eval_m = function(u - eps * direction)
            numerical_gradient.append((eval_p - eval_m) / (2 * eps))
        numerical_gradient = np.array(numerical_gradient).reshape(computed_gradient.shape)
        # print(f'numerical_gradient: {numerical_gradient} \ncomputed_gradient: {computed_gradient}')
        errors_list.append(np.linalg.norm(numerical_gradient - computed_gradient))
    
    plt.title('Gradient check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()

def check_hessian(gradient, hessian, input_shape):
    u = np.random.random(input_shape)
    computed_hessian = hessian(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_hessian = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = gradient(u + eps * direction)
            eval_m = gradient(u - eps * direction)
            numerical_hessian.append((eval_p - eval_m) / (2 * eps))
        numerical_hessian = np.array(numerical_hessian).reshape(computed_hessian.shape)
        errors_list.append(np.linalg.norm(numerical_hessian - computed_hessian))
    
    plt.title('Hessian check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()

# Decorators
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end - start} seconds')
        return result
    return wrapper

def calculate_smoothing_matrix(mesh, r):
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    weight_matrix = np.maximum(0, r - distances)
    normalized_weight_matrix = weight_matrix / (weight_matrix.sum(axis=1)[:, np.newaxis] + 1e-16)
    return normalized_weight_matrix


## plotting helpers
def plot_mesh(ax, mesh, color='black', linewidth=0.2):
    ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements, color=color, linewidth=linewidth)

def plot_boundary(ax, mesh, color='black', linewidth=1.0):
    for seg in mesh.boundary:
        ax.plot(mesh.vertices[seg, 0], mesh.vertices[seg, 1], color=color, linewidth=linewidth)

def plot_highlight(ax, mesh, idxs_list, color_list, label_list, mode='vertices'):
    for idxs, color, label in zip(idxs_list, color_list, label_list):
        if mode == 'vertices':
            ax.scatter(mesh.vertices[idxs, 0], mesh.vertices[idxs, 1], color=color, s=5, label=label)
        elif mode == 'elements':
            first = True  # Handle the label only for the first element
            for e_idx in idxs:
                vertices = mesh.vertices[mesh.elements[e_idx]]
                ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.2, label=label if first else None)
                first = False

def plot_arrows(ax, mesh, values):
    element_vertices = np.mean(mesh.vertices[mesh.elements], axis=1)
    ax.quiver(element_vertices[:, 0], element_vertices[:, 1], values[:, 0], values[:, 1], alpha=0.5, scale=10)

def setup_colorbar(ax, vlim, label=None):
    cmap = cm.get_cmap('viridis')  # Choose a colormap
    norm = plt.Normalize(vmin=vlim[0], vmax=vlim[1])  # Normalize values between vmin and vmax

    # Create a scalar mappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label)
    return cmap, norm

def plot_colored(ax, mesh, values, cbar_info=None):
    if cbar_info is None:
        cbar_info = setup_colorbar(ax, (min(values), max(values)), None)

    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    tripcolor = ax.tripcolor(triangulation, values, cmap=cbar_info[0], norm=cbar_info[1])

    # TODO: check contour
    # if contour > 0:
    #     ax.tricontour(triangulation, values, levels=np.linspace(min(values), max(values), contour), colors='k', linestyles='solid')
    return cbar_info

def change_ax_to_ax3d(ax, fig, ax_shape, ax_idx):
    if hasattr(ax, 'get_zlim'):
        return ax
    ax.remove()
    n = ax_shape[0]*100 + ax_shape[1]*10 + ax_idx[0]*ax_shape[1] + ax_idx[1] + 1
    ax = fig.add_subplot(n, projection='3d')
    return ax

def plot_surface(ax, mesh, values):
    if values.shape == (len(mesh.vertices),):
        pass
    elif values.shape == (len(mesh.elements),):
        values = mesh.convert_element_values_to_vertex_values(values)
    else:
        raise ValueError(f'Invalid values shape: {values.shape}')
    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    surf = ax.plot_trisurf(triangulation, values, cmap='viridis')

def plot_arrows(ax, mesh, values):
    # TODO: colored arrows, hard to see scale currently
    element_vertices = np.mean(mesh.vertices[mesh.elements], axis=1)
    ax.quiver(element_vertices[:, 0], element_vertices[:, 1], values[:, 0], values[:, 1], alpha=0.5, scale=10)

def plot_bc(ax, mesh, bc):
    plot_mesh(ax, mesh)
    for v_idx, value in bc.dirichlet.items():
        ax.plot(mesh.vertices[v_idx][0], mesh.vertices[v_idx][1], 'ro')
    for v_idx, value in bc.neumann.items():
        ax.quiver(mesh.vertices[v_idx][0], mesh.vertices[v_idx][1], value[0], value[1])