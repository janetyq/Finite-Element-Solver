import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

sys.path.append('../../')
from Mesh import Mesh
from Plotter import *

# Load mesh
MESH_FILE = '../../meshes/80x40.pkl'
mesh = Mesh.load(MESH_FILE)
vertices, elements, boundary = mesh.get_info()
boundary_idxs = list(set(boundary.ravel()))

# Create a triangulation object
triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], elements)

# Create the plot
fig, ax = plt.subplots()
Plotter(mesh, fig=fig, ax=ax, options={'title': 'Drawer', 'show': False}).plot_mesh(mode='wireframe')

selected_idxs = set()
mouse_pressed = False


def in_triangle(p0, p1, p2, point):
    v0, v1, v2 = p1 - p0, p2 - p0, point - p0
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v < 1)


def is_point_in_triangle(pt, v0, v1, v2):
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v < 1)

def is_point_near_triangle(pt, p0, p1, p2):
    center = (p0 + p1 + p2) / 3
    return np.linalg.norm(center - pt) < 0.04

# Function to handle mouse button press event
def on_press(event):
    global mouse_pressed
    if event.inaxes != ax:
        return
    mouse_pressed = True
    fill_triangle(event)

# Function to handle mouse motion event
def on_motion(event):
    if not mouse_pressed or event.inaxes != ax:
        return
    fill_triangle(event)

# Function to handle mouse button release event
def on_release(event):
    global mouse_pressed
    mouse_pressed = False

# Function to fill the triangle with color if the point is inside
def fill_triangle(event):
    click_point = np.array([event.xdata, event.ydata])
    for i, triangle in enumerate(elements):
        p0, p1, p2 = vertices[triangle]
        v0, v1, v2 = p1 - p0, p2 - p0, click_point - p0
        # if is_point_in_triangle(click_point, v0, v1, v2):
        if is_point_near_triangle(click_point, p0, p1, p2):
            if i not in selected_idxs:
                selected_idxs.add(i)
                ax.fill([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], 'r', alpha=0.5)
                plt.draw()

# Connect the click event to the function
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Show the plot
plt.show()


