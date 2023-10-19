import pickle
from scipy.spatial import Delaunay
from plotting import plot_mesh

def load_points(pkl_file, plot=False):
    global points, triangulation, faces, boundary
    with open(pkl_file, 'rb') as f:
        points = pickle.load(f)
    triangulation = Delaunay(points)
    faces = triangulation.simplices
    boundary = triangulation.convex_hull
    if plot:
        plot_mesh(points, faces)
    return points, faces, boundary

def load_mesh(pkl_file, plot=False):
    global points, faces, boundary
    with open(pkl_file, 'rb') as f:
        mesh = pickle.load(f)
    points = mesh['points']
    faces = mesh['faces']
    boundary = mesh['boundary']
    if plot:
        plot_mesh(points, faces)
    return points, faces, boundary