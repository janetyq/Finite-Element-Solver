import sys
import json
import numpy as np

sys.path.append('../../')
from Mesh import *


def get_rho(design=None):
    rho = []
    if design is None:
        rho = [1 for _ in mesh.faces]
    elif isinstance(design, list) or isinstance(design, set):
        for idx, face in enumerate(mesh.faces):
            if idx in design:
                rho.append(1)
            else:
                rho.append(0)
    elif design == 'sine':
        for idx, face in enumerate(mesh.faces):
            x, y = np.mean(mesh.points[face], axis=0)
            if np.abs(y - 0.5 - 0.3*sin(10*x)) < 0.1:
                rho.append(1)
            else:
                rho.append(0)
    elif design == 'bistable_graph1':
        for idx, face in enumerate(mesh.faces):
            x, y = np.mean(mesh.points[face], axis=0)
            y -= 0.5
            if np.abs(y - 0.5*x) < 0.2 or np.abs(y + 0.5*x) < 0.2:
                rho.append(1)
            elif (np.abs(y - 2*(x-0.85)) < 0.05 or np.abs(y + 2*(x-0.85)) < 0.05) and x > 0.85:
                rho.append(1)
            elif np.abs(y) < 0.06 and x > 0.85:
                rho.append(1)
            else:
                rho.append(0)
    return np.array(rho, dtype=np.float32)


if __name__ == '__main__':
    # Load Mesh
    MESH_FILE = '../../meshes/160x80.pkl'
    mesh = Mesh.load(MESH_FILE)
    points, faces, boundary = mesh.get_info()
    boundary_idxs = list(set(boundary.ravel()))
    mesh.plot(save="temp1.png")

    w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])

    
    # with open('results/selected_faces.json', 'r') as f:
    #     selected_faces = json.load(f)
    #     selected_faces = set([int(face) for face in selected_faces])

    rho = get_rho("bistable_graph1")

    with open('results/rho_bistable_graph1.json', 'w') as f:
        json.dump(list([float(r) for r in rho]), f)

    mesh.plot_colored(rho, save="temp2.png")
    

