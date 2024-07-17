import sys
import json
import numpy as np

sys.path.append('../../')
from Mesh import *


def get_rho(design=None):
    rho = []
    if design is None:
        rho = [1 for _ in mesh.elements]
    elif isinstance(design, list) or isinstance(design, set):
        for idx, element in enumerate(mesh.elements):
            if idx in design:
                rho.append(1)
            else:
                rho.append(0)
    elif design == 'sine':
        for idx, element in enumerate(mesh.elements):
            x, y = np.mean(mesh.vertices[element], axis=0)
            if np.abs(y - 0.5 - 0.3*sin(10*x)) < 0.1:
                rho.append(1)
            else:
                rho.append(0)
    elif design == 'bistable_graph1':
        for idx, element in enumerate(mesh.elements):
            x, y = np.mean(mesh.vertices[element], axis=0)
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
    vertices, elements, boundary = mesh.get_info()
    boundary_idxs = list(set(boundary.ravel()))
    mesh.plot(save="temp1.png")

    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])

    
    # with open('results/selected_elements.json', 'r') as f:
    #     selected_elements = json.load(f)
    #     selected_elements = set([int(element) for element in selected_elements])

    rho = get_rho("bistable_graph1")

    with open('results/rho_bistable_graph1.json', 'w') as f:
        json.dump(list([float(r) for r in rho]), f)

    mesh.plot_colored(rho, save="temp2.png")
    

