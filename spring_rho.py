from Mesh import *
import pickle


# Load Mesh
MESH_FILE = 'meshes/spring_80x40.pkl'
mesh = Mesh.load(MESH_FILE)
points, faces, boundary = mesh.get_info()
boundary_idxs = list(set(boundary.ravel()))
mesh.plot(save="temp.png")

w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])

rho = []
for idx, face in enumerate(mesh.faces):
    x, y = np.mean(mesh.points[face], axis=0)
    y -= 0.5
    if np.abs(y - 0.5*x) < 0.1 or np.abs(y + 0.5*x) < 0.1:
        rho.append(1)
    elif (np.abs(y - 2*(x-0.85)) < 0.1 or np.abs(y + 2*(x-0.85)) < 0.1) and x > 0.85:
        rho.append(1)
    elif np.abs(y) < 0.06 and x > 0.85:
        rho.append(1)
    else:
        rho.append(0.1)


import json
with open('selected_faces.json', 'r') as f:
    selected_faces = json.load(f)
    selected_faces = set([int(face) for face in selected_faces])

rho = []
width = 0.06
for idx, face in enumerate(mesh.faces):
    if idx in selected_faces:
        rho.append(1)
    else:
        rho.append(0)

mesh.plot_colored(rho, save="temp2.png")


# Sine wave
rho = []
for idx, face in enumerate(mesh.faces):
    x, y = np.mean(mesh.points[face], axis=0)
    if np.abs(y - 0.5 - 0.3*sin(10*x)) < 0.1:
        rho.append(1)
    else:
        rho.append(0.1)


# save rho
with open('rho_sine.pkl', 'wb') as f:
    pickle.dump(rho, f)
