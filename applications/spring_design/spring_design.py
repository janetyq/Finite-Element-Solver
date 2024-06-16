import os
import sys
import json
import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

sys.path.append('../../')
from BoundaryConditions import *
from Mesh import *
from Solver import *
from TopologyOptimizer import *

from utils.helper import *


# possible issue with compliance calculation
    # weighted by rho?
    # plot against mesh next to rho
# mesh overlap issue

def zero_force(point):
    return np.array([[0, 0]])


# Mesh
MESH_FILE = '../../meshes/160x80.pkl'
mesh = Mesh.load(MESH_FILE)
points, faces, boundary = mesh.get_info()
w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])
boundary_idxs = list(set(boundary.ravel()))
left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
mesh.plot(save="temp1.png")


with open('results/rho_bistable_graph1.json', 'r') as f:
    rho = json.load(f) 

mesh.plot_colored(rho, save="temp2.png")


def solve_with_rho(rho, bc):
    equation = Equation('linear_elastic', {'E': 125, 'nu': 0.0, 'rho': np.array(rho)})
    solver = Solver(mesh, equation, bc, load_function=zero_force)
    solver.solve()
    return solver.solution

def test_compliance_vs_displacements(rho, displacements):
    results = []
    for dx in displacements:
        print(dx)
        bc = BoundaryConditions(mesh)
        bc.add('dirichlet', left_idxs, [[0, 0] for idx in left_idxs])
        bc.add('dirichlet', right_idxs, [[dx, 0] for idx in right_idxs])
        solution = solve_with_rho(rho, bc)
        results.append([dx, solution.__copy__()])

        # mid way plotting
        compliance = solution.get_values('compliance_total')
        fig, ax = plt.subplots(3, 1)
        fig.suptitle(f'Disp.={dx}, E={compliance:.4f}')
        solution.plot_deformed('rho', fig=fig, ax=ax[0], show=False)
        solution.plot_deformed('force', fig=fig, ax=ax[2], show=False)
        solution.plot_deformed(None, fig=fig, ax=ax[1], save=f"temp3.png")

    return results

displacements = np.linspace(-0.4, 1.0, 9, endpoint=True)
results = test_compliance_vs_displacements(rho, displacements)

with open('results/results.pkl', 'wb') as f:
    pickle.dump(results, f)