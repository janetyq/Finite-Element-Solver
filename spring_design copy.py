import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from BoundaryConditions import *

from Solver import *
from TopologyOptimizer import *

from utils.helper import *
from Mesh import *

# possible issue with compliance calculation
    # weighted by rho?
    # plot against mesh next to rho
# mesh overlap issue

# # Generate rectangular mesh
# corners = [[0, 0], [2, 1]]
# mesher = RectMesher(corners, resolution=(81, 41))
# mesh = mesher.mesh()
# mesh.save('meshes/spring_81x41.pkl')
# mesh.plot(save='temp.png')
 

# Load Mesh
MESH_FILE = 'meshes/spring_80x40.pkl'
mesh = Mesh.load(MESH_FILE)
points, faces, boundary = mesh.get_info()
boundary_idxs = list(set(boundary.ravel()))
mesh.plot(save="temp.png")

w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])


# Boundary conditions
def zero_force(point):
    return np.array([[0, 0]])
def right_force(point):
    if point[0] > w-1e-6:
        return np.array([[0, -100]])
    return np.array([[0, 0]])

spring_bc = BoundaryConditions(mesh)
left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
spring_bc.add('dirichlet', left_idxs, [[0, 0] for idx in left_idxs])
spring_bc.add('dirichlet', right_idxs, [[0.5, 0] for idx in right_idxs])

# Density
with open('rho_bistable1.pkl', 'rb') as f:
    rho = pickle.load(f)

# Solve
equation = Equation('linear_elastic', {'E': 125, 'nu': 0.4})
solver = Solver(mesh, equation, spring_bc, load_function=zero_force)
solver.solve()
solver.solution.plot_deformed('compliance', save="temp3.png")
solver.solution.plot_deformed('rho', save="temp4.png")

# # input()
# # print("compliance:", solver.solution.get_values("total_compliance"))


# # TOPOLOGY OPTIMIZATION
# topoptimizer = TopologyOptimizer(solver, {'iters': 15, 'volume_frac': 1.0})
# topoptimizer.solve(target_compliance=15)
# # topoptimizer.solution.plot_colored('rhos', idx=10, save="results/rho10.png")
# # topoptimizer.solution.plot_colored('rhos', idx=20, save="results/rho20.png")
# topoptimizer.solution.plot_animation('rhos', save='results/topopt.gif', cbar_lim=[0, 1])
# # topoptimizer.solution.plot_deformed('rhos', idx=0, save="temp2.png")

# full_rho = np.full(len(mesh.faces), 1.0)


# plot mesh with rho

mesh.plot_colored(rho, save="temp2.png")

def solve_with_rho(rho, bc=spring_bc):
    equation = Equation('linear_elastic', {'E': 125, 'nu': 0.4, 'rho': np.array(rho)})
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
        compliance = solution.get_values('total_compliance')
        fig, ax = plt.subplots(3, 1)
        solution.plot_deformed('rho', ax=ax[0], title=f'Disp.={dx}, E={compliance:.4f}')
        solution.plot_deformed('force_faces', ax=ax[2], title=f'Disp.={dx}, E={compliance:.4f}')
        solution.plot_deformed(None, ax=ax[1], save=f"temp3.png", title=f'Disp.={dx}, E={compliance:.4f}')

    return results

displacements = np.linspace(-0.4, 1.0, 9, endpoint=True)
results = test_compliance_vs_displacements(rho, displacements)
# results_full = test_compliance_vs_displacements(full_rho, displacements)

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)


# solution = solve_with_rho(rho)
# solution.plot_deformed('rho', save="temp4.png", title=f'Compliance={solution.get_values("total_compliance")}')

# solution = solve_with_rho(full_rho)
# solution.plot_deformed('rho', save="temp2.png", title=f'Compliance={solution.get_values("total_compliance")}')


# # plot results
# results = np.array(results)
# plt.plot(results[:, 0], results[:, 1])
# plt.xlabel("Displacement")
# plt.ylabel("Compliance")
# plt.title("Compliance vs Displacement")
# plt.savefig("results/compliance_vs_displacement.png")
# plt.close()


# print("\n done")


# # # debugging
# # mesh.plot(save="temp.png", idxs=left_idxs)