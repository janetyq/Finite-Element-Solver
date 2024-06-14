import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from BoundaryConditions import *

from Solver import *
from TopologyOptimizer import *

from utils.helper import *
from Mesh import *

# Generate rectangular mesh
# corners = [[0, 0], [2, 1]]
# mesher = RectMesher(corners, resolution=(40, 20))
# mesh = mesher.mesh()
# mesh.save('meshes/spring_40x20.pkl')
# mesh.plot(save='temp.png')


# Load Mesh
MESH_FILE = 'meshes/spring_40x20.pkl'
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
spring_bc.add('dirichlet', right_idxs, [[0.25, 0] for idx in right_idxs])

# Solve
equation = Equation('linear_elastic', {'E': 125, 'nu': 0.4})
solver = Solver(mesh, equation, spring_bc, load_function=zero_force)
solver.solve()
solver.solution.plot_deformed('stress', save="results/stress_deformed.png")
print("compliance:", solver.solution.get_values("total_compliance"))

# TOPOLOGY OPTIMIZATION
topoptimizer = TopologyOptimizer(solver, {'iters': 25, 'volume_frac': 0.5, 'alpha': 0.0002, 'beta': 0.0})
topoptimizer.solve()
# topoptimizer.solve_with_target_compliance(1.0)
topoptimizer.solution.plot_colored('rhos', idx=10, save="results/rho10.png")
topoptimizer.solution.plot_colored('rhos', idx=20, save="results/rho20.png")
topoptimizer.solution.plot_animation('rhos', save='results/topopt.gif', cbar_lim=[0, 1])




# Plot compliance vs displacement
results = []

displacements = np.linspace(-0.5, 0.5, 5, endpoint=True)
for dx in displacements:
    bc = BoundaryConditions(mesh)
    bc.add('dirichlet', left_idxs, [[0, 0] for idx in left_idxs])
    bc.add('dirichlet', right_idxs, [[dx, 0] for idx in right_idxs])
    solver = Solver(mesh, equation, bc, load_function=zero_force)
    solver.solve()
    
    compliance = solver.solution.get_values("total_compliance")
    results.append([dx, compliance])


# plot results
results = np.array(results)
plt.plot(results[:, 0], results[:, 1])
plt.xlabel("Displacement")
plt.ylabel("Compliance")
plt.title("Compliance vs Displacement")
plt.savefig("results/compliance_vs_displacement.png")
plt.close()


print("\n done")


# # debugging
# mesh.plot(save="temp.png", idxs=left_idxs)