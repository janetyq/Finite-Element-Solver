import sys

import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from helper import *

sys.path.append('../../')
from BoundaryConditions import *
from Solver import *
from TopologyOptimizer import *

from utils.helper import *
from Mesh import *


# Parameters
w, h = 3.0, 2.0 # domain width, height
nw, nh = 49, 33 
c = 4
sink_x1, sink_x2 = 1.0, 2.0
sink_y1, sink_y2 = 0, 1

# Mesh
vertices, elements, boundary_indices, boundary_edges = generate_rect_mesh(w, h, nw, nh)
mesh = Mesh(vertices, elements, boundary_edges)

# Boundary
flux_boundary = mesh.get_boundary_idxs_in_rect([sink_x1, 0, sink_x2, 0]) # bottom of heat sink
cold_boundary = [idx for idx in boundary_indices if idx not in flux_boundary]

flux_edges = mesh.get_edges_in_idxs(flux_boundary)
cold_edges = mesh.get_edges_in_idxs(cold_boundary)

# Sink
sink_boundary = []
for i, vertex in enumerate(vertices):
    x, y = vertex
    if (x == sink_x1 or x == sink_x2) and (sink_y1 <= y <= sink_y2):
        sink_boundary.append(i)
    elif (y == sink_y1 or y == sink_y2) and (sink_x1 <= x <= sink_x2):
        sink_boundary.append(i)
sink_edges = mesh.get_edges_in_idxs(sink_boundary, exclude_corners=True)

sink_elements = []
air_elements = []
for element_idx in range(len(elements)):
    x, y = np.mean(vertices[elements[element_idx]], axis=0)
    if (sink_x1 <= x <= sink_x2) and (sink_y1 <= y <= sink_y2):
        sink_elements.append(element_idx)
    else:
        air_elements.append(element_idx)

def plot_stuff(mesh, control_handles, fixed_handles, save=None):
    color_elements = [['lightblue', air_elements, r"$\Omega_a (air)$"], ['gray', sink_elements, r"$\Omega_s (heat sink)$"]]

    fig, ax = plt.subplots()
    Plotter(mesh, fig=fig, ax=ax, options={'title': 'Mesh', 'show': False, 'save': save}).plot_mesh(mode='wireframe', color_elements=color_elements)
    vertices = mesh.vertices
    ax.scatter(control_handles[:, 0], control_handles[:, 1], c='green', s=40, label="Control handles")
    ax.scatter(fixed_handles[:, 0], fixed_handles[:, 1], c='black', s=40, label="Fixed handles")
    for edge in sink_edges:
        ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]], [vertices[edge[0]][1], vertices[edge[1]][1]], c='black', linewidth=2)
    ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]], [vertices[edge[0]][1], vertices[edge[1]][1]], c='black', linewidth=2, label="Heat Sink")
    for edge in flux_edges:
        ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]], [vertices[edge[0]][1], vertices[edge[1]][1]], c='red', linewidth=3)
    for edge in cold_edges:
        ax.plot([vertices[edge[0]][0], vertices[edge[1]][0]], [vertices[edge[0]][1], vertices[edge[1]][1]], c='gray', linewidth=3)
    ax.plot([0], [0], c='red', linewidth=3, label=r"$\Gamma_h$")
    ax.plot([0], [0], c='gray', linewidth=3, label=r"$\Gamma_c$")

    ax.legend(loc='upper right')
    plt.show()


# Thermal conductivity
def k_func(e_idx):
    if e_idx in sink_elements:
        return 10.0
    else:
        return 1.0

# Handles
fixed_handles = []
for i in range(0, nw, c):
    for j in range(0, nh, c):
        idx = i * nh + j
        if idx in boundary_indices:
            fixed_handles.append([w*i/(nw-1), h*j/(nh-1)])
control_handles = []
for i in range(0, nw, c):
    for j in range(0, nh, c):
        idx = i * nh + j
        if idx not in flux_boundary and idx in sink_boundary:
            control_handles.append([w*i/(nw-1), h*j/(nh-1)])

control_handles = np.array(control_handles, dtype=np.float64)
fixed_handles = np.array(fixed_handles, dtype=np.float64)
all_handles = np.concatenate([control_handles, fixed_handles], axis=0)
print(f"Control handles: {control_handles.shape}, Fixed handles: {fixed_handles.shape}")


# plot handles
plot_stuff(mesh, control_handles, fixed_handles)

# Handle weights
weights = bbw_weights(vertices, elements, all_handles)
lbs = igl.lbs_matrix(vertices, weights)

# PDE solver
equation = Equation('poisson', parameters={'func': k_func})
bc = BoundaryConditions(mesh)
bc.add('neumann', flux_boundary, [100])
bc.add('dirichlet', cold_boundary, [0])
solver = Solver(mesh, equation, bc)


######################################################################################################

# FUNCTIONS

def get_edge_loss(vertices):
    loss = 0
    for i in range(len(vertices)):
        x, y = vertices[i]
        loss += 1/min(x, w-x)**2 + 1/min(y, h-y)**2
    loss /= len(vertices)
    return loss

cbar_lim = None

def run(control_input: np.ndarray, plot=False, save=None):
    global cbar_lim
    new_control_handles = control_handles + control_input
    new_all_handles = np.concatenate([new_control_handles, fixed_handles], axis=0).copy()
    handle_transforms = get_handle_transforms(all_handles, new_all_handles)
    new_vertices = lbs @ handle_transforms
    new_mesh = Mesh(new_vertices, elements, boundary_edges)
    solver.set_mesh(new_mesh)
    solver.solve()
    
    average_bottom_temp = np.mean(solver.solution.values['u'][flux_boundary])
    obj_cost = average_bottom_temp
    control_cost = 5*np.sum(np.linalg.norm(control_input, axis=1))

    tri_area = w*h/len(new_mesh.areas)
    edge_loss = 0.1*get_edge_loss(new_control_handles)

    costs = [obj_cost, edge_loss]
    total_cost = sum(costs)

    if plot:
        print(f"Costs: {[round(cost, 3) for cost in costs]}")
        plot_stuff(new_mesh, new_control_handles, fixed_handles, save=f"{save}_1")

        title = "Initial Solution" if np.allclose(control_input, 0) else "Optimized Solution"
        fig, ax = plt.subplots()
        values = solver.solution.values['u']
        cbar_lim = (min(values), max(values)) if cbar_lim is None else cbar_lim
        Plotter(mesh, fig=fig, ax=ax, options={'title': title, 'show': False, 'cbar_label': 'Temperature'}).plot_values(values, mode='colored')
        for edge in sink_edges:
            ax.plot([new_vertices[edge[0]][0], new_vertices[edge[1]][0]], [new_vertices[edge[0]][1], new_vertices[edge[1]][1]], c='black')
        plt.savefig(f"{save}_2.png")

        with open(f'{save}_handles.pkl', 'wb') as f:
            pickle.dump(control_input, f)
    
    return total_cost, costs


def simulated_annealing(run, initial_solution, temperature, change=0.01, cooling_rate=0.99, max_iterations=200):
    current_solution = initial_solution
    best_solution = current_solution
    curr_cost = run(current_solution)[0]
    best_cost = curr_cost

    for i in range(max_iterations):
        # Generate a new candidate solution by perturbing the current solution
        new_solution = current_solution + np.random.normal(0, change, size=current_solution.shape)
        
        # Compute the cost difference between the current and new solutions
        new_cost = run(new_solution)[0]
        print(f"Iter: {i}, Cost: {new_cost}")
        delta_cost = new_cost - curr_cost

        # If the new solution is better, accept it
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / temperature):
            current_solution = new_solution
            curr_cost = new_cost
            print("Better" if delta_cost < 0 else "Worse")
        
        # Update the best solution found so far
        if curr_cost < best_cost:
            best_solution = current_solution
            best_cost = curr_cost
        
        # Cool down the temperature
        temperature *= cooling_rate

        if i % 5 == 0:
            print(f"Iteration {i}, Temperature {temperature}, Best {best_cost}, Curr {curr_cost}")
            run(best_solution, plot=True)
    
    return best_solution

def gradient_estimator(control_input, run, init_cost=None):
    epsilon = 1e-3
    gradients = np.zeros_like(control_input)
    init_cost = run(control_input)[0] if init_cost is None else init_cost
    for i in range(len(control_input)):
        for j in range(2):
            perturbed_input = control_input.copy()
            perturbed_input[i][j] += epsilon
            cost = run(perturbed_input, plot=False)[0]
            gradients[i][j] = (cost - init_cost) / epsilon
            perturbed_input = control_input.copy()
    return gradients

# TODO:
# better optimization algo and comparisons
#   genetic
#   iterative algos
#   simulated annealing
#   gradient descent

# easier/more problem specifications - boundary conditions, num handles, etc.

# actual gradient OR auto diff

if __name__ == '__main__':
    import json
    cost_history = []

    # Gradient descent
    control_input = np.zeros_like(control_handles)
    alpha = 0.005
    for i in range(100):
        cost, costs = run(control_input, plot=True, save=f"../../results/cage_{i}")
        gradients = gradient_estimator(control_input, run, init_cost=cost)
        control_input -= np.clip(alpha * gradients, -0.05, 0.05)
        print(f"Iter: {i}, Cost: {cost}, Alpha: {alpha}")
        cost_history.append(costs)

        alpha *= 0.93

    print(f"Cost history: {cost_history}")

    with open('../../results/cage_costs.json', 'wb') as f:
        f.write(json.dumps(cost_history))

    # Random sampling
    results = []
    for i in range(2):
        control_input = 0.5*(np.random.rand(*control_handles.shape) - 0.5)
        output = run(control_input, plot=True)
        results.append((control_input, output))

