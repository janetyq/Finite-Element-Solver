import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import pickle

def topology_optimization(solver, rho, E_0, nu, boundary_conditions, load_function=None, penalization=3, target_fraction=0.5, num_iterations=10, alpha=0.001, beta=0.5, plotting_freq=0):
    solver.initialize_bc(boundary_conditions, load_function)

    topopt_results = []
    for iter in range(num_iterations+1):
        E = 1e-9 + E_0 * rho**penalization
        solver.initialize_material(E, nu)
        result = solver.solve(stress_strain=False)
        C, C_faces = solver.calculate_compliance()
        rho_mean = solver.mesh.calculate_mean_value(rho)

        # print results
        print('\nIteration', iter)
        print('C:', C)
        print(f'Volume ratio: {rho_mean}')
        print('max displacement', np.max(result.displacement, axis=0))

        result.add_var_value('rho', rho)
        result.add_var_value('compliance', C)
        topopt_results.append(result)

        if iter == 0:
            scaling = (len(solver.mesh.faces)) / C
        elif iter == num_iterations:
            # for last iter, want to keep rho and soln consistent
            break

        # update rho
        dC_drho = penalization * E_0 * rho**(penalization-1) * C_faces
        rho += alpha * scaling * dC_drho + beta * min(target_fraction - rho_mean, 0)

        # print('rho average', solver.mesh.calculate_mean_value(rho))
        # print('dC_drho average', solver.mesh.calculate_mean_value(dC_drho))
        # print('num faces', len(solver.faces))
        # print('scale', scaling)
        print('avg. rho change:', solver.mesh.calculate_mean_value(alpha * scaling * dC_drho))
        print('avg. vol correction:', beta * min(target_fraction - rho_mean, 0))

        # smoothing
        smoothed_rho = np.zeros(len(rho))
        for face_idx, face in enumerate(solver.mesh.faces):
            neighbor_value = np.mean([rho[f_idx] for f_idx in solver.he_mesh.get_f_neighbor_f_idxs(face_idx)])
            smoothed_rho[face_idx] = 0.4 * neighbor_value + 0.6 * rho[face_idx]
        rho = smoothed_rho

        rho = np.clip(rho, 0, 1)

        if plotting_freq != 0 and iter != 0 and iter % plotting_freq == 0:
            solver.mesh.plot_colored(rho, title=f'Density (Iteration {iter})', show=True, cbar_label=r'Density $\rho$')

    
    return topopt_results

def save_topopt_results(topopt_results, name='topopt_default', fps=30):
    fig, ax = plt.subplots()
    topopt_results[0].deformed_mesh.plot_colored(topopt_results[0].values['rho'], title='Iteration 0', ax=ax, show=False, cbar_lim=[0, 1])
    def update(frame):
        ax.clear()
        topopt_results[frame].deformed_mesh.plot_colored(topopt_results[frame].values['rho'], title=f'Iteration {frame}', ax=ax, show=False, cbar_present=True)
        return ax

    ani = animation.FuncAnimation(fig, update, frames=range(len(topopt_results)), blit=False, repeat=True)

    write_directory = f'results/{name}'

    if not os.path.exists(write_directory):
        os.makedirs(write_directory)
    else:
        print('WARNING: directory already exists')

    print('writing data to file...')
    with open(f'{write_directory}/topopt_results.pkl', 'wb') as f:
        pickle.dump(topopt_results, f)

    print('writing animation to file...')
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f'{write_directory}/video.mp4', writer=writervideo) 
    print('done!')

def load_topopt_results(name='topopt_default'):
    read_directory = f'results/{name}'
    print('loading data from file...')
    with open(f'{read_directory}/topopt_results.pkl', 'rb') as f:
        topopt_results = pickle.load(f)
    print('done!')
    return topopt_results