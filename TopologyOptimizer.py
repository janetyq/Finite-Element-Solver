from Solver import *
from utils.half_edge import *

class TopologyOptimizer:
    def __init__(self, solver, parameters):
        self.solver = solver
        self.parameters = parameters # iters, volume_frac, alpha, beta
        self.solution = Solution(solver.mesh)

    def solve(self):
        he_mesh = HalfEdgeMesh(self.solver.mesh)
        if self.solver.equation.name == 'linear_elastic':
            E_0 = self.solver.equation.parameters['E']
            iters, volume_frac = self.parameters['iters'], self.parameters['volume_frac']
            alpha, beta = self.parameters['alpha'], self.parameters['beta']
            
            E_min, p = 1e-9, 3
            rho = np.full(len(self.solver.mesh.faces), volume_frac)
            
            rhos, us, deformed_meshes, compliances = [], [], [], []
            for iter in range(iters+1):
                self.solver.equation.parameters['E'] = E_min + E_0 * rho**p
                self.solver.solve()
                C = self.solver.solution.values['total_compliance']
                C_faces = self.solver.solution.values['compliance']
                rho_mean = self.solver.mesh.calculate_mean_value(rho)

                u, deformed_mesh = self.solver.solution.values['u'], self.solver.solution.values['deformed_mesh']

                # print results
                print('\nIteration', iter)
                print('C:', C)
                print(f'Volume ratio: {rho_mean}')
                print('max displacement', np.max(u, axis=0))

                if iter == 0:
                    scaling = (len(self.solver.mesh.faces)) / C
                elif iter == iters:
                    # for last iter, want to keep rho and soln consistent
                    break

                # update rho
                dC_drho = p * E_0 * rho**(p-1) * C_faces
                rho += alpha * scaling * dC_drho + beta * min(volume_frac - rho_mean, 0)

                print('avg. rho change:', self.solver.mesh.calculate_mean_value(alpha * scaling * dC_drho))
                print('avg. vol correction:', beta * min(volume_frac - rho_mean, 0))

                # smoothing
                smoothed_rho = np.zeros(len(rho))
                for face_idx, face in enumerate(self.solver.mesh.faces):
                    neighbor_value = np.mean([rho[f_idx] for f_idx in he_mesh.get_f_neighbor_f_idxs(face_idx)])
                    smoothed_rho[face_idx] = 0.4 * neighbor_value + 0.6 * rho[face_idx]
                rho = smoothed_rho

                rho = np.clip(rho, 0, 1)

                us.append(u.copy())
                deformed_meshes.append(deformed_mesh)
                rhos.append(rho.copy())
                compliances.append(C)

            self.solution.set_values('us', us)
            self.solution.set_values('deformed_meshes', deformed_meshes)
            self.solution.set_values('rhos', rhos)
            self.solution.set_values('compliances', compliances)
        else:
            raise ValueError(f'Topology optimization not supported for {self.equation.name} equation')
        pass
    

# def save_topopt_results(topopt_results, name='topopt_default', fps=30):
#     fig, ax = plt.subplots()
#     topopt_results[0].deformed_mesh.plot_colored(topopt_results[0].values['rho'], title='Iteration 0', ax=ax, show=False, cbar_lim=[0, 1])
#     def update(frame):
#         ax.clear()
#         topopt_results[frame].deformed_mesh.plot_colored(topopt_results[frame].values['rho'], title=f'Iteration {frame}', ax=ax, show=False, cbar_present=True)
#         return ax

#     ani = animation.FuncAnimation(fig, update, frames=range(len(topopt_results)), blit=False, repeat=True)

#     write_directory = f'results/{name}'

#     if not os.path.exists(write_directory):
#         os.makedirs(write_directory)
#     else:
#         print('WARNING: directory already exists')

#     print('writing data to file...')
#     with open(f'{write_directory}/topopt_results.pkl', 'wb') as f:
#         pickle.dump(topopt_results, f)

#     print('writing animation to file...')
#     writervideo = animation.FFMpegWriter(fps=fps) 
#     ani.save(f'{write_directory}/video.mp4', writer=writervideo) 
#     print('done!')

# def load_topopt_results(name='topopt_default'):
#     read_directory = f'results/{name}'
#     print('loading data from file...')
#     with open(f'{read_directory}/topopt_results.pkl', 'rb') as f:
#         topopt_results = pickle.load(f)
#     print('done!')
#     return topopt_results