from Solver import *

class TopologyOptimizer:
    def __init__(self, solver, parameters):
        self.solver = solver
        self.parameters = parameters # iters, volume_frac, alpha, beta
        self.solution = Solution(solver.mesh)

    def solve(self):
        face_neighbors = self.solver.mesh.calculate_face_neighbors()
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

                scaling = (len(self.solver.mesh.faces)) / C

                # update rho
                dC_drho = p * E_0 * rho**(p-1) * C_faces
                rho += alpha * scaling * dC_drho + beta * (volume_frac - rho)

                print('avg. rho change:', self.solver.mesh.calculate_mean_value(alpha * scaling * dC_drho))
                # print('avg. vol correction:', beta * (volume_frac - rho))

                # smoothing
                smoothed_rho = np.zeros(len(rho))
                for face_idx, face in enumerate(self.solver.mesh.faces):
                    neighbor_value = np.mean([rho[neighbor_idx] for neighbor_idx in face_neighbors[face_idx]])
                    smoothed_rho[face_idx] = 0.5 * neighbor_value + 0.5 * rho[face_idx]
                rho = np.clip(smoothed_rho, 0, 1)

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
    
