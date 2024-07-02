from Solver import *

class TopologyOptimizer:
    def __init__(self, solver, parameters):
        self.solver = solver
        self.parameters = parameters # iters, volume_frac, rho=3
        self.solution = Solution(solver.mesh)
        self.face_neighbors = self.solver.mesh.calculate_face_neighbors()

    def filter_sensitivity(self, sensitivity):
        # simple averaging with neighbors
        smoothed_sensitivity = np.zeros_like(sensitivity)
        for face_idx, face in enumerate(self.solver.mesh.faces):
            neighbor_value = np.mean([sensitivity[neighbor_idx] for neighbor_idx in self.face_neighbors[face_idx]])
            smoothed_sensitivity[face_idx] = 0.5 * neighbor_value + 0.5 * sensitivity[face_idx]
        return smoothed_sensitivity

    def oc_density(self, rho, sensitivity, volume_frac):
        # sensitivity is the gradient of the compliance with respect to the density
        l, r = 0.0, 1e15 # search interval
        while (l*(1+1e-15)) < r:
            m = 0.5*(l+r)
            rho_new = rho * np.sqrt(sensitivity / m)
            # rho_new = np.clip(rho_new, rho_new - 0.1, rho_new + 0.1) # change limit
            rho_new = np.clip(rho_new, 1e-6, 1) 

            if self.solver.mesh.calculate_mean_value(rho_new) < volume_frac:
                r = m
            else:
                l = m
        return rho_new

    def solve(self, target_compliance=None, rho_initial=None):
        if self.solver.equation.name == 'linear_elastic':
            iters, volume_frac = self.parameters['iters'], self.parameters['volume_frac']
            
            E, p = self.solver.equation.parameters['E'], 3
            if rho_initial is not None:
                assert len(rho_initial) == len(self.solver.mesh.faces), 'rho_initial must have same length as faces'
                rho = rho_initial
            else:
                rho = self.solver.boundary_conditions.rho
            
            rhos, us, deformed_meshes, compliances = [], [], [], []
            for iter in range(iters):
                self.solver.boundary_conditions.set_rho(rho) # update rho
                self.solver.solve()
                C = self.solver.solution.values['compliance_total']
                C_faces = self.solver.solution.values['compliance']

                u, deformed_mesh = self.solver.solution.values['u'], self.solver.solution.values['deformed_mesh']

                us.append(u.copy())
                deformed_meshes.append(deformed_mesh)
                rhos.append(rho.copy())
                compliances.append(C)

                # print results
                print('\nIteration', iter)
                print('C:', C)
                print('max displacement', np.max(u, axis=0))
                print('volume fraction:', self.solver.mesh.calculate_mean_value(rho))

                deformed_mesh.plot_colored(rho, save=f"results/rho{iter}.png", cbar_label='Density', title=f'Iteration {iter}, C={C:.4f}')

                # update rho
                if target_compliance is None:
                    sensitivity = p/rho * C_faces
                else:
                    sensitivity = 2*(C_faces - target_compliance) * p/rho * C_faces
                sensitivity = self.filter_sensitivity(sensitivity)
                if volume_frac == 1:
                    rho += np.clip(0.1 * sensitivity, -0.1, 0.1)
                    rho = np.clip(rho, 1e-6, 1)
                else:
                    rho = self.oc_density(rho, sensitivity, volume_frac)

            self.solution.set_values('us', us)
            self.solution.set_values('deformed_mesh', deformed_meshes)
            self.solution.set_values('rhos', rhos)
            self.solution.set_values('compliances', compliances)
        else:
            raise ValueError(f'Topology optimization not supported for {self.equation.name} equation')
        pass
