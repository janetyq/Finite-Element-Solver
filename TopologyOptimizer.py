from Solver import *

class TopologyOptimizer:
    def __init__(self, solver, parameters):
        self.solver = solver
        self.parameters = parameters # iters, volume_frac, alpha, beta
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
            rho_new = np.clip(rho_new, 0, 1) 

            if self.solver.mesh.calculate_mean_value(rho_new) < volume_frac:
                r = m
            else:
                l = m
        return rho_new


    def solve(self):
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
                # TODO: print as table
                print('\nIteration', iter)
                print('C:', C)
                print(f'Volume ratio: {rho_mean}')
                print('max displacement', np.max(u, axis=0))

                scaling = (len(self.solver.mesh.faces)) / C

                # update rho
                # dC_drho = p * E_0 * rho**(p-1) * C_faces
                # rho += alpha * scaling * dC_drho + beta * min(volume_frac - rho_mean, 0)
                sensitivity = self.filter_sensitivity(p * E_0 * rho**(p-1) * C_faces)
                rho = self.oc_density(rho, sensitivity, volume_frac)

                # # smoothing
                # smoothed_rho = np.zeros(len(rho))
                # for face_idx, face in enumerate(self.solver.mesh.faces):
                #     neighbor_value = np.mean([rho[neighbor_idx] for neighbor_idx in face_neighbors[face_idx]])
                #     smoothed_rho[face_idx] = 0.5 * neighbor_value + 0.5 * rho[face_idx]
                # rho = np.clip(smoothed_rho, 0, 1)

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

    def solve_with_target_compliance(self, target_compliance):
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

                scaling = (len(self.solver.mesh.faces)) / C

                # update rho
                # dC_drho = p * E_0 * rho**(p-1) * C_faces
                sensitivity = 2*(C - target_compliance) * p * E_0 * rho**(p-1) * C_faces # gradient of objective vs rho
                rho = self.oc_density(rho, sensitivity, volume_frac)

                # print('avg. rho change:', self.solver.mesh.calculate_mean_value(alpha * scaling * dobjective_drho))
                # print('avg. vol correction:', beta * min(volume_frac - rho_mean, 0))

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

    def solve_with_target_compliance(self, target_compliance):
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
                # dC_drho = p * E_0 * rho**(p-1) * C_faces
                sensitivity = 2*(C - target_compliance) * p * E_0 * rho**(p-1) * C_faces # gradient of objective vs rho
                rho = self.oc_density(rho, sensitivity, volume_frac)

                # print('avg. rho change:', self.solver.mesh.calculate_mean_value(alpha * scaling * dobjective_drho))
                # print('avg. vol correction:', beta * min(volume_frac - rho_mean, 0))

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
    
