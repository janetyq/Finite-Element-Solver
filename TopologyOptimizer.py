from utils.helper import *

from Solver import *

class TopologyOptimizer:
    def __init__(self, mesh, equation, boundary_conditions):
        self.solver = Solver(mesh, equation, boundary_conditions)
        self.equation = equation
        assert equation.name == 'linear_elastic',  \
            'Only linear_elastic equation is supported for topology optimization'
        self.face_neighbors = self.solver.mesh.calculate_face_neighbors()
        self.solution = Solution(mesh)

    def filter_sensitivity(self, sensitivity): #TODO: filter better than this
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

    def solve(self, target_compliance=None, plot=False):
        if self.solver.equation.name == 'linear_elastic':
            iters = self.equation.parameters['iters']
            volume_frac = self.equation.parameters['volume_frac']
            p = 3 # penalization factor

            if 'rho' in self.solver.equation.parameters:
                rho = self.solver.equation.parameters['rho']
                assert len(rho) == len(self.solver.mesh.faces), 'rho_initial must have same length as faces'
            else:
                rho = np.full(len(self.solver.mesh.faces), volume_frac)
            
            rhos, us, compliances = [], [], []
            stresses, strains = [], []
            for iter in range(iters):
                self.solver.equation.parameters['rho'] = rho
                self.solver.solve()
                
                u = self.solver.solution.values['u']
                C_faces = self.solver.solution.values['compliance']

                us.append(u.copy())
                rhos.append(rho.copy())
                compliances.append(C_faces)
                stresses.append(self.solver.solution.values['stress'].copy())
                strains.append(self.solver.solution.values['strain'].copy())

                print('\nIteration', iter)
                print(f'{color.BOLD}Total compliance: {np.sum(C_faces):4f} {color.END}')
                print('max displacement', np.max(u, axis=0))
                print('volume fraction:', self.solver.mesh.calculate_mean_value(rho))

                if plot:
                    deformed_mesh = self._get_deformed_mesh()
                    options = {'title': f'Iteration {iter}, C={np.sum(C_faces):.4f}', 'cbar_label': 'Density', 'save': f"results/rho{iter}.png"}
                    Plotter(deformed_mesh, options=options).plot_values(rho)

                # update rho
                sensitivity = p/rho * C_faces
                sensitivity = self.filter_sensitivity(sensitivity)
                rho = self.oc_density(rho, sensitivity, volume_frac)

                # # experimental
                # if target_compliance is not None:
                #     if np.abs(C - target_compliance) < 1e-6:
                #         break
                #     sensitivity = 2*(C_faces - target_compliance) * p/rho * C_faces
                # sensitivity = self.filter_sensitivity(sensitivity)
                # rho += np.clip(0.1 * sensitivity, -0.1, 0.1)
                # rho = np.clip(rho, 1e-6, 1)

            self.solution.set_values('us', us)
            self.solution.set_values('rhos', rhos)
            self.solution.set_values('compliances', compliances)
            self.solution.set_values('stresses', stresses)
            self.solution.set_values('strains', strains)
        else:
            raise ValueError(f'Topology optimization not supported for {self.equation.name} equation')
        
        return self.solution

    def _get_deformed_mesh(self, idx=-1):
        try:
            u = self.solution.values['us'][idx]
        except:
            u = self.solver.solution.values['u']
        return Mesh(self.solver.mesh.points + u.reshape(-1, 2), self.solver.mesh.faces, self.solver.mesh.boundary)

    def plot(self, name, deformed=True, options=None):
        # animation of the optimization process
        options = options if options is not None else {}
        options['cbar_label'] = options.get('cbar_label', name)

        values = self.solution.get_values(name, mode=None) # TODO: mode not supported for list values
        if len(values[0]) == len(self.solver.mesh.faces):
            values = [self.solution._convert_face_values_to_vertex_values(v) for v in values]

        if not deformed:
            plotter = Plotter(self.solver.mesh, options=options) 
            plotter.plot_animation(values, mode='colored')
        else: #TODO: deformed mesh plotting, need to modify plotter
            raise NotImplementedError('Deformed mesh animation not implemented yet')
