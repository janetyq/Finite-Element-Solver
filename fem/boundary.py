import logging
from enum import Enum

import numpy as np
from fem.plot.plotter import Plotter, PlotMode

logger = logging.getLogger(__name__)

# supports dirichlet, neumann, and mixed boundary conditions
# add boundary conditions with list of indices and values at indices


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class BoundaryConditions:
    def __init__(self, mesh):
        self.mesh = mesh

        self.neumann = {}
        self.dirichlet = {}
        self.force = {}
        self.load_func = None

    def add(self, bc_type, indices, values):
        bc_type = BCType(bc_type)  # accepts BCType or its value; unknown raises ValueError
        values = np.array(values)
        target = self.dirichlet if bc_type is BCType.DIRICHLET else self.neumann
        if len(indices) == len(values):
            pairs = zip(indices, values)
        else:  # for elastic problems: neumann value = stress, shared by all indices
            pairs = ((v_idx, values) for v_idx in indices)
        for v_idx, value in pairs:
            # Last write wins (plain dict assignment). That's usually a mistake
            # rather than intent, so say so instead of overwriting silently.
            if v_idx in target:
                logger.warning('Overwriting existing %s BC at vertex %s', bc_type.value, v_idx)
            target[v_idx] = value

    def add_force(self, load_func):
        assert len(self.force) == 0, 'load already defined'
        # Kept so the load can be re-evaluated on a remeshed domain; see for_mesh.
        self.load_func = load_func
        for v_idx in range(len(self.mesh.vertices)):
            self.force[v_idx] = np.array(load_func(self.mesh.vertices[v_idx]))

    def check_remeshable(self):
        '''Raise unless these conditions can be carried onto a different mesh.

        A body force is specified as a function of position, so it re-evaluates
        on any mesh. Dirichlet and Neumann conditions are stored per vertex
        *index*, and a remesher renumbers vertices -- carrying them over would
        silently move them to unrelated nodes. Supporting that needs
        position-based specs; see BACKLOG.md section 1.
        '''
        if self.dirichlet or self.neumann:
            raise NotImplementedError(
                'Dirichlet/Neumann conditions are index-based and cannot survive '
                'remeshing, which renumbers vertices. Only a position-defined '
                'body force (add_force) can be transferred today.'
            )

    def for_mesh(self, mesh):
        '''These conditions, resolved against `mesh`.'''
        self.check_remeshable()
        new = BoundaryConditions(mesh)
        if self.load_func is not None:
            new.add_force(self.load_func)
        return new

    def check(self):
        '''Validate boundary conditions, catching two otherwise-silent footguns:

        1. Dirichlet/Neumann conditions must live on boundary vertices.
        2. No vertex may carry both a Dirichlet and a Neumann condition (a fixed
           node ignores its Neumann load, so the pairing is ambiguous).
        '''
        boundary = set(self.mesh.boundary_idxs)
        interior = (set(self.dirichlet) | set(self.neumann)) - boundary
        if interior:
            raise ValueError(f'Boundary conditions on non-boundary vertices: {sorted(interior)}')
        overlap = set(self.dirichlet) & set(self.neumann)
        if overlap:
            raise ValueError(f'Vertices carry both Dirichlet and Neumann conditions: {sorted(overlap)}')

    def do(self, N, dim):
        self.check()
        # dirichlet boundary conditions
        self.fixed_idxs = [dim*v_idx + d  for v_idx in self.dirichlet.keys() for d in range(dim)]
        self.fixed_values = list(np.array(list(self.dirichlet.values())).flatten())
        self.free_idxs = [dim*v_idx + d  for v_idx in list(set(range(N)) - set(self.dirichlet.keys())) for d in range(dim)]

        self.neumann_load = []
        self.force_load = []
        for v_idx in range(len(self.mesh.vertices)):
            self.neumann_load.append(self.neumann[v_idx] if v_idx in self.neumann else np.zeros(dim))
            self.force_load.append(self.force[v_idx] if v_idx in self.force else np.zeros(dim))
        self.neumann_load = np.array(self.neumann_load)
        self.force_load = np.array(self.force_load)  

    def plot(self):
        plotter = Plotter(title='Boundary conditions')
        plotter.plot(self.mesh, bc=self, mode=PlotMode.BC)
        plotter.show()