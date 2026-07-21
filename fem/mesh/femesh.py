from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from fem.mesh.mesh import Mesh
from fem.elements import LinearElement, LinearTriangleElement
from fem.typing import (
    DofIndices,
    ElementField,
    Elements,
    FloatArray,
    IntArray,
    Matrix,
    VertexField,
    Vertices,
)


def dof_indices(element: IntArray | Sequence[int], n_components: int) -> DofIndices:
    '''Global DOF indices for an element's nodes, interleaved per node.

    For node indices [n0, n1, ...] and `n_components` DOFs per node, returns
    [n_components*n0, n_components*n0+1, ..., n_components*n1, n_components*n1+1, ...].
    '''
    element = np.asarray(element)
    return np.array([n_components*element + i for i in range(n_components)]).T.flatten()


class FEMesh(Mesh):
    '''
    Built on top of Mesh class
    Adds functionality for FEM calculations (ie. element areas, gradients, integrals)
    '''
    def __init__(
        self,
        vertices: Vertices | Sequence[Sequence[float]],
        elements: Elements | Sequence[Sequence[int]],
        boundary: Elements | Sequence[Sequence[int]],
        element_type: type[LinearElement] = LinearTriangleElement,
    ) -> None:
        Mesh.__init__(self, vertices, elements, boundary)

        self.element_type = element_type
        if element_type.SUB_TYPE is None:
            # Only reachable for line elements, whose facets would be points --
            # the 1D path the SUB_TYPE TODO tracks. Raising here beats a
            # "NoneType is not callable" from the comprehension below.
            raise NotImplementedError(
                f'{element_type.__name__} has no boundary element type, so boundary '
                f'integrals (and hence an FEMesh) are not defined for it yet'
            )
        self.boundary_type = element_type.SUB_TYPE
        self.element_objs = [self.element_type(self.vertices[element]) for element in self.elements]
        self.boundary_objs = [self.boundary_type(self.vertices[boundary]) for boundary in self.boundary]

        self.prepare_matrices()

    def prepare_matrices(self, n_components: int = 1, **kwargs: Any) -> None:
        # Can be called to reprepare matrices for a different component count
        self.n_components = n_components
        self.M = self.assemble_matrix('mass', 'element', n_components, **kwargs)
        self.M_b = self.assemble_matrix('mass', 'boundary', n_components, **kwargs)
        self.K = self.assemble_matrix('stiffness', 'element', n_components, **kwargs)
        if n_components == 1:
            self.K_b = self.assemble_matrix('stiffness', 'boundary', n_components, **kwargs)
        else:
            # Boundary stiffness is only used by the 1D scalar path; the n_components>1
            # (linear-elastic) solvers never touch K_b. Set None so the attribute
            # always exists and misuse fails loudly instead of as AttributeError.
            self.K_b = None

    def assemble_matrix(
        self,
        matrix_type_name: Literal['mass', 'stiffness'],
        element_type_name: Literal['element', 'boundary'],
        n_components: int = 1,
        **kwargs: Any,
    ) -> Matrix:
        # TODO: term "element" is overloaded here, and its a bit hacky

        if element_type_name == 'element':
            elements, element_objs = self.elements, self.element_objs
        elif element_type_name == 'boundary':
            elements, element_objs = self.boundary, self.boundary_objs

        matrix_calculators = {
            'mass': lambda e_idx: element_objs[e_idx].calculate_mass_matrix(n_components, idx=e_idx, **kwargs),
            'stiffness': lambda e_idx: element_objs[e_idx].calculate_stiffness_matrix(n_components, idx=e_idx, **kwargs),
        }

        N = len(self.vertices)
        A = np.zeros((n_components * N, n_components * N))
        for e_idx, element in enumerate(elements):
            idxs = dof_indices(element, n_components)
            element_matrix = matrix_calculators[matrix_type_name](e_idx)
            A[np.ix_(idxs, idxs)] += element_matrix
        return A

    # METRICS
    def calculate_total_value(self, u: VertexField | ElementField) -> float:
        if len(u) == len(self.elements):       # u defined on elements
            return float(sum([self.element_objs[e_idx].volume * u[e_idx] for e_idx in range(len(self.elements))]))
        elif len(u) == len(self.vertices):    # u defined on vertices
            return float(sum([self.element_objs[e_idx].volume * np.mean(u[self.elements[e_idx]]) for e_idx in range(len(self.elements))]))
        # A field that matches neither count is not integrable over this mesh;
        # falling through returned None, which only surfaced as a TypeError one
        # frame later in calculate_mean_value.
        raise ValueError(
            f'field of length {len(u)} matches neither {len(self.elements)} elements '
            f'nor {len(self.vertices)} vertices'
        )

    def calculate_mean_value(self, u: VertexField | ElementField) -> float:
        return self.calculate_total_value(u) / sum([element.volume for element in self.element_objs])

    def calculate_gradient(self, u: VertexField) -> FloatArray: # TODO: works, but need to understand 1D vs 2D use in dirichlet energy
        gradient = []
        for e_idx, element_obj in enumerate(self.element_objs):
            # u_elt = u[np.array([2*self.elements[e_idx], 2*self.elements[e_idx]+1]).T.flatten()]
            u_elt = u[self.elements[e_idx]]
            gradient.append(element_obj.grad_phi.T @ u_elt)
        return np.array(gradient)

    def calculate_dirichlet_energy(self, u: VertexField) -> float:
        u_gradient = self.calculate_gradient(u)
        squared_gradient_norm = np.einsum('ij,ij->i', u_gradient, u_gradient)
        return float(sum([self.element_objs[e_idx].volume * squared_gradient_norm[e_idx] for e_idx in range(len(self.elements))]))

    def calculate_energy(self, u: VertexField, dudt: VertexField, c: float = 1.0) -> float:
        '''Total wave energy 1/2 * (c^2 u^T K u + dudt^T M dudt).

        Two things make this the quantity Crank-Nicolson actually conserves, so
        it can be trusted as an integrator diagnostic:

        - the c^2 on the potential term, without which it is only conserved for c == 1;
        - the consistent mass matrix in the kinetic term. `calculate_total_value`
          would give the *lumped* approximation, and pairing a lumped kinetic term
          with an exact potential term makes the total swing by ~20% as energy
          sloshes between the two -- pure measurement artifact.
        '''
        potential_energy = c**2 * self.calculate_dirichlet_energy(u)
        kinetic_energy = dudt @ self.M @ dudt
        return float(0.5 * (potential_energy + kinetic_energy))

    def get_edges_in_idxs(
        self,
        vertices_idxs: IntArray | Sequence[int],
        exclude_corners: bool = False,
    ) -> list[IntArray]:
        in_edges = []
        for edge in self.edges:
            if edge[0] in vertices_idxs and edge[1] in vertices_idxs:
                if exclude_corners:
                    x1, y1 = self.vertices[edge[0]]
                    x2, y2 = self.vertices[edge[1]]
                    if (x1 - x2) != 0 and (y1 - y2) != 0:
                        continue
                in_edges.append(edge)
        return in_edges

    def get_boundary_idxs_in_rect(self, rect: tuple[float, float, float, float]) -> list[int]:
        x_min, y_min, x_max, y_max = rect
        in_boundary_idxs = []
        for v_idx in self.boundary_idxs:
            x, y = self.vertices[v_idx]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_boundary_idxs.append(v_idx)
        return in_boundary_idxs

    def with_topology(self, vertices: Vertices, elements: Elements, boundary: Elements) -> 'FEMesh':
        # type(self)(...) would fall back to the default element type, which is
        # wrong for anything but triangles -- carry ours across explicitly.
        return type(self)(vertices, elements, boundary, element_type=self.element_type)