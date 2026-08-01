import logging
from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, KDTree

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray
from fem.geometry import (
    calculate_polygon_area,
    calculate_triangle_min_angle,
    calculate_circumcenter,
    get_boundary_from_vertices_elements,
    point_in_polygon,
)

logger = logging.getLogger(__name__)

# A segment's own endpoints sit exactly on its diametral circle, so floating-point
# noise can push them fractionally inside.  This relative tolerance shrinks the test
# circle slightly to prevent a segment from appearing encroached by its own endpoint
# (which would split it forever).
ENCROACHMENT_TOLERANCE = 1e-12


class RuppertsAlgorithm:
    '''Ruppert's Delaunay refinement of a PSLG.

    Starting from a Delaunay triangulation of the input vertices, loop until
    no encroached segments and no skinny triangles remain:

    1. **Split encroached segments** (priority).  A segment is *encroached*
       when a mesh vertex other than its own endpoints falls inside its
       diametral circle (the circle whose diameter is the segment).  Replace
       it with two halves at its midpoint.  Fixing encroachment can also fix
       nearby skinny triangles.

    2. **Insert circumcenters of skinny triangles.**  A triangle whose
       smallest angle is below `min_angle` is refined by inserting its
       circumcenter.  If that new point would encroach a segment, split the
       segment instead and re-examine the triangle later.

    Each step adds one vertex and rebuilds the Delaunay triangulation.

    **Why segments are respected.**  The Delaunay triangulation only knows
    about points, not segments.  But if a segment's diametral circle contains
    no other vertex, it is guaranteed to appear as a Delaunay edge.  Splitting
    encroached segments clears their diametral circles, which is what makes
    the unconstrained Delaunay conform to the boundary.

    **What is returned.**  The result keeps only what the PSLG encloses — a
    Delaunay triangulation spans the convex hull, so a non-convex outline also
    produces triangles outside it.  Segments are walls: what survives is
    whatever cannot be reached from infinity without crossing one.

    Ruppert proved termination for inputs whose segments meet at 60 degrees or
    more.  Corners sharper than that (`SAFE_INPUT_ANGLE`) are exempt: the
    triangle across them is accepted at whatever angle it comes in at, since
    no refinement can improve it.

    Cost grows steeply in the number of input points: each step rebuilds the
    full Delaunay triangulation, and more input points means more steps on a
    larger point set.  Simplify a densely sampled outline
    (`fem.mesh.svg.douglas_peucker`) before handing it over.
    '''

    def __init__(self, pslg, min_angle=30, max_area=None):
        self.vertices = np.array(pslg.vertices)
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.triangulation = Delaunay(self.vertices)
        self.min_angle = min_angle
        self.max_area = max_area

    def _diametral_circles(self):
        '''Centres and squared radii of every segment's diametral circle.'''
        ends = self.vertices[self.segments]
        centers = ends.mean(axis=1)
        radii_sq = np.sum((ends[:, 1] - ends[:, 0])**2, axis=-1) / 4
        return centers, radii_sq

    def get_encroached_segments(self):
        '''Segments with a mesh vertex strictly inside their diametral circle.

        Testing the nearest vertex to each centre suffices: the segment's own
        endpoints lie on its circle, so anything nearer is strictly inside.
        '''
        centers, radii_sq = self._diametral_circles()
        distances, _ = KDTree(self.vertices).query(centers)
        return list(self.segments[distances**2 < radii_sq * (1 - ENCROACHMENT_TOLERANCE)])

    def get_segments_encroached_by(self, vertex):
        '''Segments whose diametral circle would strictly contain `vertex`.'''
        centers, radii_sq = self._diametral_circles()
        offsets = np.asarray(vertex) - centers
        inside = np.sum(offsets**2, axis=-1) < radii_sq * (1 - ENCROACHMENT_TOLERANCE)
        return list(self.segments[inside])

    def get_triangle_areas(self):
        corners = self.vertices[self.triangulation.simplices]
        edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        return 0.5 * np.abs(edge_a[:, 0]*edge_b[:, 1] - edge_a[:, 1]*edge_b[:, 0])

    def get_bad_triangles(self):
        '''Interior triangles that violate the angle bound or area cap.

        Exterior triangles are excluded — refining them would insert
        circumcenters that enlarge the convex hull, creating more exterior
        triangles and never terminating.
        '''
        simplices = self.triangulation.simplices
        bad = calculate_triangle_min_angle(self.vertices[simplices]) < self.min_angle
        if self.max_area is not None:
            bad |= self.get_triangle_areas() > self.max_area
        # Regions are only meaningful once no segment is encroached, which the
        # refinement loop always resolves first.
        bad &= ~self.get_exterior_triangles()
        return list(simplices[bad])

    def _segment_edges(self):
        '''(n_tri, 3) bool mask: which of each triangle's edges is a PSLG segment.

        Column `j` corresponds to the edge opposite vertex `j`, matching the
        layout of `Delaunay.neighbors` — so `neighbors[i, j]` is the triangle
        across a segment when `_segment_edges()[i, j]` is True.
        '''
        simplices = self.triangulation.simplices
        opposite = np.stack([
            np.sort(simplices[:, [1, 2]], axis=1),
            np.sort(simplices[:, [2, 0]], axis=1),
            np.sort(simplices[:, [0, 1]], axis=1),
        ], axis=1)
        # Each vertex pair packed into one integer, so the lookup is a single isin.
        stride = len(self.vertices)
        segments = np.sort(self.segments, axis=1)
        return np.isin(opposite[..., 0]*stride + opposite[..., 1],
                       segments[:, 0]*stride + segments[:, 1])

    def get_regions(self, segment_mask=None):
        '''Label each triangle with an integer region ID.

        Two triangles are in the same region if they can reach each other
        through shared edges without crossing a segment.  Implemented as
        connected components of the triangle adjacency graph with segment
        edges removed.
        '''
        neighbors = self.triangulation.neighbors
        if segment_mask is None:
            segment_mask = self._segment_edges()
        interior_edge = (neighbors != -1) & ~segment_mask
        triangle, edge = np.nonzero(interior_edge)
        dual = coo_matrix(
            (np.ones(len(triangle), dtype=bool), (triangle, neighbors[triangle, edge])),
            shape=(len(neighbors), len(neighbors)),
        )
        _, labels = connected_components(dual, directed=False)
        return labels

    def get_exterior_triangles(self):
        '''Bool mask: True for triangles outside the PSLG boundary.

        A triangle on the convex hull whose hull edge is not a segment can be
        reached from infinity — it is exterior, and so is every triangle in its
        region.  A hull edge that *is* a segment walls the interior off, which
        is what keeps a convex outline from being discarded entirely.
        '''
        segment_mask = self._segment_edges()
        labels = self.get_regions(segment_mask)
        reaches_infinity = ((self.triangulation.neighbors == -1) & ~segment_mask).any(axis=1)
        return np.isin(labels, np.unique(labels[reaches_infinity]))

    def _enclosed_mesh(self):
        '''The enclosed triangles as a Mesh, renumbered onto the vertices it uses.'''
        elements = self.triangulation.simplices[~self.get_exterior_triangles()]
        if len(elements) == 0:
            raise ValueError(
                'the PSLG encloses no region: every triangle can be reached from '
                'outside without crossing a segment, so there is nothing to mesh'
            )

        used = np.unique(elements)
        renumbered = np.zeros(len(self.vertices), dtype=np.intp)
        renumbered[used] = np.arange(len(used))
        elements = renumbered[elements]
        return Mesh(self.vertices[used], elements,
                    get_boundary_from_vertices_elements(elements))

    def refine(self):
        encroached_segments = self.get_encroached_segments()

        while True:
            new_encroached_segments = []

            # check if there are any encroached segments and split them
            if len(encroached_segments) > 0:
                segment = encroached_segments.pop()
                self.split_segment(segment)
            else:
                # Only asked for once the segments are clear, since that branch
                # outranks this one and the answer costs a region labelling.
                bad_triangles = self.get_bad_triangles()
                # if no encroached segments or bad triangles, we are done
                if len(bad_triangles) == 0:
                    break
                triangle = bad_triangles.pop()
                circumcenter = calculate_circumcenter(self.vertices[triangle])
                # Inserting a point inside a segment's diametral circle would cut
                # the mesh off from the outline, so split those segments instead
                # and leave the triangle to be reconsidered once they are gone.
                new_encroached_segments = self.get_segments_encroached_by(circumcenter)
                if not new_encroached_segments:
                    self.add_vertex(circumcenter)
            self.triangulation = Delaunay(self.vertices)
            encroached_segments = self.get_encroached_segments() + new_encroached_segments

        logger.debug('refined to %d triangles over %d vertices',
                     len(self.triangulation.simplices), len(self.vertices))
        return self._enclosed_mesh()

    def del_segment(self, segment):
        segment_idx = np.where((self.segments == segment).all(axis=1))[0][0]
        self.segments = np.delete(self.segments, segment_idx, axis=0)
    
    def add_vertex(self, vertex):
        self.vertices = np.append(self.vertices, [vertex], axis=0)
    
    def add_segment(self, segment):
        self.segments = np.append(self.segments, [segment], axis=0)

    def split_segment(self, segment):
        midpoint = 0.5 * (self.vertices[segment[0]] + self.vertices[segment[1]])
        new_vertex_idx = len(self.vertices)
        new_segments = [[segment[0], new_vertex_idx], [segment[1], new_vertex_idx]]
        self.del_segment(segment)
        self.add_vertex(midpoint)
        self.add_segment(new_segments[0])
        self.add_segment(new_segments[1])
        return new_segments

# Simple meshing functions
def create_rect_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured triangulation of the axis-aligned rectangle spanned by
    `corners` ((x0, y0), (x1, y1)), with `resolution` (nx, ny) nodes per axis.'''
    x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
    y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*resolution[0] + i

    for i in range(resolution[0]-1):
        for j in range(resolution[1]-1):
            if (i + j) % 2 == 0:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            else:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i, j+1)])
                elements.append([get_index(i+1, j), get_index(i+1, j+1), get_index(i, j+1)])

    boundary = get_boundary_from_vertices_elements(elements)
    mesh = Mesh(vertices, elements, boundary)

    return mesh

def create_box_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured tetrahedralization of the axis-aligned box spanned by
    `corners` ((x0, y0, z0), (x1, y1, z1)), with `resolution` (nx, ny, nz) nodes
    per axis.

    Each grid cell is split into six tets by Kuhn's decomposition: every tet
    contains the cell's main diagonal (corner 000 to 111), one per permutation
    of the three axes. Because every cell splits along the *same* diagonal
    direction, neighbouring cells agree on their shared faces and the mesh is
    conforming -- the property that makes this usable for convergence studies,
    where an unstructured generator would confound the refinement rate.
    '''
    nx, ny, nz = resolution
    x_range = np.linspace(corners[0][0], corners[1][0], nx)
    y_range = np.linspace(corners[0][1], corners[1][1], ny)
    z_range = np.linspace(corners[0][2], corners[1][2], nz)

    vertices = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])

    def get_index(i, j, k):
        return (i*ny + j)*nz + k

    # Corners of a cell, indexed by the bits of (di, dj, dk) -- corner 5 is
    # (1, 0, 1). The six tets below are written against that numbering.
    KUHN_TETS = [
        (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
        (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
    ]

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [
                    get_index(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1))
                    for c in range(8)
                ]
                elements.extend([[corner[c] for c in tet] for tet in KUHN_TETS])

    boundary = get_boundary_from_vertices_elements(elements)
    return Mesh(vertices, elements, boundary)


def create_approx_mesh(outline: FloatArray, approx_triangles: int = 100) -> Mesh:
    '''A triangulation of the polygon `outline` ((n_points, 2) vertices, in
    order) with roughly `approx_triangles` elements.'''
    dx = np.sqrt(2 * calculate_polygon_area(outline) / approx_triangles)
    x_min, x_max = np.min(outline[:, 0]), np.max(outline[:, 0])
    y_min, y_max = np.min(outline[:, 1]), np.max(outline[:, 1])
    x_range = np.arange(x_min, x_max, dx)
    y_range = np.arange(y_min, y_max, dx)
    x_range += (x_max - x_range[-1])/2
    y_range += (y_max - y_range[-1])/2

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*len(x_range) + i

    # first mesh everything
    for i, x in enumerate(x_range[:-1]):
        for j, y in enumerate(y_range[:-1]):
            elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
            elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            
    # second remove elements with centers outside of outline
    removed_elements = []
    for element in elements:
        center = np.mean(vertices[element], axis=0)
        offcenters = [(center + vertices[i])/2 for i in element]
        for offcenter in offcenters:
            if not point_in_polygon(offcenter, outline):
                removed_elements.append(element)
                break
    for element in removed_elements:
        elements.remove(element)

    # remove unnecessary vertices
    used_v_idxs = np.unique(np.array(elements).flatten())
    # map old indices to new indices
    v_idx_map = {old: new for new, old in enumerate(used_v_idxs)}
    vertices = vertices[used_v_idxs]
    elements = [[v_idx_map[e_idx] for e_idx in element] for element in elements]
    boundary = get_boundary_from_vertices_elements(elements)
    mesh = Mesh(vertices, elements, boundary)

    return mesh
