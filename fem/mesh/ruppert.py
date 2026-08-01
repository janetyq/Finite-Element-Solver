import logging
from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, KDTree

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray
from fem.geometry import (
    calculate_minimum_segment_angle,
    calculate_polygon_area,
    calculate_triangle_min_angle,
    calculate_circumcenter,
    get_boundary_from_vertices_elements,
    point_in_polygon,
)

logger = logging.getLogger(__name__)

# Refinement terminates when it runs out of work: no segment encroached, and no
# triangle under the angle bound or over the area cap. Ruppert proved that state is
# always reached when the input's segments meet at 60 degrees or more. Below that,
# splitting a segment near the corner drops a vertex inside its neighbour's diametral
# circle, which forces that one to split and encroaches the first again -- the pair
# cascades into the corner and refinement need never stop. Cost climbs steeply well
# before it diverges, so this reads as a hang rather than as a bad input. Construction
# warns; the refinement loop itself does not treat sharp corners specially.
SAFE_INPUT_ANGLE = 60.0

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
    produces triangles outside it.  Interior/exterior alternates by the
    even-odd rule: a region with an odd number of segment crossings to
    infinity is inside, so a loop inside another is a hole.  After `refine`,
    `boundary_loops` records which input loop each boundary facet came from.

    Ruppert proved termination for inputs whose segments meet at 60 degrees or
    more (`SAFE_INPUT_ANGLE`).  Construction warns when the input is sharper
    than that, but refinement does not treat such corners specially, so one can
    still cascade into an unbounded number of elements.

    Cost grows steeply in the number of input points: each step rebuilds the
    full Delaunay triangulation, and more input points means more steps on a
    larger point set.  Simplify a densely sampled outline
    (`fem.mesh.svg.douglas_peucker`) before handing it over.
    '''

    def __init__(self, pslg, min_angle=30, max_area=None):
        self.vertices = np.array(pslg.vertices)
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.segment_loops = np.array(getattr(pslg, 'loop_ids',
                                              np.zeros(len(self.segments), dtype=int)))
        self.triangulation = Delaunay(self.vertices)
        self.min_angle = min_angle
        self.max_area = max_area
        # Which loop each boundary facet of the returned mesh came from; set by refine().
        self.boundary_loops = np.zeros(0, dtype=int)

        self.input_angle = calculate_minimum_segment_angle(self.vertices, self.segments)
        if self.input_angle < SAFE_INPUT_ANGLE:
            logger.warning(
                'segments meet at %.1f degrees somewhere, below the %.0f that Delaunay '
                'refinement needs to be sure of terminating; expect this to cost far '
                'more elements than the outline suggests, or not to finish',
                self.input_angle, SAFE_INPUT_ANGLE,
            )

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

    def _crossing_counts(self, points):
        '''How many segments a ray from each point to +x passes through.'''
        starts = self.vertices[self.segments[:, 0]][None, :, :]
        ends = self.vertices[self.segments[:, 1]][None, :, :]
        points = points[:, None, :]

        rises = (starts[..., 1] > points[..., 1]) != (ends[..., 1] > points[..., 1])
        # Only the segments the ray's height crosses can be hit; the rest would
        # divide by a zero rise, so keep them out of the arithmetic entirely.
        height = np.where(rises, ends[..., 1] - starts[..., 1], 1.0)
        crossing_x = starts[..., 0] + (points[..., 1] - starts[..., 1]) * (
            ends[..., 0] - starts[..., 0]) / height
        return (rises & (crossing_x > points[..., 0])).sum(axis=1)

    def get_exterior_triangles(self):
        '''Bool mask: True for triangles outside the PSLG boundary.

        Picks one triangle per region, counts how many segments a ray from
        its centroid crosses to reach infinity, and applies the even-odd rule:
        odd crossings = inside, even = outside.  A loop inside another is
        therefore a hole without anyone having to declare it.
        '''
        labels = self.get_regions(self._segment_edges())
        representatives = np.unique(labels, return_index=True)[1]
        corners = self.vertices[self.triangulation.simplices[representatives]]
        outside = self._crossing_counts(corners.mean(axis=1)) % 2 == 0
        return outside[labels]

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

        boundary = get_boundary_from_vertices_elements(elements)
        self.boundary_loops = self._trace_boundary_to_loops(boundary, used, renumbered)
        return Mesh(self.vertices[used], elements, boundary)

    def _trace_boundary_to_loops(self, boundary, used, renumbered):
        '''The input loop each boundary facet came from, or -1 if none did.

        This is what tells an obstacle's rim from the outer wall around it, and
        it cannot be recovered from the finished mesh -- a boundary is just
        edges by then.
        '''
        is_used = np.zeros(len(self.vertices), dtype=bool)
        is_used[used] = True
        loop_of_edge = {}
        for (start, end), loop_id in zip(self.segments, self.segment_loops):
            if is_used[start] and is_used[end]:
                loop_of_edge[tuple(sorted((renumbered[start], renumbered[end])))] = int(loop_id)
        return np.array([loop_of_edge.get(tuple(sorted(facet)), -1) for facet in boundary],
                        dtype=int)

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
        '''Remove `segment`, returning the loop it belonged to.'''
        segment_idx = np.where((self.segments == segment).all(axis=1))[0][0]
        loop_id = int(self.segment_loops[segment_idx])
        self.segments = np.delete(self.segments, segment_idx, axis=0)
        self.segment_loops = np.delete(self.segment_loops, segment_idx)
        return loop_id

    def add_vertex(self, vertex):
        self.vertices = np.append(self.vertices, [vertex], axis=0)

    def add_segment(self, segment, loop_id=0):
        self.segments = np.append(self.segments, [segment], axis=0)
        self.segment_loops = np.append(self.segment_loops, loop_id)

    def split_segment(self, segment):
        midpoint = 0.5 * (self.vertices[segment[0]] + self.vertices[segment[1]])
        new_vertex_idx = len(self.vertices)
        new_segments = [[segment[0], new_vertex_idx], [segment[1], new_vertex_idx]]
        # Halves inherit the loop, so a boundary facet can still be traced back to
        # the outline it came from however many times it has been split.
        loop_id = self.del_segment(segment)
        self.add_vertex(midpoint)
        self.add_segment(new_segments[0], loop_id)
        self.add_segment(new_segments[1], loop_id)
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
