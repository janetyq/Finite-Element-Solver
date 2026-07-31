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
    calculate_segment_angles,
    calculate_triangle_angles,
    calculate_triangle_min_angle,
    calculate_circumcenter,
    get_boundary_from_vertices_elements,
    point_in_polygon,
)

logger = logging.getLogger(__name__)

# The sharpest corner -- two input segments meeting at a shared vertex -- for which
# Ruppert's is proven to finish. Refinement fixes one problem at a time (a segment with
# a vertex inside its diametral circle, or a triangle failing the angle or area test) by
# inserting a point, and stops once none are left. Below 60 degrees two segments sharing
# a corner would each re-create the other's problem, so such a corner is meshed at the
# angle it comes in at rather than refined towards a bound it cannot reach: see
# `_split_point` and `_spans_a_sharp_corner`.
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
    more (`SAFE_INPUT_ANGLE`).  Segments meeting below it are the one case where
    `min_angle` does not hold: the triangle across such a corner is meshed at the
    input's own angle, since no refinement improves it.  Everything away from
    those corners still meets the bound.

    Cost grows steeply in the number of input points: each step rebuilds the
    full Delaunay triangulation, and more input points means more steps on a
    larger point set.  Simplify a densely sampled outline
    (`fem.mesh.svg.douglas_peucker`) before handing it over.
    '''

    def __init__(self, pslg, min_angle=30, max_area=None):
        '''Refine `pslg` until every triangle clears both bounds.

        `min_angle` is in degrees: the smallest interior angle any output triangle
        may have, bar the triangles across input corners already sharper than
        `SAFE_INPUT_ANGLE`, which no refinement can improve. Ruppert's proof covers
        bounds up to about 20.7 degrees and it holds in practice to roughly 30;
        above that refinement can fail to terminate however blunt the input.

        `max_area` is an absolute area, not a fraction of the region -- callers
        wanting a fraction scale it themselves. None leaves element size
        unbounded, so a large region comes back as a handful of big triangles.
        '''
        self.vertices = np.array(pslg.vertices)
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.segment_loops = np.array(getattr(pslg, 'loop_ids',
                                              np.zeros(len(self.segments), dtype=int)))
        self.triangulation = Delaunay(self.vertices)
        self.min_angle = min_angle
        self.max_area = max_area
        # Which loop each boundary facet of the returned mesh came from; set by refine().
        self.boundary_loops = np.zeros(0, dtype=int)

        corner_angles = calculate_segment_angles(self.vertices, self.segments)
        self.input_angle = min(corner_angles.values(), default=180.0)
        # Corners the angle bound cannot be met at. Their segments split on shells
        # rather than at midpoints, and the triangle across them is accepted as it
        # comes -- otherwise refinement chases them forever.
        self.sharp_vertices = {v for v, angle in corner_angles.items()
                               if angle < SAFE_INPUT_ANGLE}
        if self.sharp_vertices:
            logger.warning(
                'segments meet at %.1f degrees somewhere, below the %.0f Delaunay '
                'refinement can hold; the %d corner(s) under it keep their own angle '
                'in the mesh, and cost extra elements around them',
                self.input_angle, SAFE_INPUT_ANGLE, len(self.sharp_vertices),
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
        segment_mask = self._segment_edges()
        bad = calculate_triangle_min_angle(self.vertices[simplices]) < self.min_angle
        if self.sharp_vertices:
            bad &= ~self._spans_a_sharp_corner(segment_mask)
        # The area cap survives that exemption: a corner triangle cannot be made
        # less sharp, but it can be made smaller.
        if self.max_area is not None:
            bad |= self.get_triangle_areas() > self.max_area
        # Regions are only meaningful once no segment is encroached, which the
        # refinement loop always resolves first.
        bad &= ~self.get_exterior_triangles(segment_mask)
        return list(simplices[bad])

    def _spans_a_sharp_corner(self, segment_mask):
        '''Triangles whose smallest angle is one the input already has.

        Where two segments meet below `SAFE_INPUT_ANGLE`, the triangle between
        them holds their angle however small the elements around it get, so
        refining it never ends. Taking it as it stands is what lets a sharp
        outline mesh at all, at the price of the bound holding everywhere else.
        '''
        simplices = self.triangulation.simplices
        corner = np.argmin(calculate_triangle_angles(self.vertices[simplices]), axis=-1)
        rows = np.arange(len(simplices))
        # The two edges meeting at a vertex are those opposite the other two.
        between_segments = (segment_mask[rows, (corner + 1) % 3]
                            & segment_mask[rows, (corner + 2) % 3])
        return between_segments & np.isin(simplices[rows, corner], list(self.sharp_vertices))

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

    def get_exterior_triangles(self, segment_mask=None):
        '''Bool mask: True for triangles outside the PSLG boundary.

        Picks one triangle per region, counts how many segments a ray from
        its centroid crosses to reach infinity, and applies the even-odd rule:
        odd crossings = inside, even = outside.  A loop inside another is
        therefore a hole without anyone having to declare it.
        '''
        labels = self.get_regions(segment_mask)
        # A centroid is safely interior to its own triangle, and segments are
        # edges here, so it can never land ambiguously on top of one.
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

    def _split_point(self, segment):
        '''Where to cut `segment` in two: its midpoint, unless it runs from a
        corner too sharp to mesh, in which case a point on a shell around that
        corner.

        Midpoints do not converge there. Splitting one segment drops a vertex
        inside its neighbour's diametral circle, splitting the neighbour drops
        one back inside the first's, and the two walk into the corner without
        ever clearing each other. Cutting at a power-of-two distance from the
        corner instead puts both splits on the same ladder of radii, and once
        two of them land on one shell they are equidistant from the corner and
        stop encroaching -- the cascade ends after a bounded number of rounds.
        '''
        start, end = self.vertices[segment[0]], self.vertices[segment[1]]
        if int(segment[0]) in self.sharp_vertices:
            corner, far = start, end
        elif int(segment[1]) in self.sharp_vertices:
            corner, far = end, start
        else:
            return 0.5 * (start + end)

        length = float(np.linalg.norm(far - corner))
        # The one power of two in [length/3, 2*length/3], so the cut stays near
        # the middle while the radius comes off a ladder shared with every other
        # segment at this corner.
        radius = 2.0 ** np.floor(np.log2(2 * length / 3))
        return corner + (far - corner) * (radius / length)

    def split_segment(self, segment):
        new_vertex_idx = len(self.vertices)
        new_segments = [[segment[0], new_vertex_idx], [segment[1], new_vertex_idx]]
        # Halves inherit the loop, so a boundary facet can still be traced back to
        # the outline it came from however many times it has been split.
        loop_id = self.del_segment(segment)
        self.add_vertex(self._split_point(segment))
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
