"""Red-green refinement produces conforming meshes: every interior edge is shared by
exactly two elements and every boundary edge by one, under single-round, multi-round,
and adjacent-element refinement.
"""
import itertools
from collections import Counter

import numpy as np
import pytest

from fem.mesh.refinement import RedGreenRefiner


def _edge_counts(mesh):
    """Return a Counter mapping each sorted edge tuple to how many elements use it."""
    counts: Counter[tuple[int, int]] = Counter()
    for element in mesh.elements:
        for pair in itertools.combinations(sorted(element), 2):
            counts[pair] += 1
    return counts


def _assert_conforming(mesh):
    """Every edge must be shared by 1 (boundary) or 2 (interior) elements."""
    boundary_edges = {tuple(sorted(edge)) for edge in mesh.boundary}
    for edge, count in _edge_counts(mesh).items():
        if edge in boundary_edges:
            assert count == 1, (
                f"boundary edge {edge} appears in {count} elements, expected 1"
            )
        else:
            assert count == 2, (
                f"interior edge {edge} appears in {count} elements, expected 2"
            )


def _assert_no_orphan_vertices(mesh):
    """Every vertex must appear in at least one element."""
    used = set(mesh.elements.ravel())
    all_idxs = set(range(len(mesh.vertices)))
    orphans = all_idxs - used
    assert not orphans, f"orphan vertex indices: {orphans}"


# ---------------------------------------------------------------------------
# Single-round refinement
# ---------------------------------------------------------------------------

def test_single_element_refinement_is_conforming(make_unit_square):
    mesh = make_unit_square(4)
    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine([0])

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)


def test_adjacent_elements_refinement_is_conforming(make_unit_square):
    """Refining two elements that share an edge is the classic green-closure
    trigger — both insert a midpoint on the shared edge."""
    mesh = make_unit_square(4)

    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for e_idx, element in enumerate(mesh.elements):
        for pair in itertools.combinations(sorted(element), 2):
            edge_to_elements.setdefault(pair, []).append(e_idx)

    adjacent_pair = next(
        elems for elems in edge_to_elements.values() if len(elems) == 2
    )

    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine(adjacent_pair)

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)


def test_all_elements_refinement_is_conforming(make_unit_square):
    mesh = make_unit_square(4)
    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine(list(range(len(mesh.elements))))

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)
    # Every element splits in four; a neighbour queued for its own red split is
    # never closed green in between.
    assert len(refined.elements) == 4 * len(mesh.elements)
    assert set(refiner.leaf_classifications()) == {'red'}


# ---------------------------------------------------------------------------
# Multi-round refinement
# ---------------------------------------------------------------------------

def test_two_rounds_of_refinement_are_conforming(make_unit_square):
    """A second round exercises the green→red rollback path."""
    mesh = make_unit_square(4)
    refiner = RedGreenRefiner(mesh)

    mesh_after_1 = refiner.refine([0])
    _assert_conforming(mesh_after_1)

    mesh_after_2 = refiner.refine([0, 1])
    _assert_conforming(mesh_after_2)
    _assert_no_orphan_vertices(mesh_after_2)
    assert len(mesh_after_2.elements) > len(mesh_after_1.elements)


def test_repeated_refinement_stays_conforming(make_unit_square):
    """Four rounds of refining random-ish elements: the mesh must stay
    conforming throughout, not just after the first round."""
    mesh = make_unit_square(6)
    refiner = RedGreenRefiner(mesh)

    rng = np.random.default_rng(42)
    mesh = mesh
    for _ in range(4):
        n = len(mesh.elements)
        targets = rng.choice(n, size=min(3, n), replace=False).tolist()
        mesh = refiner.refine(targets)
        _assert_conforming(mesh)

    _assert_no_orphan_vertices(mesh)


# ---------------------------------------------------------------------------
# Boundary integrity
# ---------------------------------------------------------------------------

def test_boundary_edges_are_subset_of_mesh_edges(make_unit_square):
    """Refinement splits boundary edges; the new edges must all appear in the
    element connectivity."""
    mesh = make_unit_square(4)
    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine([0, 1, 2])

    mesh_edges = {
        tuple(sorted(pair))
        for element in refined.elements
        for pair in itertools.combinations(element, 2)
    }
    for edge in refined.boundary:
        assert tuple(sorted(edge)) in mesh_edges, (
            f"boundary edge {tuple(edge)} not found in element edges"
        )


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

def test_refinement_conserves_area(make_unit_square):
    """Red and green splits tile their parent exactly, so the mesh keeps its area whether
    one element, an adjacent pair, or every element is refined."""
    mesh = make_unit_square(4)
    for targets in ([0], [0, 1], list(range(len(mesh.elements)))):
        refined = RedGreenRefiner(mesh).refine(targets)
        assert refined.area == pytest.approx(mesh.area, rel=1e-12)


# ---------------------------------------------------------------------------
# Heavier multi-round refinement: rollbacks meeting rollbacks
# ---------------------------------------------------------------------------

def _assert_boundary_is_the_single_edges(mesh):
    """The boundary facets are exactly the edges one element uses: a hanging node
    would show up as an interior edge used once."""
    single = {edge for edge, count in _edge_counts(mesh).items() if count == 1}
    assert single == {tuple(sorted(edge)) for edge in mesh.boundary}


def test_two_elements_a_round_on_a_small_square_stays_conforming(make_unit_square):
    """Three rounds of two elements each on a 4x4 square, over 200 seeds: enough for
    a round to touch a green pair whose parent's neighbour is itself green, the case
    the per-triangle tree left a hanging node in (seed 32, its second round)."""
    for seed in range(200):
        rng = np.random.default_rng(seed)
        mesh = make_unit_square(4)
        refiner = RedGreenRefiner(mesh)
        for _ in range(3):
            targets = rng.choice(len(mesh.elements), size=2, replace=False).tolist()
            mesh = refiner.refine(targets)
            _assert_conforming(mesh)
            _assert_boundary_is_the_single_edges(mesh)


def test_refining_a_fifth_of_the_mesh_each_round_stays_conforming():
    """Six rounds, a fifth of the elements each, on a rectangle: many rollbacks per
    round, some of them adjacent. Conforming, boundary exact, area kept, no orphan."""
    from fem.mesh.structured import box_mesh

    source = box_mesh([[0, 0], [2, 1]], (13, 7))
    refiner = RedGreenRefiner(source)
    rng = np.random.default_rng(1)
    mesh = source
    for _ in range(6):
        n = len(mesh.elements)
        mesh = refiner.refine(rng.choice(n, size=n // 5, replace=False).tolist())
        _assert_conforming(mesh)
        _assert_boundary_is_the_single_edges(mesh)
        assert len(refiner.leaf_classifications()) == len(mesh.elements)
    _assert_no_orphan_vertices(mesh)
    assert mesh.area == pytest.approx(source.area, rel=1e-12)


def test_boundary_tags_follow_their_facets_through_rounds():
    """A tag names the side a facet came from; through rounds of splitting and
    rollback every facet still carries the tag of the side it lies on."""
    from fem.mesh.mesh import Mesh
    from fem.mesh.structured import box_mesh

    base = box_mesh([[0, 0], [1, 1]], (4, 4))
    midpoints = base.vertices[base.boundary].mean(axis=1)
    sides = np.select(
        [np.isclose(midpoints[:, 1], 0), np.isclose(midpoints[:, 0], 1),
         np.isclose(midpoints[:, 1], 1)], [0, 1, 2], default=3)
    mesh = Mesh(base.vertices, base.elements, base.boundary, boundary_tags=sides)
    refiner = RedGreenRefiner(mesh)
    rng = np.random.default_rng(3)
    for _ in range(4):
        mesh = refiner.refine(rng.choice(len(mesh.elements), size=3, replace=False).tolist())
        assert mesh.boundary_tags is not None
        assert len(mesh.boundary_tags) == len(mesh.boundary)
        mid = mesh.vertices[mesh.boundary].mean(axis=1)
        on_side = [np.isclose(mid[:, 1], 0), np.isclose(mid[:, 0], 1),
                   np.isclose(mid[:, 1], 1), np.isclose(mid[:, 0], 0)]
        for tag, mask in enumerate(on_side):
            assert np.all(mesh.boundary_tags[mask] == tag)
