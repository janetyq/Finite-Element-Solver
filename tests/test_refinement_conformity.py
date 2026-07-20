"""Tests that red-green refinement produces conforming meshes.

A conforming triangle mesh has no hanging nodes: every interior edge is shared
by exactly two elements, and every boundary edge by exactly one.  These tests
pin that invariant under single-round, multi-round, and adjacent-element
refinement — the scenarios most likely to break green-closure bookkeeping.
"""
import itertools
from collections import Counter

import numpy as np

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
    femesh = make_unit_square(4)
    refiner = RedGreenRefiner(femesh)
    refined = refiner.refine([0])

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)


def test_adjacent_elements_refinement_is_conforming(make_unit_square):
    """Refining two elements that share an edge is the classic green-closure
    trigger — both insert a midpoint on the shared edge."""
    femesh = make_unit_square(4)

    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for e_idx, element in enumerate(femesh.elements):
        for pair in itertools.combinations(sorted(element), 2):
            edge_to_elements.setdefault(pair, []).append(e_idx)

    adjacent_pair = next(
        elems for elems in edge_to_elements.values() if len(elems) == 2
    )

    refiner = RedGreenRefiner(femesh)
    refined = refiner.refine(adjacent_pair)

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)


def test_all_elements_refinement_is_conforming(make_unit_square):
    femesh = make_unit_square(4)
    refiner = RedGreenRefiner(femesh)
    refined = refiner.refine(list(range(len(femesh.elements))))

    _assert_conforming(refined)
    _assert_no_orphan_vertices(refined)


# ---------------------------------------------------------------------------
# Multi-round refinement
# ---------------------------------------------------------------------------

def test_two_rounds_of_refinement_are_conforming(make_unit_square):
    """A second round exercises the green→red rollback path."""
    femesh = make_unit_square(4)
    refiner = RedGreenRefiner(femesh)

    mesh_after_1 = refiner.refine([0])
    _assert_conforming(mesh_after_1)

    mesh_after_2 = refiner.refine([0, 1])
    _assert_conforming(mesh_after_2)
    _assert_no_orphan_vertices(mesh_after_2)
    assert len(mesh_after_2.elements) > len(mesh_after_1.elements)


def test_repeated_refinement_stays_conforming(make_unit_square):
    """Four rounds of refining random-ish elements: the mesh must stay
    conforming throughout, not just after the first round."""
    femesh = make_unit_square(6)
    refiner = RedGreenRefiner(femesh)

    rng = np.random.default_rng(42)
    mesh = femesh
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
    femesh = make_unit_square(4)
    refiner = RedGreenRefiner(femesh)
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
