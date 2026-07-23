# Next Steps — the two open scaling efforts

Design notes for the two pieces of work that follow batched assembly. `BACKLOG.md`
remains the index of *what* is open; this file is the *how* for these two, written
while the measurements that motivate them were still fresh. When an effort lands,
delete its section here and its row in `BACKLOG.md` in the same commit.

Both are performance work, and they are independent — neither blocks the other.
They differ in character: the first is a design change to the solve path with a
genuine fork in it, the second is a contained fix to a function that cannot
survive the mesh sizes the solver now handles.

---

## 1. Iterative solvers + preconditioning

### Why now

Sparse matrices moved the cost off the solve and onto assembly; batching assembly
moved it back. `examples/benchmark_assembly.py`, 3D linear elasticity on a unit
cube:

| n | tets | dofs | assemble | factor+solve |
|---|---|---|---|---|
| 13 | 10,368 | 6,591 | 0.18s | 0.31s |
| 17 | 24,576 | 14,739 | 0.68s | 2.39s |
| 21 | 48,000 | 27,783 | ~0.9s | ~13s |

The solve is now 3–14× the assembly and growing faster. This is not a slow code
path — it is fill-in. A direct factorization of a 3D tet-mesh operator produces
far more nonzeros than the operator itself, and the penalty compounds with
refinement. 27,783 dofs is not a large problem; the method is what caps it.

### The design fork: this is a strategy, not a replacement

`DiscreteSystem` exists to factor once and solve many times. Its four consumers
split cleanly on whether that is the right trade:

| Consumer | Pattern | Wants |
|---|---|---|
| `solver.py:271` (heat) | one LHS, factored once, reused every timestep | **direct** |
| `solver.py:331` (wave) | same, plus a non-symmetric block LHS | **direct** |
| `solver.py:231` (Poisson / elasticity) | one solve, then the factorization is discarded | **iterative** |
| `energy_solver.py:125` (Newton) | a *new* tangent every iteration, each used once | **iterative** |

Factoring a matrix to solve against it once is the worst case for a direct
method, and it is exactly what the one-shot and Newton paths do. Reusing one
factorization across hundreds of timesteps is the best case, and it is exactly
what the time-steppers do.

So the change is a selectable strategy inside `DiscreteSystem`, keeping `splu` as
the default for the reuse-heavy callers. Replacing `splu` outright would speed up
two callers and regress the other two.

### Preconditioner: three options

CG's iteration count depends on the conditioning of the operator, which for
elasticity degrades as the mesh refines. The preconditioner is what decides
whether this is a win or a wash.

- **Jacobi** (diagonal scaling). No dependency, ten lines. Weak on 3D elasticity
  specifically — the near-null space of rigid body modes is what it fails to
  capture — so iteration counts stay high and grow with refinement.
- **`scipy.sparse.linalg.spilu`** (incomplete LU). Already available, no new
  dependency, meaningfully better than Jacobi. Its fill-in is tunable but its
  behavior under refinement is still not mesh-independent.
- **AMG via `pyamg`**. A new dependency, and the right tool: algebraic multigrid
  is close to mesh-independent on exactly these operators, so iteration count
  stays roughly flat as h shrinks. `pyamg` has a `smoothed_aggregation_solver`
  that takes the rigid body modes as near-null-space vectors, which is the part
  that makes elasticity work rather than merely run.

**Recommendation: AMG.** The whole point is to stop the cost growing with
refinement, and AMG is the only one of the three that delivers that. One
well-maintained dependency is cheap next to hand-rolling a multigrid hierarchy,
and `AGENTS.md` is explicit that development cost should not drive this kind of
decision.

### Correctness constraints — the parts that need care

**CG requires symmetric positive definite.** That holds for the Poisson stiffness,
the elastic stiffness, and the mass matrix. It does **not** hold for the wave
solver's block LHS (non-symmetric by construction), and it holds for the Newton
tangent only near a minimum — far from one it can be indefinite, which is why
`fem/energy_solver.py` conditions it. The strategy must therefore be chosen
explicitly by the caller or guarded, never inferred silently from the matrix.
Handing an indefinite matrix to CG produces a wrong answer rather than an error.

**An iterative solve is approximate, and the MMS tests are the guard.** The
convergence tests measure discretization error at ~4e-3 (3D) and tighter in 2D.
If the solver tolerance is not comfortably below that, the iterative error
pollutes the measurement and the observed order degrades — which would look like
a discretization bug. Set `rtol` at least two orders below the finest expected
discretization error, and treat any movement in the observed orders as a
tolerance problem until proven otherwise.

**`DiscreteSystem` currently factors eagerly in `__init__`.** An iterative
strategy has no factorization; what it builds once and reuses is the
preconditioner. The class's contract survives that, but the naming (`_lu`) does
not.

### Verification

- `tests/test_convergence.py` and `tests/test_convergence_elasticity.py` must stay
  green with unchanged observed orders — the 3D test now asserts a real O(h²)
  band, so it is a sharp instrument for this.
- `tests/test_system.py` covers the direct path; the iterative path needs the
  parallel cases, including a guard test that a non-SPD matrix is refused rather
  than silently mis-solved.
- Re-run `examples/benchmark_assembly.py` and record the new split.

---

## 2. Sparsify the topology-optimization smoothing matrix

### Why now

`fem/numerics.py:calculate_smoothing_matrix` materializes a dense
element × element distance matrix:

```python
diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
distances = np.linalg.norm(diff, axis=2)
weight_matrix = np.maximum(0, r - distances)
```

That is O(n_elements²) in both time and memory:

| elements | dense matrix |
|---|---|
| 3,200 | 82 MB |
| 10,000 | 800 MB |
| 48,000 | 18 GB |

Batched assembly is what makes this urgent rather than merely untidy. Mesh sizes
that the solver now handles comfortably are ones this function cannot allocate,
so topology optimization is hard-capped at a few thousand elements by a support
routine rather than by the physics or the solve.

### The fix

The weights are compactly supported — `max(0, r - distance)` is exactly zero
beyond radius `r` — so the dense matrix is mostly structural zeros. A spatial
index gives the same matrix without ever forming them:

```python
centers = mesh.vertices[mesh.elements].mean(axis=1)
tree = cKDTree(centers)
pairs = tree.query_ball_point(centers, r)
# rows/cols from pairs, weights from max(0, r - |c_i - c_j|), then row-normalize
```

**This is an exact refactor, not an approximation.** The entries being dropped are
identically zero, and the row-normalization denominator sums the same values
either way. Verified on a 2D mesh:

| elements | dense | sparse | max abs difference |
|---|---|---|---|
| 800 | 0.195s, 5 MB | 0.061s, 0.16 MB | 5.6e-17 |
| 3,200 | 1.770s, 82 MB | 0.336s, 2.5 MB | 2.8e-17 |

32× smaller and 5× faster at 3,200 elements, agreeing to floating-point rounding.
The gap widens with element count, since the dense version grows quadratically
while the sparse one grows linearly at fixed `r`.

### Details worth preserving

- **Self-inclusion.** An element is its own neighbour at distance 0, weight `r`.
  `query_ball_point` includes it, so this carries over — but it means the
  normalization denominator is always at least `r`, and the existing `+ 1e-16`
  guard against a zero row is defensive rather than load-bearing. Keep it anyway;
  it costs nothing and the invariant is not obvious.
- **Return type.** `TopologyOptimizer` uses the result only as
  `self.smoothing_matrix @ gradient` (`fem/topology.py:115`), so a CSR array is a
  drop-in. Nothing indexes it densely.
- **Radius semantics.** `query_ball_point` is inclusive at exactly `r`, where the
  weight is 0. Harmless, and it matches the dense version's behavior.

### Verification

- A test asserting sparse-equals-dense on a small mesh, which is provable here
  rather than approximate — this is the strongest kind of refactor guard and it
  is available for free.
- A test at a size the dense version could not allocate, to pin that the cap is
  actually lifted.

### Adjacent gap

`TopologyOptimizer` has no correctness test — only smoke coverage in
`tests/test_smoke.py` and `tests/test_regressions.py`. (`tests/test_topology.py`
is mesh topology, despite the name.) `AGENTS.md` asks for coverage of an untested
path *before* refactoring it, so a test pinning the optimizer's behavior belongs
in this effort, ahead of the change. A compliance-minimizing cantilever with a
volume constraint is the standard case: compliance should decrease monotonically
and the volume fraction should be met.
