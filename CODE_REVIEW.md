# Code Review — Finite Element Solver

A review of the current state of the codebase, covering bugs, correctness concerns,
performance, architecture, and open-ended ideas for the future. This is a genuinely
impressive project — a hand-rolled FEM solver spanning meshing, multiple PDEs, adaptive
refinement, and topology optimization is a large amount of correct, non-trivial numerical
code. The notes below are about hardening and scaling what's already here, not a knock on
the design.

Legend: 🔴 bug / correctness · 🟠 performance / scaling · 🟡 design / maintainability · 💡 idea

---

## 1. Bugs & Correctness

### 🔴 `Element.deformation_gradient` references undefined `N`
`Elements.py:50`
```python
def deformation_gradient(self, u_element):
    return np.eye(N-1) + self.grad_phi.T @ u_element   # N is not defined here
```
`N` is a class attribute — this must be `self.N`. Calling this method today raises
`NameError`. It appears unused so far, which is why it hasn't surfaced.

### 🔴 `Solution.load` is broken
`Solution.py:18-21`
```python
@classmethod
def load(cls, filename):
    with open(filename, 'rb') as f:
        return cls(pickle.load(f))
```
`pickle.load` already returns a fully-formed `Solution`. Wrapping it in `cls(...)` calls
`Solution.__init__(self, femesh, dim)` with a single positional arg and will raise
`TypeError`. It should simply `return pickle.load(f)`.

### 🔴 `Equation.__copy__` drops `dim`
`Solver.py:19-20`
```python
def __copy__(self):
    return self.__class__(self.name, self.parameters.copy())  # dim not passed
```
The copy re-derives `dim` from the name-based default and re-prints the warning. It happens
to resolve to the right value for `linear_elastic` (the only consumer, via
`TopologyOptimizer`), but any other equation copied this way silently reverts to `dim=1`.
Pass `dim=self.dim` explicitly.

### 🔴 Dead/broken function in `helper.py`
`utils/helper.py:12-20` — `assemble_mass_matrix(N, dim)` references `self.mesh`, `self.dim`,
and an undefined `calculate_element_matrix`. It's a leftover from before assembly moved into
`FEMesh`/`Elements`. It can only ever crash; delete it.

### 🔴 `plot_arrows` defined twice; `RefinementMesh.get_mesh` defined twice
- `utils/helper.py:224` and `:270` — two definitions of `plot_arrows`; the second silently
  wins. They're nearly identical, but this is a landmine.
- `utils/refinement.py:199` and `:232` — `get_mesh` defined twice (harmless, but confusing).

### 🔴 `Mesh._get_all_edges` hardcodes triangles
`Mesh.py:60-67` loops `for i in range(3)`, so `edges` is only correct for 2D triangle meshes.
For 1D line elements and 3D tets this produces wrong edges. Since the base `Mesh` is now used
for all dimensions, the edge extraction should key off `self.elements.shape[1]` (or live in
the element classes). `get_boundary_from_vertices_elements` in `helper.py` has the same
triangle-only assumption.

### 🔴 `adaptive_refinement` has a known bug and an inverted loop condition
`Solver.py:173-188`. There's an explicit `# TODO: there's a bug somewhere`, and the guard
```python
while len(self.femesh.elements) < max_triangles or max_iters == 0:
```
reads oddly — the `or max_iters == 0` makes the loop run forever once the triangle budget is
hit *and* iters just happens to be 0, and never terminates on `max_iters` otherwise (it only
decrements it). It's almost certainly meant to be
`while len(...) < max_triangles and max_iters > 0:`. The whole adaptive-refinement demo path
also raises `NotImplementedError` (`Tests.py:166`), so this feature is currently non-functional.

### 🟠 Wave equation ignores boundary conditions
`Solver.py:138` solves with `use_bc=False` and a `# TODO: bc not supported for wave`. Also
`b_right` uses `np.roll(self.b, -1)` (`Solver.py:127`), which rolls a *spatial* load vector by
one index — that looks like a copy-paste of a time-averaging idea applied to the wrong axis.
It's harmless while `b == 0` (the demos have no forcing), but it's a correctness trap the
moment someone adds a source term.

### 🟡 `NeohookeanEnergyDensity` is a stub
`Energies.py:83-90` — `set_grad_u` is `pass`, so the class advertises a capability it doesn't
have. Either finish it or mark it clearly as WIP / raise `NotImplementedError`.

### 🟡 `K_b` not assembled for `dim > 1`
`FEMesh.py:26-28` only builds the boundary stiffness matrix for `dim == 1`
(`# TODO: not assembling K_b for dim=2`). Fine as long as nothing downstream needs it, but
it's an asymmetry worth a docstring so a future caller doesn't assume it exists.

---

## 2. Performance & Scaling

### 🟠 Everything is dense — this is the single biggest limiter
`FEMesh.assemble_matrix` builds `A = np.zeros((dim*N, dim*N))` and solves via
`np.linalg.solve` (`Solver.py:73`). FEM matrices are extremely sparse (each row has a handful
of nonzeros), so this is `O(N²)` memory and `O(N³)` solve time. On the 40×40 meshes in the
tests that's fine; it will fall over well before "interesting" resolutions. The README already
lists "sparse solver" as a next step — concretely:
- Assemble with `scipy.sparse.lil_matrix`/COO triplets, convert to CSR.
- Solve with `scipy.sparse.linalg.spsolve` (or `cg`/`splu` with caching for the time-stepping
  loops, where the matrix is constant across iterations).

This one change probably unlocks 1–2 orders of magnitude in mesh size.

### 🟠 `assemble_everything` runs on every `solve()`
`Solver.py:33` even flags it: `# TODO: don't call this every time`. For heat/wave the system
matrices are rebuilt implicitly each timestep via re-assembly; for topology optimization
`solve()` is called every iteration. Cache `M`, `M_b`, `K` and only re-factor when the mesh or
material actually changes. For time-stepping, pre-factor `(M + K·dt)` once (LU) and reuse.

### 🟠 `calculate_smoothing_matrix` is dense `O(n_elem²)`
`utils/helper.py:196-202` materializes a full element-by-element distance matrix. For topology
optimization at any real resolution this dominates memory. A spatial hash / KD-tree
(`scipy.spatial.cKDTree.query_ball_point`) building a sparse weight matrix would scale far
better and is a near drop-in.

### 🟠 Refinement and meshing are `O(n²)` from linear scans
`utils/refinement.py` is self-described as "very inefficient": `get_shared_triangle`,
`get_triangle_idx`, and `get_point_idx` each do a full linear scan of all triangles/vertices,
inside refinement loops. `get_point_idx` in particular scans every vertex to dedupe midpoints
— an edge→midpoint-index dict would make it `O(1)`. Similarly
`get_boundary_from_vertices_elements` (`helper.py:81`) is `O(edges × elements)`; a single pass
counting edge occurrences in a dict is `O(elements)`.

### 🟠 `EnergySolver` Hessian is dense and rebuilt each Newton step
`EnergySolver.py:89-99` allocates an `(n·dim, n·dim)` dense Hessian every iteration and solves
it densely. Same sparse story as above; here it matters even more because it's inside a Newton
loop.

---

## 3. Architecture & Maintainability

### 🟡 `from X import *` everywhere
Nearly every module does `from Mesh import *`, `from Plotter import *`, `from utils.helper
import *`, etc. This makes it very hard to see where a name comes from, invites accidental
shadowing (see the duplicate `plot_arrows`), and risks circular imports (`Mesh` → `Plotter` →
`helper`; `FEMesh` → `Mesh`/`Plotter`/`Elements`). Recommend explicit imports
(`from Mesh import Mesh`). It's tedious but pays for itself the first time you debug a name
collision.

### 🟡 No packaging / pinned dependencies
`environment.txt` is a freeform note (`conda install ...`, `pip install ...`) with no versions
and it omits `pyvista` (used in `utils/tet.py`) and `svg.path`. Recommend a
`requirements.txt` or `pyproject.toml` with pinned versions, and move the source modules under
a package directory (e.g. `fem/`) so the `sys.path.append('..')` hacks in `utils/*.py` can go
away. Right now imports only work from the repo root.

### 🟡 No automated tests
`Tests.py` is a set of manual, plot-driven demos gated behind commented-out calls in
`__main__` and requiring a human to look at figures. There are zero assertions and no way to
catch regressions. High-value additions:
- **Method-of-manufactured-solutions** tests: pick `u`, compute `f = -Δu`, solve, assert L2
  error decreases at the expected `O(h²)` rate under refinement. This directly validates the
  core solver numerically.
- **Analytic-patch tests**: constant/linear fields should be reproduced exactly.
- **Unit tests** for the pure helpers (`Enu_to_Lame`↔`Lame_to_Enu` round-trip,
  `calculate_polygon_area`, `calculate_tetrahedron_volume`, `calculate_circumcenter`).
- The existing `check_gradient`/`check_hessian` finite-difference utilities are great — wire
  them into `pytest` with a numeric tolerance instead of eyeballing a log-log plot.

### 🟡 `pickle` for persistence
`Solution.save/load` uses `pickle`, which executes arbitrary code on load and is fragile
across refactors (it stores the class path). For a research tool loading only your own files
it's low-risk, but a plain `.npz`/JSON schema for the numeric arrays would be safer and more
portable.

### 🟡 Magic strings and dispatch dicts
Equation types (`"poisson"`, `"heat"`, ...), plot modes (`"surface"`, `"colored"`, ...), and
BC types (`"dirichlet"`, `"neumann"`) are stringly-typed and validated only at the dispatch
site. An `Enum` (or subclassing `Equation` per PDE) would give editor autocomplete and catch
typos earlier. This also opens the door to per-equation residual/assembly logic living on the
class instead of `if name == ...` ladders.

### 🟡 `print`-based logging
Solvers print progress unconditionally (`Solver.py`, `TopologyOptimizer`, `EnergySolver`).
The `Equation` constructor even prints a warning. Switching to the `logging` module (with
levels) would let callers silence or capture output — important once this is used as a library
rather than a script.

### 🟡 Deprecated / fragile matplotlib usage
`utils/helper.py:229` uses `matplotlib.cm.get_cmap('viridis')`, which is deprecated in modern
matplotlib (removed in 3.9+; use `matplotlib.colormaps['viridis']`). Worth fixing before an
environment bump breaks all plotting.

### 🟡 Hardcoded 2D assumptions in "generalized" paths
Despite the recent dim-generalization work, several spots still assume 2D:
`Solution.get_deformed_mesh` reshapes with `self.dim` (good) but
`TopologyOptimizer._get_deformed_mesh` hardcodes `reshape(-1, 2)` (`TopologyOptimizer.py:121`),
`EnergySolver` asserts `dim == 2`, and `Energies.LinearElasticEnergyDensity` is hardwired to
`np.eye(2)` / `(2,2,2,2)` tensors. Worth a tracking list of "not yet N-D" so the boundary is
explicit.

### 🟡 Scattered TODOs and commented-out code
There are ~40 `TODO`/commented blocks (e.g. the entire residual-calculation section in
`Solver.py:190-221`, the commented adaptive-refinement demo in `Tests.py:168-187`). Consider
moving these to a `TODO.md` / issue tracker so the source reads as finished code and the intent
isn't lost in comments.

---

## 4. Smaller Nits

- `Solver.solve_linear_elastic` (`Solver.py:156-166`) re-fetches `element = self.femesh.elements[e_idx]`
  on the first line of the loop even though it's already the loop variable, and indexes
  `self.femesh.element_objs[e_idx]` four separate times — bind it to a local.
- `BoundaryConditions.check()` (`BoundaryConditions.py:40-44`) is a stub (`pass`) but is part
  of the intended validation flow (max one BC per node, BC only on boundary). Worth finishing —
  silently-overlapping Dirichlet/Neumann on the same node is an easy footgun given the dict
  overwrite in `add`.
- `BoundaryConditions.add` overwrites on repeated indices (dict assignment) with no warning; a
  duplicate BC silently wins-last.
- `oc_density` upper bound `r = 1e15` with a `while (l*(1+1e-15)) < r` termination is a fragile
  way to bisect; a fixed iteration count or relative tolerance on `r-l` is more predictable.
- `except:` bare clauses (`EnergySolver.py:117`, `TopologyOptimizer.py:119`) swallow all
  errors including `KeyboardInterrupt`; catch the specific exception
  (`np.linalg.LinAlgError`, `KeyError`).
- `Plotter.__init__` computes `size = 5 if ... else 4` (`Plotter.py:13`) and then never uses
  `size`.
- README self-notes it is "a bit behind the current state of the project" — a docs refresh pass
  would help, especially given how much of the actual capability isn't reflected there.

---

## 5. Open-Ended Suggestions & Future Ideas

**Numerics**
- 💡 **Higher-order elements.** Already on the roadmap (quadratic basis). The `Element` class
  hierarchy is well-positioned for this — add `QuadraticTriangleElement` with its own shape
  functions and a real quadrature rule (the `utils/quadrature.py` rules are written but not yet
  wired into assembly — integrating them would let you support non-constant material fields and
  higher-order elements cleanly).
- 💡 **Proper Gaussian quadrature.** Assembly currently uses closed-form linear-element
  integrals. A general quadrature layer (reference element + Gauss points + Jacobian) would
  make adding new element types and variable coefficients far easier, and is a prerequisite for
  the quadratic elements above.
- 💡 **Iterative solvers + preconditioning.** Once sparse, add CG with a Jacobi/AMG
  preconditioner for the SPD systems (Poisson, elasticity). This is where large 3D problems
  become tractable.
- 💡 **Finish the a posteriori error estimator** so adaptive refinement is fully closed-loop —
  the residual scaffolding is already sketched in `Solver.py`.

**Features**
- 💡 The README's roadmap (thermal expansion, transport, fluid mechanics, nonlinear
  hyperelasticity via the existing `EnergySolver`/`Energies` machinery) all fit the current
  architecture well. Finishing `NeohookeanEnergyDensity` would immediately give a nonlinear
  material through the already-working Newton solver.
- 💡 **Time-integration abstraction.** Backward-Euler (heat) and Crank–Nicolson (wave) are
  hand-coded inline. A small `TimeIntegrator` interface (θ-method / generalized-α) would
  deduplicate and make it trivial to add new dynamics.
- 💡 **BC support for the wave solver**, and a general Robin BC path (the README mentions Robin
  conditions in the Poisson example but the `BoundaryConditions` class only models
  Dirichlet/Neumann explicitly).

**Engineering**
- 💡 **CI + a test suite** (GitHub Actions running `pytest`) — with the MMS convergence tests
  above, this would give real confidence that refactors (like the sparse migration) don't
  silently change results.
- 💡 **Package it** (`pip install -e .`) so it can be imported from anywhere and the
  `sys.path` hacks disappear.
- 💡 **Benchmarks.** A tiny script timing assembly + solve vs. mesh size would make the impact
  of the sparse migration concrete and guard against future regressions.
- 💡 **Type hints + docstrings on the public API** (`Solver`, `FEMesh`, `BoundaryConditions`,
  `Equation`) — this is the surface most likely to be used by others (or future-you).

---

## Suggested Priority Order

1. **Fix the outright bugs** (§1: `deformation_gradient`, `Solution.load`, dead
   `assemble_mass_matrix`, duplicate `plot_arrows`) — these are quick and remove landmines.
2. **Sparse matrices + solver** (§2) — the highest-leverage single change for capability.
3. **A minimal `pytest` suite with an MMS convergence test** (§3) — so #2 can be done safely.
4. **Packaging + explicit imports + pinned deps** (§3) — makes everything after this easier.
5. Then pick from the roadmap ideas (§5).
