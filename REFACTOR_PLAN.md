# Refactor plan — `dim`, `FieldShape`, and `FunctionSpace`

Explanatory companion to `ARCHITECTURE.md`. That document argues the design; this one walks
through *why* in slower steps, with examples. It is scaffolding for one effort — delete it
when the work lands rather than maintaining it.

---

## 1. Start with a concrete failure

Suppose you want to solve linear elasticity on a tetrahedral mesh. Today you cannot, and the
reason is one line:

```python
class LinearElastic(Equation):
    dim: ClassVar[int] = 2
```

`ClassVar` means this value belongs to the *class*, not to any particular problem. Every
`LinearElastic` anywhere in the program has `dim == 2`. But a displacement field on a tet mesh
has **three** components per node, not two. There is nowhere to say that. The class constant
has already decided.

This is why "N-D elasticity" in `BACKLOG.md` is blocked. It is not blocked on the einsums or
the tet element — `LinearTetrahedralElement` already has `calculate_B` and `calculate_D`. It is
blocked on having nowhere to write down the right number.

---

## 2. Why the number is in the wrong place: `dim` means two things

Trace what `dim` refers to across four combinations:

| Problem | Mesh | Spatial dimension | Components per node | `dim` today |
|---|---|:-:|:-:|:-:|
| Poisson | triangles | 2 | 1 | 1 |
| Poisson | tets | 3 | 1 | 1 |
| Elasticity | triangles | 2 | 2 | 2 |
| Elasticity | tets | 3 | 3 | — *can't express* |

Two genuinely different quantities are being spelled with one word:

- **spatial dimension** — how many coordinates a point has. A property of the *mesh*.
  Already recoverable as `vertices.shape[1]`.
- **components per node** — how many unknowns sit at each node. A property of the
  *equation*: 1 for temperature, 2 or 3 for a displacement vector.

Look at row 3. For 2D elasticity both quantities equal 2. That coincidence is what lets the
conflation survive — as long as the only vector problem you ever run is 2D, one integer
genuinely does serve both jobs. Rows 2 and 4 are where they come apart.

### The conflation has already produced a bug

`EnergySolver` contains this guard:

```python
self.dim = self.equation.dim      # LinearElastic.dim is a ClassVar == 2, always
if self.dim != 2:                 # so this can never fire
    raise NotImplementedError(f'EnergySolver only supports 2D for now (got dim={self.dim})')
```

The author meant "refuse anything that isn't 2D." But `self.dim` is components-per-node, which
is unconditionally 2 here, so the check is dead code. Hand `EnergySolver` a tet mesh today and
it walks straight past this guard and fails several frames later inside
`LinearElasticEnergyDensity.set_grad_u`, which rejects the `(3, 2)` gradient it gets handed.

Still loud, but the error names the wrong thing. Written against the quantity the author
actually meant, the guard becomes live:

```python
if femesh.spatial_dim != 2:   # the energy densities are built at fixed rank 2
```

**This is the general argument for the whole refactor.** The guard isn't wrong because someone
was careless. It's wrong because the vocabulary had no word for what they meant, so they
reached for the nearest available one.

### There is a third dimension, and the elements conflate it too

`fem/elements.py` has the same problem one layer down, using `N - 1` wherever it means
"spatial dimension":

```python
def deformation_gradient(self, u_element):
    return np.eye(self.N - 1) + self.grad_phi.T @ u_element

def calculate_dF_dx(self):
    dF_dx = np.zeros((self.N - 1, self.N - 1, self.N, self.N - 1))
```

For a simplex, `N - 1` is the **reference dimension** — the dimension of the element itself,
2 for a triangle and 3 for a tet. It equals the spatial dimension only when the element fills
its ambient space. So there are three quantities in play, and the code currently names none:

| | Triangle in 2D | Triangle in 3D | Tet in 3D |
|---|:-:|:-:|:-:|
| `spatial_dim` — ambient space | 2 | **3** | 3 |
| `reference_dim` — the element itself (`N-1`) | 2 | **2** | 3 |
| `n_components` — a vector unknown | 2 | **3** | 3 |

The middle column is a *surface mesh*: triangles embedded in 3D. It is the only case where
reference and spatial dimension differ, which is why the conflation has survived.

Effort 1 adds `Element.reference_dim` alongside `Mesh.spatial_dim` so the distinction is at
least sayable. It does not attempt surface support — see §7.

---

## 3. Fix one: store the *kind*, derive the *count*

The obvious move is to rename `dim` to `n_components` and set it to 3 for tets. That is
better, but still not right, because it makes you *choose* a number that isn't actually a
choice.

Here is the mathematical fact: elasticity's unknown is a **vector field on the domain**. A
vector on a 2D domain has 2 components; on a 3D domain it has 3. The count is *determined by
the domain*, not selected by the user. A class constant cannot express "however many the
domain has" — but a small type can:

```python
class FieldShape(Protocol):
    def components_for(self, spatial_dim: int) -> int: ...


@dataclass(frozen=True)
class Scalar:            # temperature, potential
    def components_for(self, spatial_dim: int) -> int:
        return 1


@dataclass(frozen=True)
class Vector:            # displacement -- one component per spatial dimension
    def components_for(self, spatial_dim: int) -> int:
        return spatial_dim
```

The equation declares what kind of thing its unknown is; the mesh supplies the dimension; the
count falls out:

```python
# before — a number, fixed at class-definition time
class Poisson(Equation):
    dim: ClassVar[int] = 1

class LinearElastic(Equation):
    dim: ClassVar[int] = 2          # also silently means "2D only"

# after — a kind, resolved against whatever mesh you hand it
class Poisson(Equation):
    field: FieldShape = Scalar()

class LinearElastic(Equation):
    field: FieldShape = Vector()
```

Now **one `LinearElastic` class describes both 2D and 3D elasticity**:

```python
n_components = equation.field.components_for(mesh.spatial_dim)
# triangle mesh -> 2
# tet mesh      -> 3
```

Nobody wrote down 2 or 3. That is the point — a number nobody writes down is a number nobody
can write down inconsistently.

### Why a sum type rather than an `Enum`

An `Enum` with `SCALAR`/`VECTOR` members would read just as well and is the obvious first
instinct. It stops working the moment a third case appears — a k-species reaction–diffusion
system has k components, unrelated to `spatial_dim`, and enum members are singletons with
nowhere to put k. That case is purely additive against the type above:

```python
@dataclass(frozen=True)
class System:
    n: int
    def components_for(self, spatial_dim: int) -> int:
        return self.n
```

Note `System` also forces `field` to be an **instance** attribute rather than a `ClassVar` —
`ReactionDiffusion(n_species=3)` takes its count as a constructor argument. That's why the
declarations above are written as plain class-level defaults.

Being accurate about the stakes: migrating enum → sum type later would touch six declarations
and one call site. Small. This isn't averting a disaster, it's picking the one of two
equal-cost options that extends cleanly. Nothing on `BACKLOG.md` needs `System` today.

### Where the derivation happens

Exactly two places today read `equation.dim`:

```python
# fem/solver.py:132  and  fem/energy_solver.py:31
self.dim = self.equation.dim
```

Both become:

```python
self.n_components = equation.field.components_for(femesh.spatial_dim)
```

That is the entire behavioral change. Everything else in Effort 1 is a rename.

### Why the rename matters as much as the type

Every remaining `dim` parameter downstream — `dof_indices(element, dim)`,
`evaluate_field(value, points, dim)`, `ResolvedBC.dim`, `Solution.dim` — means
components-per-node. Renaming them to `n_components` means the two concepts are no longer
*spellable the same way*. That is what actually prevents the next `EnergySolver`-style mixup;
`FieldShape` alone wouldn't.

`fem/typing.py` needs the same pass. Its shape comments currently use `dim` for both meanings:
`(n_vertices, dim)` is spatial, `(n_vertices * dim,)` is components.

---

## 4. Fix two: the missing object

### What a function space is

In FEM notation, you always write the problem as:

> find `u ∈ Vₕ` such that `a(u, v) = L(v)` for all `v ∈ Vₕ`

Three things appear: the domain `Ω`, the forms `a` and `L`, and the discrete space `Vₕ`.

The code has an object for the domain (`Mesh`). It has objects for the physics (`Equation`,
the assembly routines). **It has no object for `Vₕ`.** And `Vₕ` is precisely the thing that
answers "what functions can I represent, and how are the unknowns numbered?"

Because nothing owns that question, `FEMesh` answers it by accident — it is the only object
that happens to know both the element type and the DOF numbering. Everything awkward about
`FEMesh` follows from that.

### What `FEMesh` owns today

| Layer | What's on `FEMesh` |
|---|---|
| Geometry | inherited from `Mesh` |
| **Discretization** | `element_objs`, `boundary_objs`, `element_type`, `dof_indices` |
| **Assembly** | `prepare_matrices`, `assemble_matrix`, `M` / `M_b` / `K` / `K_b` |
| Post-processing | `calculate_energy`, `calculate_mean_value`, … |

Four jobs. The middle two are `Vₕ`'s.

### The concrete symptom: solving two problems on one mesh

Suppose you want Poisson *and* elasticity on the same geometry. Today:

```python
femesh = FEMesh(vertices, elements, boundary)   # assembles at dim=1 in __init__

solver_a = Solver(femesh, Poisson(...), bc)     # prepare_matrices(dim=1)
solver_b = Solver(femesh, LinearElastic(...), bc)  # prepare_matrices(dim=2) -- overwrites!

solver_a.solve()   # reads femesh.K, which is now the dim=2 operator
```

`prepare_matrices` sets `self.dim` and rebuilds `M`, `K`, `M_b`, `K_b` **in place**. The mesh's
operators depend on whichever solver touched it last, and `solver_a` silently computes against
the wrong matrices. Your options today are to duplicate the entire geometry into a second
`FEMesh`, or to be careful.

With the missing object present, the question disappears:

```python
mesh = Mesh(vertices, elements, boundary)       # geometry, once

V_scalar = FunctionSpace(mesh, P1Triangle, n_components=1)
V_vector = FunctionSpace(mesh, P1Triangle, n_components=2)

solver_a = Solver(V_scalar, Poisson(...), bc)
solver_b = Solver(V_vector, LinearElastic(...), bc)
```

One geometry, two spaces, no shared mutable state. There is nothing to overwrite because
**the space is the key** — `V_scalar.mass_matrix` and `V_vector.mass_matrix` are different
objects on different owners.

---

## 5. Composition, not inheritance

This is the part most likely to feel like churn, so here is the reasoning.

Today: `class FEMesh(Mesh)` — asserting *an FEMesh **is a** Mesh with extras.*

Proposed: `FunctionSpace` **has a** mesh.

`Vₕ` is not a kind of `Ω`. It's a **pairing** of `Ω` with an element choice and a component
count. Two spaces can share one domain (that's the example above); one space cannot be a
domain. Inheritance encodes the wrong relationship, and the code pays for it in specific ways:

**Every consumer that wants geometry is handed assembly.**

```python
# fem/solver.py:392 -- adaptive refinement, once per round
self.femesh = refiner.refine(refine_idxs)
#   -> with_topology -> FEMesh.__init__ -> prepare_matrices()  [4 dense N×N matrices]
self.solve()
#   -> assemble_everything -> prepare_matrices(dim=self.n_components)  [rebuilds all 4]
```

Every refinement iteration assembles twice and throws one away. Not because anyone chose that
— because constructing the geometry *is* constructing the operators when they're the same
class.

**Plotting triggers a full re-assembly.**

```python
# fem/solution.py:70
def get_deformed_mesh(self, u=None):
    femesh_deformed = self.femesh.copy()          # assembles 4 matrices
    femesh_deformed.vertices += u.reshape(...)    # then mutates the vertices
    return femesh_deformed
```

You wanted to move some points to draw a picture. You got four dense matrices — *and* the
copy's cached `element_objs` (holding `volume`, `grad_phi`) were computed from the
**undeformed** vertices, so they now describe geometry that no longer exists. Harmless while
it's only drawn; wrong the moment anything computes on it.

Under composition, refinement operates on `space.mesh` and returns a plain `Mesh`; you build a
`FunctionSpace` only when you actually intend to solve. Both problems dissolve rather than
getting fixed.

### One thing to be careful about

Caching operators on the space (`@cached_property mass_matrix`) is only safe if the space is
**immutable**. If someone mutates `mesh.vertices` afterward, the cache silently goes stale.

Note this risk already exists today, in a worse form — that's exactly the
`get_deformed_mesh` trap above. The difference is that a `FunctionSpace` built once and never
mutated makes it structurally impossible, whereas today it depends on everyone being careful.
Worth stating explicitly because it's the one place the proposal *adds* a rule to follow.

---

## 6. What stays on the space, and what doesn't

The operators split by whether they need material data:

| Operator | Needs material? | Home |
|---|---|---|
| `M` (mass) | no | `@cached_property` on `FunctionSpace` |
| `M_b` (boundary mass) | no | `@cached_property` on `FunctionSpace` |
| `K` (stiffness), scalar | no — pure Laplacian | interim: space |
| `K` (stiffness), elastic | **yes** — needs `mu`, `lamb` | later: `assemble(space, form)` |

That last row is the whole reason `assemble_matrix` has this signature:

```python
def assemble_matrix(self, matrix_type_name, element_type_name, dim=1, **kwargs):
```

Two stringly-typed selectors and an untyped keyword bag — because the mesh is being asked to
assemble something it doesn't have the information for, so the information gets smuggled in.
The element then reaches back into a *global* array to find itself:

```python
# fem/elements.py:49
if dim == 1:
    return self.grad_phi @ self.grad_phi.T * self.volume
# otherwise, the equation is linear elastic
idx = kwargs['idx']
B, D = self.calculate_B(), self.calculate_D(kwargs['mu'][idx], kwargs['lamb'][idx])
```

Fixing that needs a `Form` object that carries its own material. **That is deliberately not in
this plan** — it's the step after. Worth knowing the shape now so `FunctionSpace` doesn't get
built in a way that blocks it: material-dependent stiffness stays on the old path for now
rather than being cached onto the space, so it has somewhere to move to later.

---

## 7. Tradeoffs and what I might be wrong about

**Scalar problems on surface meshes are nearly free — but out of scope anyway.** Worth knowing,
because it's less broken than it looks. `LinearElement.__init__` computes shape-function
gradients with a *pseudo*-inverse:

```python
J = (self.vertices[1:] - self.vertices[0]).T
self.grad_phi = dshape_dphi @ np.linalg.pinv(J)
```

For a triangle in 3D that gives `J` of shape (3,2), `pinv(J)` of (2,3), and `grad_phi` of
(3 nodes, 3 ambient components) — the *tangential* gradient, which is exactly the surface
gradient. `calculate_stiffness_matrix(dim=1)` then produces the correct Laplace–Beltrami
stiffness. Whoever wrote `pinv` rather than `inv` already made the scalar case work.

The only thing stopping it is `calculate_polygon_area`, which raises
`NotImplementedError('Polygon area not supported for 3D')`; the 3D triangle formula is
`0.5 * norm(cross(b - a, c - a))`. That's a small standalone commit if you ever want it, and
it deserves an MMS test on a sphere. Not part of these efforts.

**Vector problems on surface meshes are a modelling project, not a typing fix.** `Vector()`
would derive 3 components on a surface mesh, which is arguably right — a shell displacement
does have 3 ambient components. But surface elasticity is not 3D elasticity restricted to a
surface; it's a membrane or shell model with its own constitutive law. And a real shell breaks
the abstraction outright: a Reissner–Mindlin node carries 3 translations *plus* 2 rotations,
five DOFs that don't mean the same kind of thing. "Components per node" as one integer assumes
the DOFs at a node are homogeneous, and shells violate that assumption rather than the count.

`LinearTriangleElement.calculate_B` at least fails loudly here — `b, c = self.grad_phi.T`
unpacks exactly two rows, so a 3D-embedded triangle raises `ValueError`.

**`field` may not live on `Equation` forever.** Once physics moves into `Form`, an
`ElasticityForm` arguably implies vector-valued already, making the declaration redundant.
It's right for the user-facing API today; it's the piece of this proposal most likely to move.

**3D elasticity is unblocked, not demonstrated.** Removing the `ClassVar` removes the blocker.
Whether the tet path then works is untested — treat it as a claim needing an MMS test, not a
consequence. Realistically it's about even odds; if it fails, the right response is a loud
`NotImplementedError` at the boundary, not a partial fix.

**Migration cost is real.** `FEMesh` is referenced by `io.py`, `Solution`, `RedGreenRefiner`'s
generic bound, and the test suite. A delegating shim keeps everything green while call sites
move one at a time, but the shim is itself temporary code, and retiring it is its own commit.

**Two objects are more to learn than one.** Genuinely a cost. The counter is that it's the
standard decomposition every mature FEM library converges on, and it matches how the math is
already written — `Ω` and `Vₕ` are separate in the notation because they vary independently.

---

## 8. Deliberately out of scope

- **The `if dim == 1` physics branch in `calculate_stiffness_matrix`.** Renaming makes its
  fragility *sharper*, not better: 1D elasticity would be `n_components == 1` and silently get
  the Laplacian. It needs the `Form` seam.
- **Batched element geometry.** `FEMesh.__init__` builds one stateful object per element —
  3200 on a 40×40 mesh, each caching a rank-4 `dF_dx` tensor only `EnergySolver` reads. The
  scalable alternative is stateless element *types* plus `(n_elements, …)` arrays. Not in this
  plan, but Effort 2 routes consumers through `space` accessors instead of letting them reach
  into `element_objs`, so that later change stays internal.
- **Surface meshes.** `Element.reference_dim` makes the case *sayable*; nothing more. The
  scalar path needs only a 3D area formula (§7), and the vector path is a shell model.
- **`System(k)` for multi-species problems.** The type accommodates it; nothing needs it yet.
- **Sparse matrices, `TimeIntegrator`, typed `Solution`.** All downstream of these two efforts.

---

## 9. The plan

### Effort 1 — `refactor/field-shape`

No public API change; `Solver(mesh, Poisson(...), bc)` is unaffected throughout.

1. `refactor: add Mesh.spatial_dim and Element.reference_dim` — the two missing names from §2.
   Pure addition.
2. `fix: make EnergySolver's 2D guard live` — the dead-guard fix from §2. The only behavior
   change in the effort, so it gets its own commit.
3. `refactor: introduce FieldShape, derive the component count` — add the sum type, replace the
   six `dim: ClassVar[int]` declarations, update the two read sites. Removing the old ClassVars
   in the same commit matters: leaving both is two sources of truth.
4. `refactor: rename dim → n_components` — the mechanical pass, including `fem/typing.py`.
5. `test: MMS convergence for 3D linear elasticity` — the checkpoint, with the caveat above.

### Effort 2 — `refactor/function-space`

Starts only once Effort 1 lands, so `FunctionSpace` has an unambiguous thing to own.

1. `feat: add FunctionSpace` — new `fem/space.py`, composition over `Mesh`, cached mass
   matrices. Unwired, so it lands green trivially.
2. `refactor: route Solver through FunctionSpace` — `prepare_matrices`' mutation of `self.dim`
   dies here.
3. `refactor: give EnergySolver accessors instead of element reach-ins` — replaces eight
   `femesh.element_objs[e_idx].{grad_phi,dF_dx,volume}` reads. This is what keeps the batching
   decision internal later.
4. `refactor: move calculate_* metrics to fem/postprocess.py` — free functions over
   `(space, u, …)`.
5. `refactor: reduce FEMesh to a shim` — retiring it is a separate, later commit.

### Verification

`uv run pytest` and `uv run ruff check` at every commit. The MMS convergence test in
`tests/test_convergence.py` must stay green **without modification** — if it needs adjusting,
the refactor is wrong, not the test.
