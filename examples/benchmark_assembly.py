"""Time assembly and the two solve backends versus mesh size, on a 3D elastic box.

Makes the scaling work concrete and guards against regressions: run it before and
after a change to see where the time goes. Sparse matrices moved the cost off the
solve and onto assembly; batching assembly moved it back onto the sparse
factorization, which dominates at any interesting 3D resolution. This benchmark is
what motivated the iterative backend, so it now times both: the direct `splu`
factor+solve against AMG-preconditioned CG. The direct cost grows super-linearly
with fill-in; the AMG-CG cost should overtake it as the mesh grows.

    uv run python -m examples.benchmark_assembly
"""
import logging
import time

from fem.boundary import BCType, BoundaryConditions
from fem.linalg import DirectBackend, IterativeBackend
from fem.materials import LinearElasticMaterial
from fem.mesh.generation import create_box_mesh
from fem.problem import linear_elastic
from fem.regions import everywhere
from fem.system import DiscreteSystem

logging.disable(logging.CRITICAL)  # silence per-solve logging for clean timing


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def benchmark(n: int) -> None:
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0, 0.0])
    material = LinearElasticMaterial(E=200.0, nu=0.3)

    # Building the LinearProblem assembles the stiffness and the load.
    problem, t_assemble = _time(
        lambda: linear_elastic(mesh, material, bc, source=lambda p: [1.0, 0.0, 0.0])
    )
    A, b = problem.tangent(None), problem.load

    # Each backend factors/preconditions in DiscreteSystem's constructor and solves
    # once; timing the whole construct+solve captures the setup each pays.
    _, t_direct = _time(lambda: DiscreteSystem(A, problem.constraints, DirectBackend()).solve(b))
    _, t_iter = _time(lambda: DiscreteSystem(A, problem.constraints, IterativeBackend()).solve(b))

    dofs = problem.space.n_dofs
    print(
        f'n={n:>3}  tets={len(mesh.elements):>8}  dofs={dofs:>8}  '
        f'assemble={t_assemble:>6.2f}s  direct={t_direct:>7.2f}s  amg_cg={t_iter:>6.2f}s'
    )


if __name__ == '__main__':
    for n in (5, 9, 13, 17, 21):
        benchmark(n)
