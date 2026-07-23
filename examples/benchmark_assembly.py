"""Time assembly and solve versus mesh size, split into their two phases.

Makes the scaling work concrete and guards against regressions: run it before and
after a change to see where the time goes. Sparse matrices moved the cost off the
solve and onto the per-element assembly loop; batching assembly has moved it back
onto the sparse factorization, which now dominates at any interesting resolution
-- the motivation for an iterative solver next.

    uv run python -m examples.benchmark_assembly
"""
import logging
import time

from fem.boundary import BCType, BoundaryConditions
from fem.mesh.generation import create_box_mesh
from fem.regions import everywhere
from fem.solver import LinearElastic, Solver
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
    solver = Solver(mesh, LinearElastic(E=200.0, nu=0.3, source=lambda p: [1.0, 0.0, 0.0]), bc)

    _, t_assemble = _time(solver.assemble_everything)
    bc_r = solver.resolved_bc
    constraints = (bc_r.free_idxs, bc_r.fixed_idxs, bc_r.fixed_values)
    _, t_solve = _time(lambda: DiscreteSystem(solver.K, constraints).solve(solver.b))

    nnz = solver.K.nnz
    dofs = solver.space.n_dofs
    print(
        f'n={n:>3}  tets={len(mesh.elements):>7}  dofs={dofs:>7}  '
        f'nnz/dof={nnz / dofs:>5.1f}  assemble={t_assemble:>6.2f}s  '
        f'factor+solve={t_solve:>6.2f}s'
    )


if __name__ == '__main__':
    for n in (5, 9, 13, 17):
        benchmark(n)
