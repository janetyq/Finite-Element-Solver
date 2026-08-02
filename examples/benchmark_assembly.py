"""Time assembly and the two solve backends versus mesh size, on a 3D elastic box.

Makes the scaling work concrete and guards against regressions: run it before and
after a change to see where the time goes. Sparse matrices moved the cost off the
solve and onto assembly; batching assembly moved it back onto the sparse
factorization, which dominates at any interesting 3D resolution. This benchmark is
what motivated the iterative backend, so it now times both: the direct `splu`
factor+solve against AMG-preconditioned CG. The direct cost grows super-linearly
with fill-in; the AMG-CG cost should overtake it as the mesh grows.

    uv run python -m examples.benchmark_assembly
    uv run python examples/cli.py run backends
"""
import logging
import time

from fem.boundary import BCType, BoundaryConditions
from fem.backends import DirectBackend, IterativeBackend
from fem.materials import LinearElasticMaterial
from fem.mesh.ruppert import create_box_mesh
from fem.problem import linear_elastic
from fem.regions import everywhere
from fem.system import DiscreteSystem

from demo_registry import Demo, DemoResult

logging.disable(logging.CRITICAL)  # silence per-solve logging for clean timing

DEFAULT_SIZES = (5, 9, 13, 17, 21)


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def benchmark(n: int) -> str:
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
    return (
        f'n={n:>3}  tets={len(mesh.elements):>8}  dofs={dofs:>8}  '
        f'assemble={t_assemble:>6.2f}s  direct={t_direct:>7.2f}s  amg_cg={t_iter:>6.2f}s'
    )


def demo_backends(sizes=DEFAULT_SIZES):
    """Time assembly and both solve backends on a 3D elastic box, over a range of sizes."""
    return DemoResult(text='\n'.join(benchmark(n) for n in sizes))


DEMOS = [
    # Text rather than a figure: the result is a scaling trend across sizes, not a field
    # over a mesh, so there is nothing for a Plotter to draw.
    # The sweep is the point -- a crossover is a claim about two curves -- so the CLI
    # and the gallery run all five sizes. The test only needs to know that assembly and
    # both backends still compose, which n=5 answers in 0.01s where the full sweep
    # takes 11.6s, over half of it one sparse factorisation at n=21.
    Demo('backends', demo_backends, section='Accuracy & performance',
         smoke_kwargs={'sizes': (5,)}),
]


if __name__ == '__main__':
    print(demo_backends().text)
