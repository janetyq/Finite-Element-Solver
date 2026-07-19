"""Demo: red-green adaptive mesh refinement.

Refine random triangles of a small mesh several times, then plot the original
vs. refined mesh. Run from the repo root:

    uv run python examples/refinement_demo.py
"""
import random

import numpy as np

from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RefinementMesh
from fem.plot.plotter import Plotter


def main():
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    mesh = Mesh(vertices, elements, boundary)
    original_mesh = mesh.copy()

    refiner = RefinementMesh(mesh)
    for _ in range(8):
        refine_list = set(random.randint(0, len(mesh.elements) - 1) for _ in range(5))
        refiner.refine_triangles(refine_list)
        mesh = refiner.get_mesh()

    plotter = Plotter(1, 2, title="Red-Green Refinement")
    plotter.plot(original_mesh, mode="mesh", idx=(0, 0), title="Original")
    plotter.plot(mesh, mode="mesh", idx=(0, 1), title="Refined")
    plotter.show()


if __name__ == "__main__":
    main()
