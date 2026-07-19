"""Utility: generate and save a rectangular triangle mesh to files/.

Run from the repo root:

    uv run python examples/generate_mesh.py
"""
from fem.mesh.generation import create_rect_mesh


def main():
    mesh = create_rect_mesh(corners=[[0, 0], [2, 1]], resolution=(60, 30))
    mesh.save("files/mesh_60x30.json")


if __name__ == "__main__":
    main()
