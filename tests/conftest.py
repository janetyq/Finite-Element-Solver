"""Shared pytest setup.

The project's modules import matplotlib (via Plotter). Force the non-interactive
Agg backend so importing/exercising them never tries to open a window — required
for headless/CI runs.
"""
import matplotlib

matplotlib.use("Agg")
