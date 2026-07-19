"""Shared `Demo` descriptor used by each example file's `DEMOS` registry and by cli.py."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Demo:
    name: str
    func: Callable
    needs_mesh: bool = True      # cli.py loads --mesh and passes it as the first arg
    returns_plotter: bool = True  # False: demo manages its own display/output; --save is rejected
