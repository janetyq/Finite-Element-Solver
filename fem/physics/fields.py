"""What kind of value a PDE's unknown takes at each point.

Elasticity's unknown is a vector field, so its component count follows from the
domain: 2 on a triangle mesh, 3 on a tet mesh. Storing the shape and deriving the
count lets one `Elasticity` class describe both. A small sum type rather than an
`Enum`, so a k-component system can be added as a `System(n)` later.
"""
from dataclasses import dataclass
from typing import Protocol


class FieldShape(Protocol):
    """Resolves an unknown's component count against the domain it lives on."""

    def components_for(self, spatial_dim: int) -> int: ...


@dataclass(frozen=True)
class Scalar:
    """One value per node: temperature, potential, concentration."""

    def components_for(self, spatial_dim: int) -> int:
        return 1


@dataclass(frozen=True)
class Vector:
    """One component per spatial dimension: displacement, velocity."""

    def components_for(self, spatial_dim: int) -> int:
        return spatial_dim
