"""What kind of value a PDE's unknown takes at each point.

The number of components an unknown carries per node is not always a free
choice. Elasticity's unknown is a *vector field on the domain*, so its component
count is determined by the domain: 2 on a triangle mesh, 3 on a tet mesh. Storing
the shape and deriving the count is what lets one `LinearElastic` class describe
both, where a class constant could only ever describe one.

Deliberately a small sum type rather than an `Enum`. A k-species reaction-diffusion
system has k components unrelated to the spatial dimension, and enum members are
singletons with nowhere to put k; against the protocol below that case is an
additive `System(n)` rather than a change to every existing declaration. Nothing
needs it yet, so it is not written.
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
