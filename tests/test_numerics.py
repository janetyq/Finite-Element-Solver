"""`fem.numerics.scatter_add`: the indexed accumulation the recoveries and estimators use."""
import numpy as np

from fem.numerics import scatter_add


def test_scatter_add_matches_add_at_on_a_tensor_field():
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 7, size=(5, 3))          # an (n_elements, N) node map
    values = rng.standard_normal((15, 2, 2))            # one (2, 2) reading per entry
    expected = np.zeros((7, 2, 2))
    np.add.at(expected, indices.ravel(), values)
    result = scatter_add(indices, values, 7)
    assert result.shape == (7, 2, 2)
    np.testing.assert_allclose(result, expected)


def test_scatter_add_on_a_scalar_field_keeps_empty_slots_at_zero():
    result = scatter_add(np.array([0, 2, 2]), np.array([1.0, 2.0, 3.0]), 4)
    np.testing.assert_allclose(result, [1.0, 0.0, 5.0, 0.0])
