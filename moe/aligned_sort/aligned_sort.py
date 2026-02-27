from functools import partial
from typing import Any, Literal
import dataclasses

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.custom_vjp, nondiff_argnames=("empty_scatter",))
def unique_gather(x: jax.Array, idx: jax.Array, inv_idx: jax.Array | None = None, empty_scatter: bool = True):
  del empty_scatter, inv_idx
  return x[idx, ...]


def unique_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, empty_scatter: bool):
  return (unique_gather(x, idx, inv_idx, empty_scatter=empty_scatter), (x.shape, idx, inv_idx))


def unique_gather_bwd(empty_scatter: bool, res: tuple[Any, jax.Array, jax.Array], g: jax.Array):
  (x_shape, idx, inv_idx) = res
  if g.shape[0] == x_shape[0]:
    inv_idx = inv_idx if inv_idx is not None else jnp.argsort(idx)
    grad = g[inv_idx, ...]
  elif g.shape[0] != x_shape[0] and inv_idx is not None:  # padded gather
    grad = g[inv_idx, ...]
  else:  # scatter
    _init = jax.lax.empty if empty_scatter else jnp.zeros
    grad = _init(x_shape, dtype=g.dtype).at[idx, ...].set(g, mode="drop")
  return (grad, None, None)


unique_gather.defvjp(unique_gather_fwd, unique_gather_bwd)


def count_groups(idx: jax.Array, length: int, method: Literal["bincount", "masked_sum"]) -> jax.Array:
  if method == "bincount":
    return jnp.bincount(idx, length=length)
  return jnp.sum(idx[..., None] == jnp.arange(length), -2)


@partial(jax.jit, static_argnames=("max_size_per_idx",))
def add_indices(idx_list: jax.Array, sizes: jax.Array, max_size_per_idx: int, fill_value: int = 2**31 - 1):
  """One way to add indices from a list in desired counts, filling the rest with a fill value."""
  start_idx, end_idx = jnp.cumsum(sizes) - sizes, jnp.cumsum(sizes)
  iota = jnp.arange(idx_list.size * max_size_per_idx)[None, :]
  mask = (iota >= start_idx[:, None]) & (iota < end_idx[:, None])
  # the result is a list having sizes[0] of idx_list[0], sizes[1] of idx_list[1] and so on
  # the unfilled values are filled with fill_value
  return jnp.sum(idx_list[:, None] * mask, axis=0) + ~jnp.any(mask, axis=0) * fill_value


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PaddedGroupPaddedMetadata:
  group_idx: jax.Array
  group_idx_with_padding: jax.Array
  group_counts: jax.Array
  group_counts_with_padding: jax.Array
  sort_idx: jax.Array
  isort_idx: jax.Array


def compute_padded_group_gather(
  group_idx: jax.Array, num_groups: int, multiple: int, group_counts: jax.Array | None = None
) -> PaddedGroupPaddedMetadata:
  """Compute metadata for sorting tokens according to group_idx with padding to make groups divisible by `multiple`."""

  assert multiple >= 1
  if group_counts is None:
    group_counts = count_groups(group_idx, length=num_groups, method="masked_sum")
  else:
    assert group_counts.size == num_groups

  if multiple != 1:
    padding_idxs = add_indices(jnp.arange(num_groups), -group_counts % multiple, max_size_per_idx=multiple - 1)
    group_idx_with_padding = jnp.concat([group_idx, padding_idxs], axis=0)
    group_counts_with_padding = group_counts + (-group_counts % multiple)
  else:
    group_idx_with_padding, group_counts_with_padding = group_idx, group_counts
  sort_idx = jnp.argsort(group_idx_with_padding)
  isort_idx = jnp.argsort(sort_idx)[: group_idx.size]

  return PaddedGroupPaddedMetadata(
    group_idx, group_idx_with_padding, group_counts, group_counts_with_padding, sort_idx, isort_idx
  )


########################################################################################################################
# Tests ################################################################################################################
########################################################################################################################


class TestUniqueGather(parameterized.TestCase):
  def test_forward_pass(self):
    keys = iter(jax.random.split(jax.random.key(0), 10))
    x = jax.random.normal(next(keys), (10, 32))
    idx = jnp.array([0, 5, 2, 9, 2])

    y_custom = unique_gather(x, idx)
    y_native = x[idx, ...]

    np.testing.assert_allclose(y_custom, y_native, atol=1e-6)

  def test_backward_pass_scatter(self):
    # This test verifies the backward pass for the scatter case (when output size != input size).
    # The VJP uses .set(..., mode="drop"), which implies that for a correct gradient sum,
    # the indices in `idx` should be unique. The function name "unique_gather" also suggests this.
    keys = iter(jax.random.split(jax.random.key(1), 10))
    x = jax.random.normal(next(keys), (16, 4))
    # Use a permutation for unique indices
    idx = jax.random.permutation(next(keys), 10)  # take 10 unique indices from 0-15

    def f_custom_sum(data):
      # Use empty_scatter=False to get zero-initialization, which is easier to reason about for a sum grad.
      return jnp.sum(unique_gather(data, idx, inv_idx=None, empty_scatter=False))

    def f_native_sum(data):
      return jnp.sum(data[idx, ...])

    grad_custom = jax.grad(f_custom_sum)(x)
    grad_native = jax.grad(f_native_sum)(x)

    expected_grad = jnp.zeros_like(x).at[idx].set(1.0)

    np.testing.assert_allclose(grad_native, expected_grad, atol=1e-6)
    np.testing.assert_allclose(grad_custom, grad_native, atol=1e-6)

  def test_backward_pass_gather_permutation(self):
    keys = iter(jax.random.split(jax.random.key(2), 10))
    x = jax.random.normal(next(keys), (10, 8))
    # A permutation of indices
    perm_idx = jax.random.permutation(next(keys), x.shape[0])
    inv_perm_idx = jnp.argsort(perm_idx)

    def f(data):
      gathered = unique_gather(data, perm_idx, inv_perm_idx)
      # Ensure shape is the same to trigger the correct branch in bwd pass
      self.assertEqual(gathered.shape, data.shape)
      return jnp.sum(gathered * jnp.arange(data.size).reshape(data.shape).astype(data.dtype))

    grad_custom = jax.grad(f)(x)
    grad_native = jax.grad(
      lambda data: jnp.sum(data[perm_idx, ...] * jnp.arange(data.size).reshape(data.shape).astype(data.dtype))
    )(x)

    np.testing.assert_allclose(grad_custom, grad_native, atol=1e-5)


class TestPaddedGroupGather(parameterized.TestCase):
  @parameterized.product(num_groups=[4, 32], num_tokens=[256, 2048], multiple=[1, 8, 16])
  def test_padding_and_sorting_logic(self, num_groups, num_tokens, multiple):
    seed = num_groups + num_tokens + multiple  # Create a deterministic seed from params
    keys = iter(jax.random.split(jax.random.key(seed), 10))

    group_idx = jax.random.randint(next(keys), (num_tokens,), 0, num_groups)

    metadata = compute_padded_group_gather(group_idx, num_groups, multiple)

    self.assertTrue(jnp.all(metadata.group_counts_with_padding % multiple == 0))
    self.assertGreaterEqual(metadata.group_idx_with_padding.size, jnp.sum(metadata.group_counts_with_padding))
    self.assertTrue(jnp.all(metadata.group_counts_with_padding >= metadata.group_counts))

    original_data = jax.random.uniform(next(keys), (num_tokens,))

    padded_data = jnp.full((metadata.group_idx_with_padding.size,), -1.0, dtype=original_data.dtype)
    padded_data = padded_data.at[:num_tokens].set(original_data)

    sorted_data = padded_data[metadata.sort_idx]
    processed_sorted_data = sorted_data * 2.0
    unsorted_processed_data = processed_sorted_data[metadata.isort_idx]
    expected_result = original_data * 2.0

    self.assertEqual(unsorted_processed_data.shape, expected_result.shape)
    np.testing.assert_allclose(unsorted_processed_data, expected_result, atol=1e-6)

  def test_group_counts_provided(self):
    num_groups, num_tokens, multiple = 4, 10, 4
    keys = iter(jax.random.split(jax.random.key(42), 10))
    group_idx = jax.random.randint(next(keys), (num_tokens,), 0, num_groups)

    # Calculate metadata without providing counts
    metadata1 = compute_padded_group_gather(group_idx, num_groups, multiple)

    # Pre-calculate counts and provide them
    group_counts = count_groups(group_idx, length=num_groups, method="bincount")
    metadata2 = compute_padded_group_gather(group_idx, num_groups, multiple, group_counts=group_counts)

    # The results should be identical
    np.testing.assert_array_equal(metadata1.group_idx_with_padding, metadata2.group_idx_with_padding)
    np.testing.assert_array_equal(metadata1.group_counts, metadata2.group_counts)
    np.testing.assert_array_equal(metadata1.group_counts_with_padding, metadata2.group_counts_with_padding)
    np.testing.assert_array_equal(metadata1.sort_idx, metadata2.sort_idx)
    np.testing.assert_array_equal(metadata1.isort_idx, metadata2.isort_idx)

  @parameterized.product(num_groups=[4, 32], num_tokens=[64, 2048], multiple=[8, 16])
  def test_padded_gather_grad(self, num_groups, num_tokens, multiple):
    dim = 128
    seed = num_groups + num_tokens + multiple
    keys = iter(jax.random.split(jax.random.key(seed), 10))

    group_idx = jax.random.randint(next(keys), (num_tokens,), 0, num_groups)
    x = jax.random.normal(next(keys), (num_tokens, dim))

    metadata = compute_padded_group_gather(group_idx, num_groups, multiple)
    loss_const = jax.random.normal(next(keys), num_tokens)
    group_weights = jax.random.normal(next(keys), num_groups)

    self.assertTrue(jnp.all(metadata.group_counts_with_padding % multiple == 0))

    # def assert_idxs_bounds(shape, idxs):
    #   assert np.all(idxs < shape[0])
    #   assert np.all(idxs >= 0)

    def padded_path_fn(data):
      sorted_data = unique_gather(data, metadata.sort_idx, metadata.isort_idx, empty_scatter=False)
      # jax.debug.callback(partial(assert_idxs_bounds, sorted_data.shape), metadata.isort_idx)

      end_group = jnp.cumsum(metadata.group_counts_with_padding)
      start_group = end_group - metadata.group_counts_with_padding
      iota = jnp.arange(sorted_data.shape[0])
      weighting = jnp.sum(group_weights[None, :]
                          * ((iota[:, None] >= start_group[None, :]) & (iota[:, None] < end_group[None, :])), -1)
      sorted_data *= weighting[:, None]

      inv_idx = jnp.where(metadata.sort_idx < metadata.isort_idx.size, metadata.sort_idx, 0)
      unsorted_data = unique_gather(sorted_data, metadata.isort_idx, inv_idx, empty_scatter=False)
      return jnp.sum(unsorted_data * loss_const[..., None])

    def reference_fn(data):
      sort_idx = jnp.argsort(group_idx)
      sorted_data = data[sort_idx]

      end_group = jnp.cumsum(metadata.group_counts)
      start_group = end_group - metadata.group_counts
      iota = jnp.arange(sorted_data.shape[0])
      weighting = jnp.sum(group_weights[None, :]
                          * ((iota[:, None] >= start_group[None, :]) & (iota[:, None] < end_group[None, :])), -1)
      sorted_data *= weighting[:, None]

      unsorted_data = sorted_data[jnp.argsort(sort_idx)]
      return jnp.sum(unsorted_data * loss_const[..., None])

    # print("#" * 80)
    # print(jax.jit(padded_path_fn).trace(x).jaxpr)
    # print("#" * 80)
    # print(jax.jit(jax.grad(padded_path_fn)).trace(x).jaxpr)
    # print("#" * 80)

    grad_padded = jax.grad(padded_path_fn)(x)
    grad_ref = jax.grad(reference_fn)(x)

    # The gradient from the padded path should match the reference
    np.testing.assert_allclose(grad_padded, grad_ref, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
