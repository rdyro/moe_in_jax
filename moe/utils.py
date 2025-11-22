from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Sharding
import jax.experimental.pallas as pl

zip_ = zip
zip = partial(zip_, strict=True)


def empty(shape, dtype, out_sharding=None):
  """Create an empty array with the given shape and dtype, and the given sharding."""
  if out_sharding is None:
    out_shape, out_specs = jax.ShapeDtypeStruct(shape, dtype), pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(lambda *args: None, out_shape=out_shape, out_specs=out_specs)()

  spec = out_sharding.spec if isinstance(out_sharding, Sharding) else out_sharding
  assert len(spec) <= len(shape)
  shard_axes = tuple(spec) + (None,) * (len(shape) - len(spec))

  if isinstance(out_sharding, Sharding):
    decorator = partial(jax.shard_map, mesh=out_sharding.mesh, out_specs=out_sharding.spec, check_vma=False)
  else:
    decorator = partial(jax.shard_map, out_specs=out_sharding, check_vma=False)

  @decorator
  def _():
    local_shape = [(s // jax.lax.axis_size(a)) if a is not None else s for s, a in zip(shape, shard_axes)]
    out_shape, out_specs = jax.ShapeDtypeStruct(local_shape, dtype), pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(lambda *args: None, out_shape=out_shape, out_specs=out_specs)()

  return _()


@partial(jax.jit, static_argnames=("pad_size",))
def padding_to_tile(sort_idx, group_sizes, pad_size: int = 128):
  """Align a sort_idx groups to a multiple of the pad_size after the gather.

  The sort_idx is considered split into groups by the entries in group_sizes. This function modified sort_idx
  to produce groups padded to a multiple of the pad_size after the gather with sort_idx.
  """
  padded_group_sizes = pad_size * ((group_sizes + pad_size - 1) // pad_size)
  insert_indices = jnp.cumsum(padded_group_sizes) - padded_group_sizes
  get_indices = jnp.cumsum(group_sizes) - group_sizes

  def fn(i, new_sort_idx):
    slice = jnp.where(jnp.arange(sort_idx.size) < group_sizes[i], jnp.roll(sort_idx, -get_indices[i]), 0)
    new_sort_idx = jax.lax.dynamic_update_index_in_dim(new_sort_idx, slice, insert_indices[i], axis=-1)
    return new_sort_idx

  new_sort_idx = jax.lax.fori_loop(0, group_sizes.size, fn, jnp.zeros(sort_idx.shape[0] * 2, sort_idx.dtype))
  worst_case_size = sort_idx.size + group_sizes.size * (pad_size - 1)
  return new_sort_idx[:worst_case_size], padded_group_sizes


@partial(jax.jit, static_argnames=("max_size",))
def add_indices(idx_list: jax.Array, sizes: jax.Array, max_size: int, fill_value: int = 2 ** 31 - 1):
  """One way to add indices from a list in desired counts, filling the rest with a fill value."""
  start_idx = jnp.cumsum(sizes) - sizes
  end_idx = jnp.cumsum(sizes)
  iota = jnp.arange(idx_list.size * max_size)[None, :]
  mask = (iota >= start_idx[:, None]) & (iota < end_idx[:, None])
  # the result is a list having sizes[0] of idx_list[0], sizes[1] of idx_list[1] and so on
  # the unfilled values are filled with fill_value
  return jnp.sum(idx_list[:, None] * mask, axis=0) + ~jnp.any(mask, axis=0) * fill_value


def spread_arange_gather(total_length: jax.Array, counts: jax.Array, multiple: int = 4):
  """Creates a dilated arange corresponding to group sizes (conuts) padded to `multiple`."""
  new_counts = counts + (-counts % multiple)
  offsets = (jnp.cumsum(new_counts) - new_counts) - (jnp.cumsum(counts) - counts)
  starts = jnp.cumsum(new_counts) - new_counts
  ends = starts + counts
  arange = jnp.arange(total_length)[None, :]
  mask = ((arange >= starts[:, None]) & (arange < ends[:, None]))
  full = jnp.where(mask, (arange - offsets[:, None]), 0)
  some_mask = jnp.any(mask, axis=0)
  return jnp.where(some_mask, jnp.sum(full, 0), -1), some_mask


def _padded_group_gather(x: jax.Array, idx: jax.Array, max_idx: int, multiple: int):
  """Gather a tensor into groups according to group idx `idx`, but pad so groups are aligned to `multiple`."""
  counts = jnp.bincount(idx, length=max_idx)
  padding_idxs = add_indices(jnp.arange(max_idx), -counts % multiple, max_size=multiple - 1)
  idx_with_padding = jnp.concat([idx, padding_idxs], axis=0)
  gather_idx = jnp.argsort(idx_with_padding)
  inv_gather_idx = jnp.argsort(gather_idx)
  return (x[gather_idx, ...], inv_gather_idx), (idx, inv_gather_idx)


@partial(jax.custom_vjp, nondiff_argnames=("max_idx", "multiple"))
def padded_group_gather(x: jax.Array, idx: jax.Array, max_idx: int, multiple: int):
  return _padded_group_gather(x, idx, max_idx=max_idx, multiple=multiple)[0]


def padded_group_gather_fwd(x: jax.Array, idx: jax.Array, max_idx: int, multiple: int):
  return _padded_group_gather(x, idx, max_idx=max_idx, multiple=multiple)


def padded_gather_bwd(max_idx: int, multiple: int, res, g):
  del max_idx, multiple
  g = g[0]  # the other sensitivity term belongs to the indices
  (idx, inv_gather_idx) = res
  return g[inv_gather_idx[jnp.argsort(idx)], ...], None


padded_group_gather.defvjp(padded_group_gather_fwd, padded_gather_bwd)
