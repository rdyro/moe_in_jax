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
  # first shorter description
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
