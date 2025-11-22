from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import auto_axes, PartitionSpec as P

from moe.core import RA2AMeta


def balance_indices(indices, n, multiple):
  """Minimally reassigns indices so per-device counts are divisible by `multiple`."""
  p = jnp.argsort(indices)
  cum_counts = jnp.cumsum(jnp.bincount(indices, length=n))
  targets = jnp.diff(jnp.round(cum_counts / multiple) * multiple, prepend=0).astype(int)
  return jnp.empty_like(indices).at[p].set(
    jnp.sum(jnp.arange(indices.shape[0]) >= jnp.cumsum(targets)[:-1][:, None], axis=0)
  )


@partial(jax.jit, static_argnames=("n", "k", "device_num", "multiple", "axis_name"))
def generate_data(n, k, device_num, *, multiple: int = 1, axis_name: str, key: int | jax.Array = 0):
  """Generate synthetic data and routing metadata for a non-uniform all-to-all communication."""
  x = auto_axes(
    lambda: jnp.tile(jnp.arange(n, dtype=jnp.bfloat16)[:, None], (1, k)),
    out_sharding=P(axis_name, None)
  )()
  key = jax.random.key(key) if jnp.asarray(key).ndim == 0 else key
  idx = random.randint(key, shape=(n,), minval=0, maxval=device_num)
  if multiple != 1:
    idx = idx.reshape((device_num, -1))
    idx = jax.vmap(partial(balance_indices, n=n, multiple=multiple))(idx).reshape(-1)

  @partial(jax.shard_map, in_specs=(P(axis_name, None), P(None)), out_specs=(P(axis_name, None), (P(axis_name))))
  def fn(x, idx):
    id = jax.lax.axis_index(axis_name)
    local_idx = jax.lax.dynamic_slice_in_dim(idx, id * x.shape[0], x.shape[0], axis=0)

    sizes = jax.vmap(lambda idx: jnp.bincount(idx, length=device_num))(idx.reshape((device_num, -1)))

    send_sizes = jnp.take_along_axis(sizes, id[None, None], axis=0)[0, :]
    recv_sizes = jnp.take_along_axis(sizes, id[None, None], axis=1)[:, 0]

    input_offsets = jnp.concat([jnp.zeros((1,), send_sizes.dtype), jnp.cumsum(send_sizes)[:-1]])

    output_offsets = jnp.concat([jnp.zeros((1, sizes.shape[0]), sizes.dtype), jnp.cumsum(sizes, 0)])[:-1, :]
    output_offsets = jnp.take_along_axis(output_offsets, id[None, None], axis=0)[0, :]

    # x_sort = jnp.take_along_axis(x, jnp.argsort(local_idx)[:, None], 0)
    x_sort = x[jnp.argsort(local_idx), ...]
    return x_sort, RA2AMeta(input_offsets, send_sizes, output_offsets, recv_sizes)

  return fn(x, idx)
