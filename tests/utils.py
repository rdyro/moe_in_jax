from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import auto_axes, PartitionSpec as P


@partial(jax.jit, static_argnames=("n", "k"))
def generate_data(n, k, device_num):
  x = auto_axes(lambda: jnp.tile(jnp.arange(n)[:, None, None], (1, 8, k // 8)), out_sharding=P("x", None, None))()
  idx = random.randint(random.key(0), shape=(n,), minval=0, maxval=device_num)

  @partial(jax.shard_map, in_specs=(P("x", None, None), P(None)), out_specs=(P("x", None, None),) + (P("x"),) * 4)
  def fn(x, idx):
    id = jax.lax.axis_index("x")
    local_idx = jax.lax.dynamic_slice_in_dim(idx, id * x.shape[0], x.shape[0], axis=0)

    sizes = jax.vmap(lambda idx: jnp.bincount(idx, length=device_num))(idx.reshape((device_num, -1)))

    send_sizes = jnp.take_along_axis(sizes, id[None, None], axis=0)[0, :]
    recv_sizes = jnp.take_along_axis(sizes, id[None, None], axis=1)[:, 0]

    input_offsets = jnp.concat([jnp.zeros((1,), send_sizes.dtype), jnp.cumsum(send_sizes)[:-1]])

    output_offsets = jnp.concat([jnp.zeros((1, sizes.shape[0]), sizes.dtype), jnp.cumsum(sizes, 0)])[:-1, :]
    output_offsets = jnp.take_along_axis(output_offsets, id[None, None], axis=0)[0, :]

    x_sort = jnp.take_along_axis(x, jnp.argsort(local_idx)[:, None, None], 0)
    return x_sort, input_offsets, send_sizes, output_offsets, recv_sizes

  return fn(x, idx)
