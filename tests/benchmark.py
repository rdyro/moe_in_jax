import os
import time
from functools import partial

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=''"  # let named_scopes show up on GPU

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import moe

try:
  jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError:
  print("CPU devices already set")


def main():
  axis_name = "x"
  g = 128
  m = 4096
  experts_per_tok = 8
  embed = 7168
  multiple = 8

  keys = iter(jax.random.split(jax.random.key(0), 1024))

  x: jax.Array
  x = jax.jit(lambda key: jax.random.normal(key, (m, embed), dtype="bfloat16"),
              out_shardings=P(axis_name, None))(next(keys))
  all_idxs = jax.jit(lambda key: jax.random.randint(key, (experts_per_tok * x.shape[0],), minval=0, maxval=g),
                     out_shardings=P(None))(next(keys))
  r = jax.jit(lambda key: jax.random.normal(key, (x.shape[0], experts_per_tok, x.shape[1]), dtype=x.dtype),
              out_shardings=x.sharding)(next(keys))

  def compute(y, group_sizes):
    # construct dummy weight where the weight is just the expert index
    shard_idx = jax.lax.axis_index(axis_name)
    iota = jax.lax.broadcasted_iota("int32", (y.shape[0], group_sizes.shape[-1]), 0)
    starts, ends = jnp.cumsum(group_sizes) - group_sizes, jnp.cumsum(group_sizes)
    assert (g // len(devices)) == group_sizes.size
    group_idxs = group_sizes.size * shard_idx + jnp.arange(group_sizes.size)
    weights = jnp.sum(((iota >= starts[None, :]) & (iota < ends[None, :])) * group_idxs[None, :], -1)
    return y * weights[:, None]

  # ra2a_fn = partial(moe.ra2a.ra2a, multiple=multiple)
  # ra2a_fn = jax.lax.ragged_all_to_all
  ra2a_fn = moe.ra2a_simulator.ragged_all_to_all
  opts = dict(axis_name="x", experts_num=g, multiple=multiple, ragged_all_to_all=ra2a_fn, compute_block=compute)

  run_moe = partial(moe.core.run_moe, **opts)
  run_moe2 = partial(moe.core.run_moe, **opts, custom_gathers=True)

  run_moe_jit = jax.jit(run_moe)
  run_moe2_jit = jax.jit(run_moe2)

  @jax.jit
  def vjp_jit(x, all_idxs, r):
    return jax.vjp(partial(run_moe, all_idxs=all_idxs), x)[1](r)

  @jax.jit
  def vjp2_jit(x, all_idxs, r):
    return jax.vjp(partial(run_moe2, all_idxs=all_idxs), x)[1](r)

  y1 = jax.block_until_ready(run_moe_jit(x, all_idxs))
  y2 = jax.block_until_ready(run_moe2_jit(x, all_idxs))
  (do1,) = jax.block_until_ready(vjp_jit(x, all_idxs, r))
  (do2,) = jax.block_until_ready(vjp2_jit(x, all_idxs, r))

  with moe.utils.profile():
    for _ in range(2):
      jax.block_until_ready(run_moe_jit(x, all_idxs))
    for _ in range(2):
      jax.block_until_ready(run_moe2_jit(x, all_idxs))
    for _ in range(2):
      jax.block_until_ready(vjp_jit(x, all_idxs, r))
    for _ in range(2):
      jax.block_until_ready(vjp2_jit(x, all_idxs, r))

  y_err = jnp.sum(jnp.abs(y1 - y2) != 0)
  do_err = jnp.max(jnp.linalg.norm(do1 - do2, axis=-1) / jnp.maximum(jnp.linalg.norm(do1, axis=-1), 1e-7))

  @jax.jit
  def compute_ref(x, all_idxs):
    y_ref = jnp.repeat(x, experts_per_tok, axis=0, out_sharding=P(axis_name))
    y_ref *= jnp.expand_dims(all_idxs, tuple(range(1, x.ndim)))
    y_ref = y_ref.reshape((x.shape[0], experts_per_tok, *x.shape[1:]))
    return y_ref
    # return jnp.abs(y1 - y_ref)

  print(f"y_err =    {int(y_err):d}")
  print(f"do_err =   {float(do_err):.4e}")
  print(f"y_id_err = {float(jnp.sum(jnp.abs(compute_ref(x, all_idxs) - y1))):.4e}")


if __name__ == "__main__":
  devices = jax.devices()
  axis_name = "x"
  mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=(jax.sharding.AxisType.Explicit,), devices=devices)
  with jax.sharding.set_mesh(mesh):
    main()
  try:
    while True:
      time.sleep(10)
  except KeyboardInterrupt:
    pass
