import dataclasses
import os
from functools import partial
import gc
import time

os.environ["LIBTPU_INIT_ARGS"] = " ".join([
    "--xla_tpu_enable_offloading_gather_to_sparsecore=true",
    "--xla_tpu_enable_offloading_scatter_to_sparsecore=true",
    "--xla_tpu_offload_all_supported_gathers_to_sparsecore=true",
    "--xla_tpu_offload_gather_to_sparsecore=true",
    "--xla_tpu_offload_all_supported_gathers_to_sparsecore=true",
])
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=''"  # let named_scopes show up on GPU

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.xla_metadata import set_xla_metadata

import moe

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

try:
  jax.config.update("jax_num_cpu_devices", 4)
except RuntimeError:
  print("CPU devices already set")


def generate_expert_matrices(g, k, n, axis_name):
  # w1 = jnp.ones((g, k, n), dtype="bfloat16", out_sharding=P(axis_name, None, None)) / (k + n)
  # w2 = jnp.ones((g, k, n), dtype="bfloat16", out_sharding=P(axis_name, None, None)) / (k + n)
  # w3 = jnp.ones((g, n, k), dtype="bfloat16", out_sharding=P(axis_name, None, None)) / (n + k)
  keys = iter(jax.random.split(jax.random.key(7), 1024))
  w1 = jax.random.normal(next(keys), (g, k, n), dtype="bfloat16", out_sharding=P(axis_name)) / (k + n) ** 0.5
  w2 = jax.random.normal(next(keys), (g, k, n), dtype="bfloat16", out_sharding=P(axis_name)) / (k + n) ** 0.5
  w3 = jax.random.normal(next(keys), (g, n, k), dtype="bfloat16", out_sharding=P(axis_name)) / (n + k) ** 0.5
  return w1, w2, w3


def compute_block(y, group_sizes, w1, w2, w3):
  # jax.debug.print("group_sizes = {}, pct of total = {}%", group_sizes, jnp.sum(group_sizes) / y.shape[0] * 1e2)
  y_shape = y.shape
  y = y.reshape((y.shape[0], -1))
  with set_xla_metadata(ragged_dot_tiling="1024,1024,1024"):
    y1 = jax.lax.ragged_dot(y, w1, group_sizes)
    y2 = jax.lax.ragged_dot(y, w2, group_sizes)
    y3 = y1 * jax.nn.gelu(y2, approximate=True)
    y4 = jax.lax.ragged_dot(y3, w3, group_sizes)
  return y4.reshape(y_shape)


def compute_block_simple(y, group_sizes, *, g: int):
  # construct dummy weight where the weight is just the expert index
  shard_idx = jax.lax.axis_index(axis_name)
  iota = jax.lax.broadcasted_iota("int32", (y.shape[0], group_sizes.shape[-1]), 0)
  starts, ends = jnp.cumsum(group_sizes) - group_sizes, jnp.cumsum(group_sizes)
  assert (g // len(devices)) == group_sizes.size
  group_idxs = group_sizes.size * shard_idx + jnp.arange(group_sizes.size)
  weights = jnp.sum(((iota >= starts[None, :]) & (iota < ends[None, :])) * group_idxs[None, :], -1)
  return y * jnp.expand_dims(weights, tuple(range(1, y.ndim)))


def main(devices):
  axis_name = "x"
  g = 32
  m = 4096 * 8 * len(devices)
  experts_per_tok = 8
  embed = 7168
  x_shape = (m, 8, embed // 8)
  multiple = 1

  keys = iter(jax.random.split(jax.random.key(0), 1024))

  x: jax.Array
  x = jax.jit(lambda key: jax.random.normal(key, x_shape, dtype="bfloat16"), out_shardings=P(axis_name))(next(keys))
  all_idxs = jax.jit(
      lambda key: jax.random.randint(key, (experts_per_tok * x.shape[0],), minval=0, maxval=g),
      out_shardings=P())(next(keys)
  )
  r = jax.jit(lambda key: jax.random.normal(key, x.shape, dtype=x.dtype), out_shardings=P(axis_name))(next(keys))

  config = moe.core.MoEConfig(ra2a=jax.lax.ragged_all_to_all, multiple=multiple)
  # config = moe.core.MoEConfig(ra2a=moe.ra2a_simulator.ragged_all_to_all, multiple=multiple)
  # config = moe.core.MoEConfig(ra2a=partial(moe.ra2a.ra2a, multiple=multiple), multiple=multiple)

  opts = dict(axis_name="x", experts_per_tok=experts_per_tok, experts_num=g, compute_block=compute_block)
  config2 = dataclasses.replace(config, gathers="custom")
  run_moe = partial(moe.core.run_moe, **opts, config=config)
  run_moe2 = partial(moe.core.run_moe, **opts, config=config2)
  run_moe3 = partial(moe.pipelined.run_moe_pipelined, **opts, config=config2, splits=4)

  def make_jit(run_moe_fn):
    # return jax.jit(lambda *args: jnp.sum(run_moe_fn(*args), tuple(range(1, x.ndim))),
    #               in_shardings=(P(), *[P(axis_name)] * 4))
    return jax.jit(run_moe_fn, in_shardings=(P(), *[P(axis_name)] * 4))

  run_moe_jit = make_jit(run_moe)
  run_moe2_jit = make_jit(run_moe2)
  run_moe3_jit = make_jit(run_moe3)

  extra_args = generate_expert_matrices(g, k=embed, n=2048, axis_name=axis_name)

  def make_vjp_jit(run_moe_fn):
    @partial(jax.jit, in_shardings=(P(), *[P(axis_name)] * 5))
    def vjp_jit(all_idxs, x, r, *extra_args):
      o, vjp_fn = jax.vjp(partial(run_moe_fn, all_idxs), x, *extra_args)
      do = vjp_fn(r)
      return (o, tuple(range(1, o.ndim))), do
    return vjp_jit

  vjp_jit = make_vjp_jit(run_moe_jit)
  vjp2_jit = make_vjp_jit(run_moe2_jit)
  vjp3_jit = make_vjp_jit(run_moe3_jit)

  y1 = jax.block_until_ready(run_moe_jit(all_idxs, x, *extra_args))
  y2 = jax.block_until_ready(run_moe2_jit(all_idxs, x, *extra_args))
  y3 = jax.block_until_ready(run_moe3_jit(all_idxs, x, *extra_args))

  (_, do1) = jax.block_until_ready(vjp_jit(all_idxs, x, r, *extra_args))
  (_, do2) = jax.block_until_ready(vjp2_jit(all_idxs, x, r, *extra_args))
  (_, do3) = jax.block_until_ready(vjp3_jit(all_idxs, x, r, *extra_args))

  y_err = jnp.sum(jnp.abs(y1 - y2) != 0)
  y_err3 = jnp.sum(jnp.abs(y1 - y3) != 0)
  print(f"y_err =  {jax.tree.map(float, y_err)}")
  print(f"y_err3 = {jax.tree.map(float, y_err3)}")

  error_fn = lambda x, y: jnp.linalg.norm(x - y, axis=tuple(range(1, x.ndim))) / jnp.maximum(
    jnp.linalg.norm(y, axis=tuple(range(1, y.ndim))), 1e-7
  )
  do_err = jax.tree.map(jnp.max, jax.tree.map(error_fn, do1, do2))
  do_err3 = jax.tree.map(jnp.max, jax.tree.map(error_fn, do1, do3))
  print(f"do_err =  {jax.tree.map(float, do_err)}")
  print(f"do_err3 = {jax.tree.map(float, do_err3)}")
  del y_err, y_err3, do1, do2, do3, y1, y2, y3, do_err, do_err3
  [gc.collect() for _ in range(3)]

  with moe.utils.profile():
    for _ in range(2):
      jax.block_until_ready(run_moe_jit(all_idxs, x, *extra_args))
    for _ in range(2):
      jax.block_until_ready(run_moe2_jit(all_idxs, x, *extra_args))
    for _ in range(2):
      jax.block_until_ready(run_moe3_jit(all_idxs, x, *extra_args))
    for _ in range(2):
      jax.block_until_ready(vjp_jit(all_idxs, x, r, *extra_args))
    for _ in range(2):
      jax.block_until_ready(vjp2_jit(all_idxs, x, r, *extra_args))
    for _ in range(2):
      jax.block_until_ready(vjp3_jit(all_idxs, x, r, *extra_args))

  print("#" * 80)
  print("#" * 80)
  print("#" * 80)


if __name__ == "__main__":
  devices = jax.devices()
  axis_name = "x"
  mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices)
  with jax.sharding.set_mesh(mesh):
    main(devices)
  try:
    while True:
      time.sleep(10)
  except KeyboardInterrupt:
    pass
