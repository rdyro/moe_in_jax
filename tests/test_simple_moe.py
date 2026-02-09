import dataclasses
from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from moe.core import run_moe, add_indices, MoEConfig
from moe.ra2a_simulator import ragged_all_to_all as ra2a_via_ag
from .utils import generate_data

try:
  jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
  pass

random_normal = lambda key, shape, dtype: jnp.array(np.random.default_rng(key).normal(size=shape)).astype(dtype)
random_randint = lambda key, shape, minval, maxval: jnp.array(np.random.default_rng(key).integers(
    minval, maxval, size=shape
)).astype(jnp.int32)


class MoeTest(parameterized.TestCase):
  @parameterized.product(
      experts_per_tok=[1, 2, 4], device=["cpu", "tpu"], multiple=[1, 2, 8],
      ra2a=[jax.lax.ragged_all_to_all, ra2a_via_ag], device_num=[1, 4],
  )
  def test_unique_gather_derivative(self, experts_per_tok, device, multiple, ra2a, device_num):
    if device == "cpu" and ra2a != ra2a_via_ag:
      self.skipTest("No jax.lax.ragged_all_to_all on CPU")
    try:
      devices = jax.devices(device)[:device_num]
    except RuntimeError:
      self.skipTest(f"Device {device} not available")
    axis_name = "x"
    mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices)

    with jax.sharding.set_mesh(mesh):
      n, k, g = 1024, 128, 32
      x = jax.random.normal(jax.random.key(0), (n, k), dtype="float32")
      all_idxs = jax.random.randint(jax.random.key(0), (experts_per_tok * x.shape[0],), minval=0, maxval=g)
      x, all_idxs = jax.device_put(x, P(axis_name, None)), jax.device_put(all_idxs, P(None))

      def compute(y, group_sizes):
        # construct dummy weight where the weight is just the expert index
        shard_idx = jax.lax.axis_index(axis_name)
        iota = jax.lax.broadcasted_iota("int32", (y.shape[0], group_sizes.shape[-1]), 0)
        starts, ends = jnp.cumsum(group_sizes) - group_sizes, jnp.cumsum(group_sizes)
        assert (g // len(devices)) == group_sizes.size
        group_idxs = group_sizes.size * shard_idx + jnp.arange(group_sizes.size)
        weights = jnp.sum(((iota >= starts[None, :]) & (iota < ends[None, :])) * group_idxs[None, :], -1)
        return y * jnp.expand_dims(weights, tuple(range(1, y.ndim)))

      # config = MoEConfig(multiple=multiple, ra2a=ra2a_via_ag)
      config = MoEConfig(multiple=multiple, ra2a=ra2a)
      opts = dict(axis_name="x", experts_per_tok=experts_per_tok, experts_num=g, compute_block=compute)
      moe1_fn = jax.jit(partial(run_moe, **opts, config=dataclasses.replace(config, gathers="builtin")))
      moe2_fn = jax.jit(partial(run_moe, **opts, config=dataclasses.replace(config, gathers="custom")))
      o1, vjp1_fn = jax.vjp(partial(moe1_fn, all_idxs), x)
      o2, vjp2_fn = jax.vjp(partial(moe2_fn, all_idxs), x)

      # np.testing.assert_allclose(x, x_new)
      x_ref = jnp.repeat(x, experts_per_tok, axis=0, out_sharding=P(axis_name, None))
      x_ref = x_ref.reshape((x.shape[0], experts_per_tok, x.shape[1]))
      x_ref *= all_idxs.reshape((x.shape[0], experts_per_tok, 1))
      x_ref = jnp.sum(x_ref, 1)
      np.testing.assert_allclose(o1, o2, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(x_ref, o1, atol=1e-5, rtol=1e-5)

      r = jax.jit(
        lambda: jax.random.normal(jax.random.key(1), o1.shape, dtype=x.dtype), out_shardings=P(axis_name, None)
      )()
      (do1,) = vjp1_fn(r)
      (do2,) = vjp2_fn(r)
      do1_error = jnp.max(jnp.linalg.norm(do1 - do2, axis=-1) / jnp.maximum(jnp.linalg.norm(do1, axis=-1), 1e-7))
      self.assertLess(do1_error, 1e-6)

  @parameterized.product(experts_per_tok=[1, 2, 4, 8], device=["cpu", "tpu"])
  def test_simple_moe(self, experts_per_tok, device):
    try:
      devices = jax.devices(device)
    except RuntimeError:
      self.skipTest(f"Device {device} not available")
    axis_name = "x"
    mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices)

    with jax.sharding.set_mesh(mesh):
      n, k, g = 256, 2048, 32
      x = jax.random.normal(jax.random.key(0), (n, k), dtype="bfloat16")
      all_idxs = jax.random.randint(jax.random.key(0), (experts_per_tok * x.shape[0],), minval=0, maxval=g)
      x, all_idxs = jax.device_put(x, P(axis_name, None)), jax.device_put(all_idxs, P(None))
      out = run_moe(all_idxs, x, axis_name="x", experts_per_tok=experts_per_tok, experts_num=g,
                    config=MoEConfig(ra2a=ra2a_via_ag))
      out = out / experts_per_tok
      self.assertEqual(out.shape, (n, x.shape[-1]))

      np.testing.assert_allclose(x, out)

  @parameterized.product(experts=[32, 128], multiple=[2, 4, 8])
  def test_add_indices_works_for_moe(self, experts, multiple):
    all_idxs = jax.random.randint(jax.random.key(0), 128, minval=0, maxval=experts)
    idx_count = jnp.bincount(all_idxs, length=experts)
    pad_indices = add_indices(jnp.arange(experts), -idx_count % multiple, max_size_per_idx=multiple - 1)
    # check if the pad_indices actually added the desired number of pad indices to each group
    np.testing.assert_array_equal(jnp.bincount(pad_indices, length=experts), -idx_count % multiple)

  @parameterized.product(experts_per_tok=[1, 2, 4], device=["cpu", "tpu"], multiple=[1, 2, 8])
  def test_identity_moe_block(self, experts_per_tok, device, multiple):
    try:
      devices = jax.devices(device)
    except RuntimeError:
      self.skipTest(f"Device {device} not available")
    axis_name = "x"
    m, k, g = 4096, 128, 32
    mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices)

    with jax.sharding.set_mesh(mesh):
      all_idxs = jax.random.randint(jax.random.key(0), experts_per_tok * m, minval=0, maxval=g)
      x, ra2a_meta = generate_data(m, k, device_num=len(devices), axis_name=axis_name)
      del ra2a_meta
      config = MoEConfig(multiple=multiple, ra2a=ra2a_via_ag)
      moe_fn = jax.jit(
        partial(run_moe, axis_name=axis_name, experts_per_tok=experts_per_tok, experts_num=g, config=config)
      )
      out = moe_fn(all_idxs, x) / experts_per_tok
      self.assertEqual(out.shape, (m, x.shape[-1]))
      np.testing.assert_allclose(x, out, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
