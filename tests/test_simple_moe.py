from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from moe.core import run_moe, add_indices
from moe.ra2a_simulator import ragged_all_to_all as cpu_ra2a
from .utils import generate_data

try:
  jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
  pass


class MoeTest(parameterized.TestCase):
  @parameterized.product(experts_per_tok=[1, 2, 4], device=["cpu", "tpu"], multiple=[1, 2, 8])
  def test_custom_gather_derivative(self, experts_per_tok, device, multiple):
    try:
      devices = jax.devices(device)
    except RuntimeError:
      self.skipTest(f"Device {device} not available")
    axis_name = "x"
    mesh = jax.make_mesh((len(devices),), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices)

    with jax.sharding.set_mesh(mesh):
      n, k, g = 256, 128, 32
      x = jax.random.normal(jax.random.key(0), (n, k), dtype="bfloat16")
      all_idxs = jax.random.randint(jax.random.key(0), (experts_per_tok * x.shape[0],), minval=0, maxval=g)
      x, all_idxs = jax.device_put(x, P(axis_name, None)), jax.device_put(all_idxs, P(None))
      opts = dict(axis_name="x", experts_num=g, ragged_all_to_all=cpu_ra2a, multiple=multiple)
      moe1_fn = partial(run_moe, **opts, custom_gathers=False)
      moe2_fn = partial(run_moe, **opts, custom_gathers=True)
      o1, vjp1_fn = jax.vjp(partial(moe1_fn, all_idxs=all_idxs), x)
      o2, vjp2_fn = jax.vjp(partial(moe2_fn, all_idxs=all_idxs), x)

      x_new = np.array(o1[:, 0, :])
      np.testing.assert_allclose(x, x_new)
      np.testing.assert_allclose(o1, o2)

      r = jax.jit(lambda: jax.random.normal(jax.random.key(1), o1.shape, dtype="bfloat16"),
                  out_shardings=P(axis_name, None, None))()
      do1 = vjp1_fn(r)
      do2 = vjp2_fn(r)
      np.testing.assert_allclose(do1, do2)

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
      # x, ra2a_meta = generate_data(n, k, len(devices), axis_name="x")
      # del ra2a_meta
      x = jax.random.normal(jax.random.key(0), (n, k), dtype="bfloat16")
      all_idxs = jax.random.randint(jax.random.key(0), (experts_per_tok * x.shape[0],), minval=0, maxval=g)
      x, all_idxs = jax.device_put(x, P(axis_name, None)), jax.device_put(all_idxs, P(None))
      out = run_moe(x, all_idxs, axis_name="x", experts_num=g, ragged_all_to_all=cpu_ra2a)
      self.assertEqual(out.shape, (n, experts_per_tok, x.shape[-1]))

      x_new = np.array(out[:, 0, :])
      np.testing.assert_allclose(x, x_new)

  @parameterized.product(experts=[32, 128], multiple=[2, 4, 8])
  def test_add_indices_works_for_moe(self, experts, multiple):
    all_idxs = jax.random.randint(jax.random.key(0), 128, minval=0, maxval=experts)
    idx_count = jnp.bincount(all_idxs, length=experts)
    pad_indices = add_indices(jnp.arange(experts), -idx_count % multiple, max_size=multiple - 1)
    # check if the pad_indices actually added the desired number of pad indices to each group
    np.testing.assert_array_equal(jnp.bincount(pad_indices, length=experts), -idx_count % multiple)

  @parameterized.product(experts=[32, 128], multiple=[1, 2, 4, 8])
  def test_identity_moe_block(self, experts, multiple):
    n_devices = jax.device_count()
    axis_name = "x"
    m, k = 4096, 2048
    mesh = jax.make_mesh((n_devices,), (axis_name,), axis_types=(jax.sharding.AxisType.Explicit,))

    with jax.sharding.set_mesh(mesh):
      all_idxs = jax.random.randint(jax.random.key(0), experts * m, minval=0, maxval=experts)
      x, ra2a_meta = generate_data(m, k, device_num=n_devices, axis_name=axis_name)
      del ra2a_meta
      reduce_block = lambda x: x[:, 0, ...]
      moe_fn = jax.jit(partial(
          run_moe, reduce_block=reduce_block, axis_name=axis_name, experts_num=experts, multiple=multiple
      ))
      out = moe_fn(x, all_idxs)
      self.assertEqual(out.shape, (m, x.shape[-1]))
      np.testing.assert_array_equal(out, x)


if __name__ == "__main__":
  absltest.main()
