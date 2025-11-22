from functools import partial
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from moe.ra2a_simulator import ragged_all_to_all as _ragged_all_to_all
from .utils import generate_data

try:
  jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
  print("jax_num_cpu_devies already set")


class UtilsTests(parameterized.TestCase):
  @parameterized.product(key=[0, 1, 2], device_num=[4, 8])
  def test_simple_moe(self, key, device_num):
    devices = jax.devices()
    if len(devices) < device_num:
      self.skipTest("Not enough devices")
    axis_name = "x"
    mesh = jax.make_mesh(
      (device_num,), (axis_name,), axis_types=jax.sharding.AxisType.Explicit, devices=devices[:device_num]
    )
    with jax.sharding.set_mesh(mesh):
      x, meta = generate_data(64, 128, device_num, axis_name=axis_name, key=key)
      out = jax.device_put(jnp.zeros_like(x, shape=(2 * x.shape[0],) + x.shape[1:]), P(axis_name, None))

      @partial(jax.shard_map, out_specs=P(axis_name, None))
      def fn(x, out, meta):
        assert x.ndim == 2
        out1 = jax.lax.ragged_all_to_all(x, out, *dataclasses.astuple(meta), axis_name=axis_name)
        out2 = _ragged_all_to_all(x, out, *dataclasses.astuple(meta), axis_name=axis_name)
        return out1, out2

      out1, out2 = fn(x, out, meta)
      np.testing.assert_array_equal(out1, out2)


if __name__ == "__main__":
  absltest.main()
