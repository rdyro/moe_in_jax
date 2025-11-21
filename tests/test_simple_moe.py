from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from moe.core import run_moe, add_indices
from .utils import generate_data


class MoeTest(parameterized.TestCase):
  @parameterized.product(experts_per_tok=[1, 2, 4, 8])
  def test_simple_moe(self, experts_per_tok):
    n_devices = jax.device_count()
    axis_name = "x"
    mesh = jax.make_mesh((n_devices,), (axis_name,), axis_types=(jax.sharding.AxisType.Explicit,))

    with jax.sharding.set_mesh(mesh):
      n, k = 16, 2048
      g = 32  # experts
      x, ra2a_meta = generate_data(n, k, n_devices, axis_name="x")
      del ra2a_meta
      all_idxs = jax.random.randint(jax.random.key(0), (experts_per_tok * x.shape[0],), minval=0, maxval=g)
      out = run_moe(x, all_idxs, axis_name="x", experts_num=g)
      self.assertEqual(out.shape, (n, experts_per_tok, x.shape[-2], x.shape[-1]))

      x_new = np.array(out[:, 0, :, :])
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
      self.assertEqual(out.shape, (m, x.shape[-2], x.shape[-1]))
      np.testing.assert_array_equal(out, x)


if __name__ == "__main__":
  absltest.main()
