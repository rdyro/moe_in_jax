
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np


from moe.core import run_moe
from .utils import generate_data


class MoeTest(parameterized.TestCase):

  @parameterized.product(
    experts_per_tok=[1, 2, 4, 8]
  )
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
      

if __name__ == "__main__":
  absltest.main()