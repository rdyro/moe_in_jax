
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.experimental.pallas.tpu as pltpu
import numpy as np

from moe.sc_kernels import sc_gather, sc_scatter

random_normal = lambda key, shape, dtype: jnp.array(np.random.default_rng(key).normal(size=shape)).astype(dtype)
random_randint = lambda key, shape, minval, maxval: jnp.array(np.random.default_rng(key).integers(
    minval, maxval, size=shape
)).astype(jnp.int32)


SPARSECORE_SUPPORT = [
    pltpu.ChipVersion.TPU_V5E, pltpu.ChipVersion.TPU_V5P, pltpu.ChipVersion.TPU_V6E, pltpu.ChipVersion.TPU_7X,
]


class SparseCoreTest(parameterized.TestCase):
  def setUp(self):
    try:
      self.devices = jax.devices("tpu")
      tpu_info = pltpu.get_tpu_info()
      if tpu_info.chip_version not in SPARSECORE_SUPPORT:
        raise ValueError(f"Sparse core not supported on TPU `{tpu_info.chip_version}`")
    except (RuntimeError, ValueError):
      self.devices = None

  @parameterized.product(align_lanes=[True, False], m=[4 * 17 * 256, 4096 * 8], k=[1024, 2048, 4096, 7168], m_mult=[1, 1.5, 2])
  def test_sc_gather(self, align_lanes, m, k, m_mult):
    if self.devices is None:
      self.skipTest("Sparse core not supported")
    #m = 4096 * 8
    m_final = ((round(m_mult * m) + 128 - 1) // 128) * 128

    x = random_normal(0, (m, k), "bfloat16")
    all_idxs = jnp.argsort(random_normal(0, (m_final,), "bfloat16")) % m

    x = x.reshape((x.shape[0], -1, 128)) if align_lanes else x.reshape((x.shape[0], 8, -1))
    y = jax.jit(sc_gather)(x, all_idxs)
    y_ref = x[all_idxs, ...]
    np.testing.assert_array_equal(y, y_ref)

  @parameterized.product(align_lanes=[True, False], k=[1024, 2048, 4096, 7168], m_mult=[1, 1.5, 2])
  def test_sc_scatter(self, align_lanes, k, m_mult):
    if self.devices is None:
      self.skipTest("Sparse core not supported")
    m = 4096 * 8
    m_final = ((round(m_mult * m) + 128 - 1) // 128) * 128

    x = random_normal(0, (m, k), "bfloat16")
    all_idxs = jnp.argsort(random_normal(0, (m_final,), "bfloat16"))[:m]

    x = x.reshape((x.shape[0], -1, 128)) if align_lanes else x.reshape((x.shape[0], 8, -1))

    out = 17 * jnp.ones((m_final, *x.shape[1:]), "bfloat16")  # use a filled buffer to avoid comparing nan == nan

    y = jax.jit(sc_scatter)(out, all_idxs, x)
    y_ref = jax.jit(lambda out, idx, x: out.at[idx, ...].set(x))(out, all_idxs, x)
    np.testing.assert_array_equal(y, y_ref)


if __name__ == "__main__":
  absltest.main()
