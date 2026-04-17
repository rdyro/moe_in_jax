import dataclasses
from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.experimental import xla_metadata
import numpy as np
from scipy.special import softmax
import tune_jax
tune_jax.logger.setLevel("INFO")

from aligned_sort import compute_padded_group_gather

def random_group_sizes(g: int, m: int):
  gs = np.round(m * softmax(np.random.randn(g))).astype(np.int32)
  while np.sum(gs) != m:
    idx = np.argmax(gs)
    gs[idx] = np.maximum(0, m - np.sum(gs) + gs[idx])
  return jnp.array(gs)

MULTIPLE = 8
multof = lambda x: (x // MULTIPLE) * MULTIPLE


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GroupMetadata:
  rhs_group_idx: jax.Array
  lhs_tile_offset: jax.Array
  lhs_tile_sizes: jax.Array
  actual_tile_number: jax.Array

def new_make_group_metadata(*, group_sizes: jax.Array, m: int, tm: int) -> GroupMetadata:
  MAX_INT = 2 ** 31 - 1
  visits_per_group = (group_sizes + tm - 1) // tm
  group_ends = jnp.cumsum(visits_per_group)
  group_starts = group_ends - visits_per_group
  max_grid_size = (m + tm - 1) // tm + group_sizes.size + 1
  iota = jnp.arange(max_grid_size)
  groups_mask = ((iota[:, None] >= group_starts[None, :]) & (iota[:, None] < group_ends[None, :]))
  rhs_group_idx = jnp.sum(groups_mask * jnp.arange(group_sizes.size)[None, :], -1)

  group_offsets = jnp.cumsum(group_sizes) - group_sizes
  lhs_tile_offset = tm * (jnp.arange(max_grid_size)[:, None] - group_starts[None, :]) + group_offsets[None, :]
  lhs_tile_offset = jnp.sum(groups_mask * lhs_tile_offset, -1)
  lhs_tile_sizes = jnp.ones(max_grid_size, dtype=jnp.int32) * tm
  truncated_tile_mask = (group_sizes > 0) & (group_sizes % tm != 0)
  last_tile_idx = jnp.where(truncated_tile_mask, group_ends - 1, MAX_INT)
  lhs_tile_sizes = lhs_tile_sizes.at[last_tile_idx].set(group_sizes % tm, mode="drop")
  rhs_group_idx = jnp.where(jnp.arange(max_grid_size) < group_ends[-1], rhs_group_idx, MAX_INT)
  actual_tile_number = group_ends[-1]
  return GroupMetadata(rhs_group_idx, lhs_tile_offset, lhs_tile_sizes, actual_tile_number[None])

@partial(jax.jit, static_argnames=["tiling"])
def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, tiling: tuple[int, int, int] = (128, 128, 128)):
  m, k = lhs.shape
  n = rhs.shape[-1]
  tile_m, tile_k, tile_n = tiling
  metadata = new_make_group_metadata(group_sizes=group_sizes, m=lhs.shape[0], tm=tiling[0])
  cdiv = lambda a, b: (a + b - 1) // b
  grid = (metadata.actual_tile_number[0], cdiv(n, tile_n), cdiv(k, tile_k))

  metadata_hbm_ref = jax.tree.map(jax.new_ref, metadata)

  lhs_ref, rhs_ref = jax.new_ref(lhs), jax.new_ref(rhs)
  # out_ref = jax.new_ref(jax.lax.empty((m, n), dtype=lhs.dtype))
  out_ref = jax.new_ref(jnp.zeros((m, n), dtype=lhs.dtype))

  metadata_spec = jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), metadata)
  metadata_spec_flat, metadata_spec_tree = jax.tree.flatten(metadata_spec)
  acc_scratch = pltpu.VMEM((tile_m, tile_n), jnp.float32)
  metadata_copy_sem = pltpu.SemaphoreType.DMA((len(metadata_spec_flat),))

  @pl.core_map(mesh=pltpu.create_tensorcore_mesh("core"))
  def _():
    @partial(pl.run_scoped, metadata_smem_ref=metadata_spec_flat, acc_scratch=acc_scratch, metadata_copy_sem=metadata_copy_sem)
    def _(metadata_smem_ref, acc_scratch, metadata_copy_sem):
      copies = [pltpu.async_copy(src_ref, dst_ref, metadata_copy_sem.at[i])
                for i, (src_ref, dst_ref)
                in enumerate(zip(jax.tree.leaves(metadata_hbm_ref), metadata_smem_ref))]
      for copy in copies:
        copy.wait()
      metadata_smem_ref: GroupMetadata = jax.tree.unflatten(metadata_spec_tree, metadata_smem_ref)

      def lhs_index_map(i, j, k):
        return (pl.ds(multof(metadata_smem_ref.lhs_tile_offset[i]), multof(metadata_smem_ref.lhs_tile_sizes[i])), k)

      def rhs_index_map(i, j, k):
        return (metadata_smem_ref.rhs_group_idx[i], k, j)

      def out_index_map(i, j, k):
        return (pl.ds(multof(metadata_smem_ref.lhs_tile_offset[i]), multof(metadata_smem_ref.lhs_tile_sizes[i])), j)

      lhs_spec = pl.BlockSpec((pl.BoundedSlice(tile_m), tile_k), lhs_index_map)
      out_spec = pl.BlockSpec((pl.BoundedSlice(tile_m), tile_n), out_index_map)
      rhs_spec = pl.BlockSpec((None, tile_k, tile_n), rhs_index_map)

      def kernel_body(lhs_ref, rhs_ref, out_ref):
        pid = namedtuple("size", ["i", "j", "k"])(*[pl.program_id(i) for i in range(3)])

        @pl.when(pid.k == 0)
        def _():
          acc_scratch[...] = jnp.zeros_like(acc_scratch)

        acc_scratch[...] += pl.dot(lhs_ref[...], rhs_ref[...]).astype(acc_scratch.dtype)

        @pl.when(pid.k == cdiv(k, tile_k) - 1)
        def _():
          out_ref[...] = acc_scratch[...].astype(out_ref.dtype)

      pltpu.emit_pipeline(kernel_body, grid=grid, in_specs=[lhs_spec, rhs_spec], out_specs=out_spec)(
        lhs_ref, rhs_ref, out_ref)

  return out_ref[...]


########################################################################################################################
# Tests ################################################################################################################
########################################################################################################################

if __name__ == "__main__":
  keys = iter(jax.random.split(jax.random.key(0), 1024))
  m, k, n = 4 * 8 * 4096, 7168, 2048
  gs = random_group_sizes(256, m)
  group_idx = jax.random.randint(next(keys), (m,), 0, gs.size)
  padded_gather = jax.jit(compute_padded_group_gather, static_argnames=("multiple", "num_groups"))(group_idx, gs.size, multiple=MULTIPLE)
  keys = iter(jax.random.split(jax.random.key(0), 1024))
  lhs = jax.random.normal(next(keys), (m, k))
  rhs = jax.random.normal(next(keys), (gs.size, k, n))


  lhs = lhs[padded_gather.sort_idx, ...]
  lhs = jnp.pad(lhs, ((0, 512 - (lhs.shape[0] % 512)), (0, 0)))
  gs = padded_gather.group_counts_with_padding
  assert jnp.all(gs % MULTIPLE == 0)


  hyperparams = {
    "tile_m": [128, 256, 512],
    "tile_k": [7168, 7168 // 2, 7168 // 4, 7168 // 8],
    "tile_n": [512, 1024, 2048],
  }

  @partial(jax.jit, static_argnames=("tile_m", "tile_k", "tile_n"))
  def gmm_(lhs, rhs, group_sizes, tile_m, tile_k, tile_n):
    return gmm(lhs, rhs, group_sizes, tiling=(tile_m, tile_k, tile_n))

  @partial(jax.jit, static_argnames=("tile_m", "tile_k", "tile_n"))
  def gmm2_(lhs, rhs, group_sizes, tile_m, tile_k, tile_n):
    with xla_metadata.set_xla_metadata(ragged_dot_tiling=f"{tile_m},{tile_k},{tile_n}"):
      return jax.lax.ragged_dot(lhs, rhs, group_sizes)

  fn = tune_jax.tune(gmm_, hyperparams=hyperparams)
  out1 = fn(lhs, rhs, gs)
  print(tune_jax.tabulate(fn))
  fn2 = tune_jax.tune(gmm2_, hyperparams=hyperparams)
  out2 = fn2(lhs, rhs, gs)
  print(tune_jax.tabulate(fn2))
  breakpoint()

if False:

  print("Launching the kernel", flush=True)
  out = gmm(lhs, rhs, gs)
  out_ref = jax.lax.ragged_dot(lhs, rhs, padded_gather.group_counts_with_padding)
  print("Done running, attempting to print the value", flush=True)
  print(out)
  print(out[:, 0].reshape((-1, 8)))
  breakpoint()

if False and __name__ == "__main__":
  m = 4096
  gs = random_group_sizes(32, m)
  print(gs)

  tm = 16
  rhs_group_idx, lhs_tile_offset, lhs_tile_sizes = new_make_group_metadata(group_sizes=gs, m=m, tm=tm)
  # print(rhs_group_idx)
  assert jnp.all(jnp.bincount(rhs_group_idx, length=gs.size) == (gs + tm - 1) // tm)
  print(jnp.bincount(rhs_group_idx, length=gs.size))

  print(lhs_tile_offset)
  print(lhs_tile_sizes)
