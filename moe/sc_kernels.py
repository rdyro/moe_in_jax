import math
from functools import partial
from typing import Literal

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.experimental.pallas.tpu_sc as plsc


def kernel_with_scratch(x_ref, idx_ref, out_ref, x_scratch_ref):
  rows = 1
  chunk = 8
  cols = 8
  pltpu.sync_copy(x_ref.at[idx_ref[...]], x_scratch_ref)

  @pl.loop(0, x_scratch_ref.shape[0], step=rows)
  def _(j):
    @pl.loop(0, x_scratch_ref.shape[1], step=1)
    def _(k):
      @pl.loop(0, x_scratch_ref.shape[2], step=chunk)
      def _(l):
        j_ = k * x_scratch_ref.shape[2] + l
        # out_ref[pl.ds(j, rows), pl.ds(j_, chunk)] = x_scratch_ref[
        #   pl.ds(j, rows), pl.ds(k, 1), pl.ds(l, chunk)
        # ].reshape((rows, -1))
        # out_ref[rows, pl.ds(j_, cols * chunk)] = x_scratch_ref[rows, k, pl.ds(l, cols * chunk)].reshape((-1,))
        out_ref[rows, pl.ds(j_, cols * chunk)] = x_scratch_ref[rows, k, pl.ds(l, cols * chunk)].reshape((-1,))


def kernel(x_ref, idx_ref, out_ref):
  tiling = ((8, 128), (1, 1))
  pl.run_scoped(
    partial(kernel_with_scratch, x_ref, idx_ref, out_ref),
    x_scratch_ref=plsc.MemoryRef(
      (idx_ref.shape[0], *x_ref.shape[1:]), x_ref.dtype, memory_space=pltpu.VMEM, tiling=tiling
    ),
  )


@jax.jit
def gather_3d_to_2d(x, idx):
  rows = 8
  out_shape = jax.ShapeDtypeStruct((idx.shape[0], math.prod(x.shape[1:])), x.dtype)
  in_specs = [pl.BlockSpec(memory_space=pltpu.HBM), pl.BlockSpec((rows,), lambda i: (i,))]
  out_specs = pl.BlockSpec((rows, out_shape.shape[1]), lambda i: (i, 0))
  grid = (pl.cdiv(x.shape[0], rows),)
  dimension_semantics: list[Literal["arbitrary", "parallel"]] = ["arbitrary"]
  return pl.pallas_call(
    kernel,
    out_shape=out_shape,
    in_specs=in_specs,
    out_specs=out_specs,
    grid=grid,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=dimension_semantics, kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
    ),
  )(x, idx)
