import math
from functools import partial
import os
from typing import Literal

LIBTPU_INIT_ARGS = os.environ.get("LIBTPU_INIT_ARGS", "").split(" ")
os.environ["LIBTPU_INIT_ARGS"] = " ".join([
    "--xla_tpu_use_tc_device_shape_on_sc=true"
] + LIBTPU_INIT_ARGS)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.experimental.pallas as pl  # noqa: E402
import jax.experimental.pallas.tpu as pltpu  # noqa: E402
import jax.experimental.pallas.tpu_sc as plsc  # noqa: E402
from jax.experimental.compute_on import compute_on  # noqa: E402


def sc_gather(x, idx, window: int | None = None):
  tpu_info = pltpu.get_tpu_info().sparse_core
  num_cores, num_subcores = tpu_info.num_cores, tpu_info.num_subcores
  window = tpu_info.num_lanes if window is None else window
  out_shape = jax.ShapeDtypeStruct((idx.shape[0], *x.shape[1:]), x.dtype)
  out = jax.lax.empty(out_shape.shape, out_shape.dtype)
  x_ref, idx_ref, o_ref = jax.tree.map(jax.new_ref, (x, idx, out))

  @pl.kernel(
      out_shape=(),
      mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=num_cores
      ),
      scratch_shapes=(
          pltpu.VMEM((window,), jnp.int32),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
          pltpu.SemaphoreType.DMA((4,)),
      ),
  )
  def _gather(idx_vmem, scratch1_ref, scratch2_ref, sems):
    core_id, subcore_id = jax.lax.axis_index("core"), jax.lax.axis_index("subcore")
    assert idx_ref.shape[0] % (window * num_subcores * num_cores) == 0
    subcore_slice = idx_ref.shape[0] // (num_subcores * num_cores)
    offset = (core_id * num_subcores + subcore_id) * subcore_slice

    # prologue
    start_i = (offset // window) * window
    pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem)
    pltpu.sync_copy(x_ref.at[idx_vmem], scratch2_ref)

    @pl.loop(0, subcore_slice - window, step=window)
    def _(i):
      start_i = ((offset + i) // window) * window
      slc, next_slc = pl.ds(start_i, window), pl.ds(start_i + window, window)

      def stage1():
        with jax.named_scope("stage1"):
          pltpu.sync_copy(idx_ref.at[next_slc], idx_vmem)
          copy_to = pltpu.async_copy(x_ref.at[idx_vmem], scratch1_ref, sems.at[0])
          copy_from = pltpu.async_copy(scratch2_ref, o_ref.at[slc, ...], sems.at[1])
          copy_to.wait()
          copy_from.wait()

      def stage2():
        with jax.named_scope("stage2"):
          pltpu.sync_copy(idx_ref.at[next_slc], idx_vmem)
          copy_to = pltpu.async_copy(x_ref.at[idx_vmem], scratch2_ref, sems.at[2])
          copy_from = pltpu.async_copy(scratch1_ref, o_ref.at[slc, ...], sems.at[3])
          copy_to.wait()
          copy_from.wait()

      jax.lax.cond(jax.lax.rem(i // window, 2) == 0, stage1, stage2)

    # epilogue
    start_i = ((offset + subcore_slice - window) // window) * window
    pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem)
    pltpu.sync_copy(x_ref.at[idx_vmem], scratch2_ref)
    pltpu.sync_copy(scratch2_ref, o_ref.at[pl.ds(start_i, window), ...])

  _gather()
  return o_ref[...]


def sc_scatter(out: jax.Array, idx: jax.Array, x: jax.Array, window: int | None = None):
  tpu_info = pltpu.get_tpu_info().sparse_core
  num_cores, num_subcores = tpu_info.num_cores, tpu_info.num_subcores
  window = tpu_info.num_lanes if window is None else window
  o_ref, idx_ref, x_ref = jax.tree.map(jax.new_ref, (out, idx, x))

  @pl.kernel(
      out_shape=(),
      mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=num_cores
      ),
      scratch_shapes=(
          pltpu.VMEM((window,), jnp.int32),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
          pltpu.SemaphoreType.DMA((4,)),
      ),
  )
  def _scatter(idx_vmem, scratch1_ref, scratch2_ref, sems):
    core_id, subcore_id = jax.lax.axis_index("core"), jax.lax.axis_index("subcore")
    assert idx_ref.shape[0] % (window * num_subcores * num_cores) == 0
    subcore_slice = idx_ref.shape[0] // (num_subcores * num_cores)
    offset = (core_id * num_subcores + subcore_id) * subcore_slice

    # prologue
    start_i = (offset // window) * window
    pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem)
    pltpu.sync_copy(x_ref.at[pl.ds(start_i, window), ...], scratch2_ref)

    @pl.loop(0, subcore_slice - window, step=window)
    def _(i):
      start_i = ((offset + i) // window) * window
      slc, next_slc = pl.ds(start_i, window), pl.ds(start_i + window, window)

      def stage1():
        with jax.named_scope("stage1"):
          pltpu.sync_copy(idx_ref.at[slc], idx_vmem)
          copy_to = pltpu.async_copy(x_ref.at[next_slc, ...], scratch1_ref, sems.at[0])
          copy_from = pltpu.async_copy(scratch2_ref, o_ref.at[idx_vmem], sems.at[1])
          copy_to.wait()
          copy_from.wait()

      def stage2():
        with jax.named_scope("stage2"):
          pltpu.sync_copy(idx_ref.at[slc], idx_vmem)
          copy_to = pltpu.async_copy(x_ref.at[next_slc, ...], scratch2_ref, sems.at[2])
          copy_from = pltpu.async_copy(scratch1_ref, o_ref.at[idx_vmem], sems.at[3])
          copy_to.wait()
          copy_from.wait()

      jax.lax.cond(jax.lax.rem(i // window, 2) == 0, stage1, stage2)

    # epilogue
    start_i = ((offset + subcore_slice - window) // window) * window
    pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem)
    pltpu.sync_copy(x_ref.at[pl.ds(start_i, window)], scratch2_ref)
    pltpu.sync_copy(scratch2_ref, o_ref.at[idx_vmem])

  _scatter()
  return o_ref[...]


@compute_on("tpu_sparsecore")
@jax.jit
def scatter(xshape, idx, val):
  return jax.lax.empty(xshape, val.dtype).at[idx, ...].set(val, out_sharding=jax.typeof(val).sharding)


@compute_on("tpu_sparsecore")
@jax.jit
def gather(x, idx):
  return x.at[idx, ...].get(out_sharding=jax.typeof(x).sharding)


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
