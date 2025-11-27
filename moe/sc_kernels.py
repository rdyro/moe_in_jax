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


def sc_gather(x, idx, window):
  tpu_info = pltpu.get_tpu_info().sparse_core
  # num_cores, num_subcores, window = tpu_info.num_cores, tpu_info.num_subcores, tpu_info.num_lanes
  num_cores, num_subcores = tpu_info.num_cores, tpu_info.num_subcores
  num_cores = 2
  out_shape = jax.ShapeDtypeStruct((idx.shape[0], *x.shape[1:]), x.dtype)
  # out = 17 * jnp.ones(out_shape.shape, out_shape.dtype)
  out = jax.lax.empty(out_shape.shape, out_shape.dtype)
  x_ref, idx_ref, o_ref = jax.tree.map(jax.new_ref, (x, idx, out))

  @pl.kernel(
      # out_shape=out_shape,
      out_shape=(),
      mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=num_cores
      ),
      scratch_shapes=(
          pltpu.VMEM((window,), jnp.int32),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
          pltpu.VMEM((window, *x.shape[1:]), x.dtype),
      #    pltpu.VMEM((window, x.shape[1] * LANES), dtype),
      ),
  )
  def gather(idx_vmem, x_scratch_ref, scratch_2):  # , o_scratch_ref):
    core_id = jax.lax.axis_index("core")
    subcore_id = jax.lax.axis_index("subcore")
    # @pl.when((subcore_id == 0) & (core_id == 0))
    # def _():
    assert idx_ref.shape[0] % (window * num_subcores * num_cores) == 0
    subcore_slice = idx_ref.shape[0] // (num_subcores * num_cores)

    offset = ((core_id * num_subcores + subcore_id) * subcore_slice) // window
    # @partial(pl.run_scoped, idx_v=pltpu.VMEM(idx_ref.shape, jnp.int32))
    # def _(idx_v):
    #  pltpu.sync_copy(idx_ref, idx_v)

    # def kernel(x_ref, idx_ref, out_ref):
    #   @partial(pl.run_scoped, idx_vmem2=pltpu.VMEM((window,), jnp.int32))
    #   def _(idx_vmem2):
    #     i = pl.program_id(0)
    #     start_i = (offset + i) * window
    #     pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2)
    #     pltpu.sync_copy(x_ref.at[idx_vmem2], out_ref)
    #     #pltpu.sync_copy(x_ref.at[idx_v[pl.ds(start_i, window)]], out_ref)
    #     #pltpu.sync_copy(x_ref.at[idx_ref[...]], out_ref)
    # grid = (subcore_slice // window,)
    # in_specs = [pl.BlockSpec(x_ref.shape, memory_space=pltpu.ANY),
    #             pl.BlockSpec(idx_ref.shape, memory_space=pltpu.ANY)]
    # out_spec = pl.BlockSpec((window,) + o_ref.shape[1:], lambda i: (offset + i,) + (0,) * (o_ref.ndim - 1))
    # pltpu.emit_pipeline(kernel, grid=grid, in_specs=in_specs, out_specs=out_spec, no_pipelining=True)(
    #     x_ref, idx_ref, o_ref)

    # @partial(pl.run_scoped, idx_v=pltpu.VMEM(idx_ref.shape, jnp.int32))
    # def _(idx_v):
    #  pltpu.sync_copy(idx_ref, idx_v)

    @partial(pl.run_scoped, idx_vmem2=pltpu.VMEM((2, window,), jnp.int32), sems=pltpu.SemaphoreType.DMA((6,)))
    def _(idx_vmem2, sems):
      # for i in range(0, subcore_slice, window):
      # @pl.loop(0, subcore_slice, step=64)
      # def _(i):
      #  for j in range(0, 64, window):

      start_i = (core_id * num_subcores + subcore_id) * subcore_slice
      pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2.at[0, ...])
      pltpu.sync_copy(x_ref.at[idx_vmem2.at[0, ...]], scratch_2)
      pltpu.sync_copy(idx_ref.at[pl.ds(start_i + window, window)], idx_vmem2.at[0, ...])

      @pl.loop(0, subcore_slice - window, step=window)
      def _(i):
        for _ in range(1):
          i_ = i
          # i_ = i + j
          start_i = (core_id * num_subcores + subcore_id) * subcore_slice + i_
          start_i = (start_i // window) * window
          # start_i = pl.multiple_of(start_i, window)

          slc = pl.ds(start_i, window)
          nxt_slc = pl.ds(start_i + window, window)

          # pltpu.sync_copy(x_ref.at[idx_v[slc]], x_scratch_ref)

          # with jax.named_scope("indices"):
          #  pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2.at[0, ...])
          # with jax.named_scope("to_vmem"):
          #  pltpu.sync_copy(x_ref.at[idx_vmem2.at[0, ...]], x_scratch_ref.at[0, ...])
          # with jax.named_scope("from_vmem"):
          #  pltpu.sync_copy(x_scratch_ref.at[0, ...], o_ref.at[slc, ...])

          # idx_vmem[1, ...] = idx_vmem[0, ...]
          # x_scratch_ref[1, ...] = x_scratch_ref[0, ...]
          # pltpu.sync_copy(x_scratch_ref.at[0, ...], x_scratch_ref.at[1, ...])

          # copy_idx = pltpu.async_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2.at[0, ...], sems.at[0])
          def ver1():
            with jax.named_scope("ver1"):
              pltpu.sync_copy(idx_ref.at[nxt_slc], idx_vmem2.at[0, ...])
              # copy_idx = pltpu.async_copy(idx_ref.at[nxt_slc], idx_vmem2.at[1, ...], sems.at[0])
              copy_to = pltpu.async_copy(x_ref.at[idx_vmem2.at[0, ...]], x_scratch_ref, sems.at[1])
              copy_from = pltpu.async_copy(scratch_2, o_ref.at[slc, ...], sems.at[2])

            # @pl.loop(0, x_scratch_ref.shape[0], step=1, unroll=True)
            # def _(j):
            #  @pl.loop(0, x_scratch_ref.shape[1], step=1, unroll=True)
            #  def _(k):
            #    @pl.loop(0, x_scratch_ref.shape[2], step=64, unroll=False)
            #    def _(l):
            #      x_scratch_ref[j, k, pl.ds(l, 64)] = scratch_2[j, k, pl.ds(l, 64)]

              # copy_idx.wait()
              copy_to.wait()
              copy_from.wait()

          def ver2():
            with jax.named_scope("ver2"):
              pltpu.sync_copy(idx_ref.at[nxt_slc], idx_vmem2.at[1, ...])
              # copy_idx = pltpu.async_copy(idx_ref.at[nxt_slc], idx_vmem2.at[0, ...], sems.at[3])
              copy_to = pltpu.async_copy(x_ref.at[idx_vmem2.at[1, ...]], scratch_2, sems.at[4])
              copy_from = pltpu.async_copy(x_scratch_ref, o_ref.at[slc, ...], sems.at[5])
              # copy_idx.wait()
              copy_to.wait()
              copy_from.wait()

          jax.lax.cond(jax.lax.rem(i // window, 2) == 0, ver1, ver2)
          # pltpu.sync_copy(x_scratch_ref, o_ref.at[pl.ds((subcore_slice- window), window)])

          # pltpu.sync_copy(x_ref.at[idx_vmem], o_ref.at[pl.ds(start_i, window), ...])

          # pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2)
          # pltpu.sync_copy(x_ref.at[idx_vmem2], o_ref.at[pl.ds(start_i, window), ...])
      print("final2")
      final_idx = (core_id * num_subcores + subcore_id + 1) * subcore_slice - window
      final_idx = (final_idx // window) * window
      pltpu.sync_copy(idx_ref.at[pl.ds(final_idx, window)], idx_vmem2.at[0, ...])
      pltpu.sync_copy(x_ref.at[idx_vmem2.at[0, ...]], scratch_2)
      pltpu.sync_copy(scratch_2, o_ref.at[pl.ds(final_idx, window), ...])
      # pltpu.sync_copy(scratch_2, o_ref.at[pl.ds((subcore_slice- window), window), ...])

  gather()
  return o_ref[...]


def _sc_gather(x, idx):
  tpu_info = pltpu.get_tpu_info().sparse_core
  num_cores, num_subcores, window = tpu_info.num_cores, tpu_info.num_subcores, tpu_info.num_lanes
  num_cores = 2
  out_shape = jax.ShapeDtypeStruct((idx.shape[0], *x.shape[1:]), x.dtype)
  out = 17 * jnp.ones(out_shape.shape, out_shape.dtype)
  x_ref, idx_ref, o_ref = jax.tree.map(jax.new_ref, (x, idx, out))

  @pl.kernel(
      # out_shape=out_shape,
      out_shape=(),
      mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core", subcore_axis_name="subcore", num_cores=num_cores
      ),
      scratch_shapes=(
          pltpu.VMEM((window,), jnp.int32),
          pltpu.VMEM((window, x.shape[1]), x.dtype),
      #    pltpu.VMEM((window, x.shape[1] * LANES), dtype),
      ),
  )
  def gather(idx_vmem, x_scratch_ref):  # , o_scratch_ref):
    core_id = jax.lax.axis_index("core")
    subcore_id = jax.lax.axis_index("subcore")
    # @pl.when((subcore_id == 0) & (core_id == 0))
    # def _():
    assert idx_ref.shape[0] % (window * num_subcores * num_cores) == 0
    subcore_slice = idx_ref.shape[0] // (num_subcores * num_cores)

    offset = ((core_id * num_subcores + subcore_id) * subcore_slice) // window

    @partial(pl.run_scoped, idx_v=pltpu.VMEM(idx_ref.shape[0], jnp.int32))
    def _(idx_v):
      pltpu.sync_copy(idx_ref, idx_v)

      def kernel(out_ref):
        @partial(pl.run_scoped, idx_vmem2=pltpu.VMEM((window,), jnp.int32))
        def _(idx_vmem2):
          i = pl.program_id(0)
          start_i = (offset + i) * window
          # pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2)
          # pltpu.sync_copy(x_ref.at[idx_vmem2], out_ref)
          pltpu.sync_copy(x_ref.at[idx_v[pl.ds(start_i, window)]], out_ref)
          # pltpu.sync_copy(x_ref.at[idx_ref[...]], out_ref)

      grid = (subcore_slice // window,)
      in_specs = [
        # pl.BlockSpec((window,), lambda i: offset + i)
      ]
      out_spec = pl.BlockSpec((window,) + o_ref.shape[1:], lambda i: (offset + i, 0))
      pltpu.emit_pipeline(kernel, grid=grid, in_specs=in_specs, out_specs=out_spec)(o_ref)

    # @pl.loop(0, subcore_slice, step=window)
    # def _(i):
    #  @partial(pl.run_scoped, idx_vmem2=pltpu.VMEM((window,), jnp.int32))
    #  def _(idx_vmem2):
    #    start_i = (core_id * num_subcores + subcore_id) * subcore_slice + i
    #    start_i = (start_i // window) * window
    #    #start_i = pl.multiple_of(start_i, window)
    #    #pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2)
    #    #pltpu.sync_copy(x_ref.at[idx_vmem2], x_scratch_ref)
    #    #pltpu.sync_copy(x_scratch_ref, o_ref.at[pl.ds(start_i, window), ...])
    #    #pltpu.sync_copy(x_ref.at[idx_vmem], o_ref.at[pl.ds(start_i, window), ...])

    #    pltpu.sync_copy(idx_ref.at[pl.ds(start_i, window)], idx_vmem2)
    #    pltpu.sync_copy(x_ref.at[idx_vmem2], o_ref.at[pl.ds(start_i, window), ...])

  gather()
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
