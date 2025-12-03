import dataclasses
import math
import os
from functools import partial
from typing import Any, Callable, Literal

LIBTPU_INIT_ARGS = os.environ.get("LIBTPU_INIT_ARGS", "").split(" ")
os.environ["LIBTPU_INIT_ARGS"] = " ".join([
    "--xla_tpu_use_tc_device_shape_on_sc=true"
] + LIBTPU_INIT_ARGS)

import jax  # noqa: E402
import jax.experimental.pallas as pl  # noqa: E402
import jax.experimental.pallas.tpu as pltpu  # noqa: E402
import jax.experimental.pallas.tpu_sc as plsc  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental.compute_on import compute_on  # noqa: E402

########################################################################################################################
########################################################################################################################
########################################################################################################################

SPARSECORE_PAD_SIZE = 1024

AsyncCopyDescriptor = Any


@dataclasses.dataclass
class RDMACopy:
  copy: AsyncCopyDescriptor | None
  start: Callable[[], None]
  wait: Callable[[], None]


multiple_of = lambda a, b: (a // b) * b


@partial(jax.jit, static_argnames=("axis_name", "multiple"))
def ra2a_sc(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name: str = "x", multiple: int):
  n_devices = jax.lax.axis_size(axis_name)

  src_ref, dst_ref = jax.tree.map(jax.new_ref, (src, output))

  pad_multiple = 32  # sparsecore specific for SMEM
  size_pad = (((4 * input_offsets.size + pad_multiple - 1) // pad_multiple) * pad_multiple) // 4  # pad to 32 bytes
  pad_to_size = lambda x: jnp.pad(x, ((0, size_pad - x.size),))

  (input_offsets, send_sizes, output_offsets, recv_sizes) = jax.tree.map(
      pad_to_size, (input_offsets, send_sizes, output_offsets, recv_sizes)
  )
  input_offsets_ref, send_sizes_ref, output_offsets_ref, recv_sizes_ref = jax.tree.map(
      jax.new_ref, (input_offsets, send_sizes, output_offsets, recv_sizes)
  )
  cost_estimate = pl.CostEstimate(
    flops=0, transcendentals=0,
    bytes_accessed=2 * (src.size * src.itemsize),
    remote_bytes_transferred=src.size * src.itemsize,
  )

  @pl.kernel(
      out_shape=(),
      mesh=plsc.ScalarSubcoreMesh(axis_name='core', num_cores=1),
      scratch_shapes=(
          pltpu.SMEM((2 * n_devices,), jnp.int32),
          pltpu.SMEM((2 * n_devices,), jnp.int32),
          pltpu.SMEM((2 * n_devices,), jnp.int32),
          pltpu.SMEM((2 * n_devices,), jnp.int32),
          pltpu.SemaphoreType.DMA((n_devices, 2, 2)),
      ),
      cost_estimate=cost_estimate,
  )
  def _ra2a_2d_kernel_sync(input_offsets, send_sizes, output_offsets, recv_sizes, sems):
    # core_id, subcore_id = jax.lax.axis_index("core"), jax.lax.axis_index("subcore")
    core_id, subcore_id = jax.lax.axis_index("core"), 0
    idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)

    @pl.when((core_id == 0) & (subcore_id == 0))
    def _():
      pltpu.sync_copy(input_offsets_ref, input_offsets)
      pltpu.sync_copy(send_sizes_ref, send_sizes)
      pltpu.sync_copy(output_offsets_ref, output_offsets)
      pltpu.sync_copy(recv_sizes_ref, recv_sizes)

      def make_dma(id):
        sem_id = lax.rem(idx + idx, n_devices)
        src = src_ref.at[pl.ds(multiple_of(input_offsets[id], multiple), multiple_of(send_sizes[id], multiple)), ...]
        dst = dst_ref.at[pl.ds(multiple_of(output_offsets[id], multiple), multiple_of(send_sizes[id], multiple)), ...]
        copy = pltpu.make_async_copy(src, dst, sems.at[0, sem_id, 1])
        return RDMACopy(copy, copy.start, copy.wait)

      dma_copy = make_dma(idx)
      dma_copy.start()

      def make_rdma(other_id, send: bool = True):
        src_id, dst_id = (idx, other_id) if send else (other_id, idx)
        size = lax.select(idx == src_id, send_sizes[dst_id], recv_sizes[src_id])
        src = src_ref.at[pl.ds(multiple_of(input_offsets[dst_id], multiple), multiple_of(size, multiple)), ...]
        dst = dst_ref.at[pl.ds(multiple_of(output_offsets[dst_id], multiple), multiple_of(size, multiple)), ...]
        sem_id, direction_id = lax.rem(idx + other_id, n_devices), (src_id > dst_id).astype(jnp.int32)
        send_sem = sems.at[sem_id, direction_id, 0]
        recv_sem = sems.at[sem_id, direction_id, 1]
        copy = pltpu.make_async_remote_copy(src, dst, send_sem, recv_sem, device_id=dst_id)
        start_fn = copy.start if send else (lambda: None)
        wait_fn = copy.wait_send if send else copy.wait_recv
        return RDMACopy(copy, start_fn, wait_fn)

      send_rdmas, recv_rdmas = [], []
      for i in range(1, n_devices):
        other_id = jax.lax.rem(idx + i, n_devices)
        send_rdmas.append(make_rdma(other_id, send=True))
        recv_rdmas.append(make_rdma(other_id, send=False))
      [rdma.start() for rdma in send_rdmas]
      [rdma.wait() for rdma in (send_rdmas + recv_rdmas)]
      dma_copy.wait()

  _ra2a_2d_kernel_sync()
  return dst_ref[...]


@partial(jax.custom_vjp, nondiff_argnames=("axis_name", "multiple"))
def ra2a(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name: str, multiple: int):
  return ra2a_sc(src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
                 axis_name=axis_name, multiple=multiple)


def ra2a_fwd(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name: str, multiple: int):
  out = ra2a_sc(src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
                axis_name=axis_name, multiple=multiple)
  res = (input_offsets, send_sizes, output_offsets, recv_sizes, src.shape)
  return out, res


def ra2a_bwd(axis_name: str, multiple: int, res, g):
  (input_offsets, send_sizes, output_offsets, recv_sizes, src_shape) = res
  buf = jax.lax.empty(src_shape, dtype=g.dtype)
  inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
  inv_input_offsets = jax.lax.all_to_all(output_offsets, axis_name, split_axis=0, concat_axis=0)
  inv_output_offsets = jax.lax.all_to_all(input_offsets, axis_name, split_axis=0, concat_axis=0)
  dsrc = ra2a_sc(g, buf, inv_input_offsets, inv_send_sizes, inv_output_offsets, inv_recv_sizes,
                 axis_name=axis_name, multiple=multiple)
  return (dsrc, *[None for _ in range(1 + 4)])


ra2a.defvjp(ra2a_fwd, ra2a_bwd)


########################################################################################################################
########################################################################################################################
########################################################################################################################

# (gather -> scatter-add)
@jax.custom_vjp
def nonunique_sc_gather(x, idx):
  return sc_gather(x, idx)


def nonunique_sc_gather_fwd(x, idx):
  return sc_gather(x, idx), (x.shape, idx)


def nonunique_sc_gather_bwd(res, g):
  x_shape, idx = res
  return jnp.zeros(x_shape, dtype=g.dtype).at[idx, ...].set(g, out_sharding=jax.typeof(g).sharding), None


nonunique_sc_gather.defvjp(nonunique_sc_gather_fwd, nonunique_sc_gather_bwd)


# (gather, gather/scatter)
@partial(jax.custom_vjp, nondiff_argnames=("ad_mode", "empty_for_scatter"))
def unique_sc_gather(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, ad_mode: str, empty_for_scatter: bool = True):
  assert ad_mode in ("gather", "scatter")
  # return x[idx, ...]
  return sc_gather(x, idx)


def unique_sc_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, ad_mode: str, empty_for_scatter: bool):
  static = dict(ad_mode=ad_mode, empty_for_scatter=empty_for_scatter)
  return unique_sc_gather(x, idx, inv_idx, **static), (x.shape, inv_idx,)


def unique_sc_gather_bwd(ad_mode: str, empty_for_scatter: bool, res, g):
  (x_shape, inv_idx,) = res
  if ad_mode == "gather":
    grad = sc_gather(g, inv_idx)
  else:  # scatter
    # TODO(rdyro): check if this gather optimization actually outperforms scatter
    if g.shape[0] == x_shape[0]:  # shortcut if input/output shape matches
      grad = sc_gather(g, jnp.argsort(inv_idx))
    else:  # otherwise really use scatter
      buf = jax.lax.empty(x_shape, dtype=g.dtype) if empty_for_scatter else jnp.zeros(x_shape, dtype=g.dtype)
      grad = sc_scatter(buf, inv_idx, g)
  return (grad, None, None)


unique_sc_gather.defvjp(unique_sc_gather_fwd, unique_sc_gather_bwd)

########################################################################################################################
########################################################################################################################
########################################################################################################################


def sc_gather(x, idx, window: int | None = None):
  tpu_info = pltpu.get_tpu_info().sparse_core
  num_cores, num_subcores = tpu_info.num_cores, tpu_info.num_subcores
  window = tpu_info.num_lanes if window is None else window
  out_shape = jax.ShapeDtypeStruct((idx.shape[0], *x.shape[1:]), x.dtype)
  out = jax.lax.empty(out_shape.shape, out_shape.dtype)
  x_ref, idx_ref, o_ref = jax.tree.map(jax.new_ref, (x, idx, out))

  assert idx_ref.size % 1024 == 0, f"{idx_ref.size=} must be divisible by 1024"

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

  assert idx_ref.size % 1024 == 0, f"{idx_ref.size=} must be divisible by 1024"

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


########################################################################################################################
########################################################################################################################
########################################################################################################################

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

########################################################################################################################
########################################################################################################################
########################################################################################################################
