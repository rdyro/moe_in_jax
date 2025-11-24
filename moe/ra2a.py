import dataclasses
from functools import partial
from typing import Any, Callable

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

AsyncCopyDescriptor = Any


@dataclasses.dataclass
class RDMACopy:
  copy: AsyncCopyDescriptor | None
  start: Callable[[], None]
  wait: Callable[[], None]


# synchronous ra2a 2D kernel ###########################################################################################

multiple_of = lambda a, b: (a // b) * b


def _ra2a_2d_kernel_sync(src_ref,
                         # out_ref,
                         input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems,
                         *, axis_name: str, multiple: int):
  # del out_ref  # aliased in dst_ref
  idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
  # raise NotImplementedError("This is a 3D version, it needs to be adapted to 2D.")

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


@partial(jax.jit, static_argnames=("axis_name", "multiple"))
def ra2a_2d(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name: str = "x", multiple: int):
  n_devices = jax.lax.axis_size(axis_name)
  # del output
  out = pl.pallas_call(
    partial(_ra2a_2d_kernel_sync, axis_name=axis_name, multiple=multiple),
    out_shape=jax.ShapeDtypeStruct(output.shape, output.dtype),
    # in_specs=2 * [pl.BlockSpec(memory_space=pltpu.ANY)] + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)],
    in_specs=1 * [pl.BlockSpec(memory_space=pltpu.ANY)] + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
    scratch_shapes=[pltpu.SemaphoreType.DMA((n_devices, 2, 2))],
    # input_output_aliases={1: 0},
    interpret=False,
  )(src,
    # output,
    input_offsets, send_sizes, output_offsets, recv_sizes)
  return out


@partial(jax.custom_vjp, nondiff_argnames=("axis_name", "multiple"))
def ra2a(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name: str, multiple: int):
  return ra2a_2d(src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
                 axis_name=axis_name, multiple=multiple)


def ra2a_fwd(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name: str, multiple: int):
  out = ra2a_2d(src, output, input_offsets, send_sizes, output_offsets, recv_sizes,
                axis_name=axis_name, multiple=multiple)
  res = (input_offsets, send_sizes, output_offsets, recv_sizes, src.shape)
  return out, res


def ra2a_bwd(axis_name: str, multiple: int, res, g):
  (input_offsets, send_sizes, output_offsets, recv_sizes, src_shape) = res
  buf = jax.lax.empty(src_shape, dtype=g.dtype)
  inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
  inv_input_offsets = jax.lax.all_to_all(output_offsets, axis_name, split_axis=0, concat_axis=0)
  inv_output_offsets = jax.lax.all_to_all(input_offsets, axis_name, split_axis=0, concat_axis=0)
  dsrc = ra2a_2d(g, buf, inv_input_offsets, inv_send_sizes, inv_output_offsets, inv_recv_sizes,
                 axis_name=axis_name, multiple=multiple)
  return (dsrc, *[None for _ in range(1 + 4)])


ra2a.defvjp(ra2a_fwd, ra2a_bwd)


# asynchronous ra2a 3D kernel ##########################################################################################


def _ra2a_3d_kernel_async(
  src_ref,
  out_ref,
  input_offsets,
  send_sizes,
  output_offsets,
  recv_sizes,
  dst_ref,
  sems,
  *,
  axis_name: str,
  start: bool = True,
):
  del out_ref  # aliased in dst_ref
  idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)

  def make_dma(id):
    sem_id = lax.rem(idx + idx, n_devices)
    src = src_ref.at[pl.ds(input_offsets[id], send_sizes[id]), ...]
    dst = dst_ref.at[pl.ds(output_offsets[id], send_sizes[id]), ...]
    copy = pltpu.make_async_copy(src, dst, sems.at[0, sem_id, 1])
    return RDMACopy(copy, copy.start, copy.wait)

  dma_copy = make_dma(idx)
  if start:
    dma_copy.start()

  def make_rdma(other_id, send: bool = True):
    src_id, dst_id = (idx, other_id) if send else (other_id, idx)
    size = lax.select(idx == src_id, send_sizes[dst_id], recv_sizes[src_id])
    src = src_ref.at[pl.ds(input_offsets[dst_id], size), ...]
    dst = dst_ref.at[pl.ds(output_offsets[dst_id], size), ...]
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

  if start:
    [rdma.start() for rdma in (send_rdmas + recv_rdmas)]
  else:
    [rdma.wait() for rdma in (send_rdmas + recv_rdmas)]
    dma_copy.wait()


def make_ra2a_3d(axis_name: str = "x"):
  def start(src, output, input_offsets, send_sizes, output_offsets, recv_sizes):
    n_devices = jax.lax.axis_size(axis_name)

    def ra2a_kernel_start(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems):
      return _ra2a_3d_kernel_async(
        src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems,
        axis_name=axis_name, start=True,
      )

    sems_spec = pltpu.SemaphoreType.DMA((n_devices, 2, 2))
    out, sems = pl.pallas_call(
      ra2a_kernel_start,
      out_shape=[output, sems_spec],
      in_specs=2 * [pl.BlockSpec(memory_space=pltpu.ANY)] + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)],
      out_specs=[pl.BlockSpec(memory_space=pltpu.ANY), pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
      input_output_aliases={1: 0},
      interpret=False,
    )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes)
    return out, sems

  def wait(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, future):
    sems = future

    def ra2a_kernel_wait(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, sems, dst_ref):
      return _ra2a_3d_kernel_async(
        src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems,
        axis_name=axis_name, start=False
      )

    # sems_spec = pltpu.SemaphoreType.DMA((n_devices, 2, 2))
    out = pl.pallas_call(
      ra2a_kernel_wait,
      out_shape=output,
      in_specs=2 * [pl.BlockSpec(memory_space=pltpu.ANY)]
      + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)]
      + [pl.BlockSpec(memory_space=pltpu.SEMAPHORE)],
      out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
      input_output_aliases={1: 0},
      interpret=False,
    )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, sems)
    return out

  return start, wait


# synchronous ra2a 3D kernel ###########################################################################################


def _ra2a_3d_kernel_sync(
  src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems, *, axis_name
):
  del out_ref  # aliased in dst_ref
  idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)

  def make_dma(id):
    sem_id = lax.rem(idx + idx, n_devices)
    # src = src_ref.at[pl.ds(input_offsets[id], send_sizes[id]), ...]
    # dst = dst_ref.at[pl.ds(output_offsets[id], send_sizes[id]), ...]
    src = src_ref.at[pl.ds(input_offsets[id], send_sizes[id]), ...]
    dst = dst_ref.at[pl.ds(output_offsets[id], send_sizes[id]), ...]
    copy = pltpu.make_async_copy(src, dst, sems.at[0, sem_id, 1])
    return RDMACopy(copy, copy.start, copy.wait)

  dma_copy = make_dma(idx)
  dma_copy.start()

  def make_rdma(other_id, send: bool = True):
    src_id, dst_id = (idx, other_id) if send else (other_id, idx)
    size = lax.select(idx == src_id, send_sizes[dst_id], recv_sizes[src_id])
    src = src_ref.at[pl.ds(input_offsets[dst_id], size), ...]
    dst = dst_ref.at[pl.ds(output_offsets[dst_id], size), ...]
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


@partial(jax.jit, static_argnames=("axis_name",))
def ra2a_3d(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name: str = "x"):
  n_devices = jax.lax.axis_size(axis_name)
  out = pl.pallas_call(
    partial(_ra2a_3d_kernel_sync, axis_name=axis_name),
    out_shape=output,
    in_specs=2 * [pl.BlockSpec(memory_space=pltpu.ANY)] + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
    scratch_shapes=[pltpu.SemaphoreType.DMA((n_devices, 2, 2))],
    input_output_aliases={1: 0},
    interpret=False,
  )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes)
  return out
