import dataclasses
from typing import Callable, Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu


AsyncCopyDescriptor = Any


@dataclasses.dataclass
class RDMACopy:
  copy: AsyncCopyDescriptor | None
  start: Callable[[], None]
  wait: Callable[[], None]


# synchronous ra2a 2D kernel ###########################################################################################


def _ra2a_2d_kernel_sync(
  src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems, *, axis_name
):
  del out_ref  # aliased in dst_ref
  idx, n_devices = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
  raise NotImplementedError("This is a 3D version, it needs to be adapted to 2D.")

  def make_dma(id):
    sem_id = lax.rem(idx + idx, n_devices)
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
def ra2a(src, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name: str = "x"):
  raise NotImplementedError("This is a 3D version, it needs to be adapted to 2D.")

  n_devices = jax.lax.axis_size(axis_name)
  out = pl.pallas_call(
    partial(_ra2a_2d_kernel_sync, axis_name=axis_name),
    out_shape=output,
    in_specs=2 * [pl.BlockSpec(memory_space=pltpu.ANY)] + 4 * [pl.BlockSpec(memory_space=pltpu.SMEM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
    scratch_shapes=[pltpu.SemaphoreType.DMA((n_devices, 2, 2))],
    input_output_aliases={1: 0},
    interpret=False,
  )(src, output, input_offsets, send_sizes, output_offsets, recv_sizes)
  return out


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
  axis_name,
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
      kws = dict(axis_name=axis_name, start=True)
      return _ra2a_3d_kernel_async(
        src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems, **kws
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
    n_devices = jax.lax.axis_size(axis_name)
    sems = future

    def ra2a_kernel_wait(src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, sems, dst_ref):
      kws = dict(axis_name=axis_name, start=False)
      return _ra2a_3d_kernel_async(
        src_ref, out_ref, input_offsets, send_sizes, output_offsets, recv_sizes, dst_ref, sems, **kws
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
