import contextlib
import dataclasses
import os
from functools import partial
from pathlib import Path
from subprocess import Popen
from typing import Any, Callable, TypeVar

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import psutil

zip_ = zip
zip = partial(zip_, strict=True)

T = TypeVar("T")


def register_jax_dataclass(cls: T | None = None, *, meta_fields: list[str] | None = []) -> T | Callable[[T], T]:
  """Register a dataclass with jax.tree_util, but only by specifying meta fields, data fields are implicit."""

  def _register_fn(cls) -> T:
    meta_fields_ = meta_fields or []
    if not dataclasses.is_dataclass(cls):
      cls = dataclasses.dataclass(cls)
    data_fields = [x.name for x in dataclasses.fields(cls) if x.name not in meta_fields_]
    return jax.tree_util.register_dataclass(cls, meta_fields=meta_fields_, data_fields=data_fields)

  return _register_fn(cls) if cls is not None else _register_fn


@register_jax_dataclass
@dataclasses.dataclass
class RA2AMeta:
  """Holds sizes and offsets for ragged all-to-all communication."""
  input_offsets: jax.Array
  send_sizes: jax.Array
  output_offsets: jax.Array
  recv_sizes: jax.Array


def tpu_sublane_size():
  try:
    return pltpu.get_tpu_info().num_sublanes
  except ValueError:  # we don't have TPU
    return 1


@partial(jax.jit, static_argnames=("pad_size",))
def padding_to_tile(sort_idx, group_sizes, pad_size: int = 128):
  """Align a sort_idx groups to a multiple of the pad_size after the gather.

  The sort_idx is considered split into groups by the entries in group_sizes. This function modified sort_idx
  to produce groups padded to a multiple of the pad_size after the gather with sort_idx.
  """
  padded_group_sizes = pad_size * ((group_sizes + pad_size - 1) // pad_size)
  insert_indices = jnp.cumsum(padded_group_sizes) - padded_group_sizes
  get_indices = jnp.cumsum(group_sizes) - group_sizes

  def fn(i, new_sort_idx):
    slice = jnp.where(jnp.arange(sort_idx.size) < group_sizes[i], jnp.roll(sort_idx, -get_indices[i]), 0)
    new_sort_idx = jax.lax.dynamic_update_index_in_dim(new_sort_idx, slice, insert_indices[i], axis=-1)
    return new_sort_idx

  new_sort_idx = jax.lax.fori_loop(0, group_sizes.size, fn, jnp.zeros(sort_idx.shape[0] * 2, sort_idx.dtype))
  worst_case_size = sort_idx.size + group_sizes.size * (pad_size - 1)
  return new_sort_idx[:worst_case_size], padded_group_sizes


def scatter_arange(total_length: jax.Array, counts: jax.Array, multiple: int = 4):
  """Creates a dilated arange corresponding to group sizes (counts) padded to `multiple`."""
  new_counts = counts + (-counts % multiple)
  offsets = (jnp.cumsum(new_counts) - new_counts) - (jnp.cumsum(counts) - counts)
  starts = jnp.cumsum(new_counts) - new_counts
  ends = starts + counts
  arange = jnp.arange(total_length)[None, :]
  mask = ((arange >= starts[:, None]) & (arange < ends[:, None]))
  full = jnp.where(mask, (arange - offsets[:, None]), 0)
  some_mask = jnp.any(mask, axis=0)
  return jnp.where(some_mask, jnp.sum(full, 0), -1), some_mask


@partial(jax.jit, static_argnames=("max_size_per_idx",))
def add_indices(idx_list: jax.Array, sizes: jax.Array, max_size_per_idx: int, fill_value: int = 2 ** 31 - 1):
  """One way to add indices from a list in desired counts, filling the rest with a fill value."""
  start_idx, end_idx = jnp.cumsum(sizes) - sizes, jnp.cumsum(sizes)
  iota = jnp.arange(idx_list.size * max_size_per_idx)[None, :]
  mask = (iota >= start_idx[:, None]) & (iota < end_idx[:, None])
  # the result is a list having sizes[0] of idx_list[0], sizes[1] of idx_list[1] and so on
  # the unfilled values are filled with fill_value
  return jnp.sum(idx_list[:, None] * mask, axis=0) + ~jnp.any(mask, axis=0) * fill_value


@register_jax_dataclass
@dataclasses.dataclass
class PaddedGroupPaddedMetadata:
  group_idx: jax.Array
  group_idx_with_padding: jax.Array
  group_counts: jax.Array
  group_counts_with_padding: jax.Array
  sort_idx: jax.Array
  isort_idx: jax.Array


def compute_padded_group_gather(group_idx: jax.Array, num_groups: int, multiple: int,
                                group_counts: jax.Array | None = None) -> PaddedGroupPaddedMetadata:
  """Compute metadata for sorting tokens according to group_idx with padding to make groups divisible by `multiple`."""

  assert multiple >= 1
  if group_counts is None:
    group_counts = jnp.bincount(group_idx, length=num_groups)
  else:
    assert group_counts.size == num_groups

  if multiple != 1:
    padding_idxs = add_indices(jnp.arange(num_groups), -group_counts % multiple, max_size_per_idx=multiple - 1)
    group_idx_with_padding = jnp.concat([group_idx, padding_idxs], axis=0)
    group_counts_with_padding = group_counts + (-group_counts % multiple)
  else:
    group_idx_with_padding, group_counts_with_padding = group_idx, group_counts
  sort_idx = jnp.argsort(group_idx_with_padding)
  isort_idx = jnp.argsort(sort_idx)[:group_idx.size]

  return PaddedGroupPaddedMetadata(
    group_idx, group_idx_with_padding, group_counts, group_counts_with_padding, sort_idx, isort_idx
  )


@partial(jax.custom_vjp, nondiff_argnames=("empty_for_scatter", "mode"))
def unique_gather(
  x: jax.Array, idx: jax.Array, inv_idx: jax.Array, empty_for_scatter: bool = True, mode: str = "default"
):
  assert mode in ("default", "padded_gather")
  """Gather (unique indices): Backwards pass is gather/scatter, avoiding costly scatter-add."""
  return x[idx, ...]


def unique_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, empty_for_scatter: bool, mode: str):
  return unique_gather(x, idx, inv_idx, empty_for_scatter=empty_for_scatter), (x.shape, idx, inv_idx)


def unique_gather_bwd(empty_for_scatter: bool, mode: str, res: tuple[Any, jax.Array], g: jax.Array):
  (x_shape, idx, inv_idx) = res
  if mode == "padded_gather":
    if inv_idx is None:
      raise ValueError("For padded gather the inv_idx has to be specified.")
    grad = g[inv_idx, ...]
  elif g.shape[0] == x_shape[0]:
    inv_idx = inv_idx if inv_idx is not None else jnp.argsort(idx)
    grad = g[inv_idx, ...]
  else:  # scatter
    buf = jax.lax.empty(x_shape, dtype=g.dtype) if empty_for_scatter else jnp.zeros(x_shape, dtype=g.dtype)
    grad = buf.at[idx, ...].set(g, mode="drop")
  return (grad, None, None)


unique_gather.defvjp(unique_gather_fwd, unique_gather_bwd)


_tb_process, _tb_port = None, None


@contextlib.contextmanager
def profile(path="/tmp/profiles"):
  global _tb_process, _tb_port
  if _tb_process is None:
    _tb_port = 52432
    used_ports = {p.laddr.port for p in psutil.net_connections()}
    while _tb_port in used_ports:
      _tb_port += 1
    devnull = open(os.devnull, "w")
    _tb_process = Popen(["xprof", "--port", str(_tb_port), "--logdir", path], stdout=devnull, stderr=devnull)

  with jax.profiler.trace("/tmp/profiles"):
    yield
  profiles = sorted(Path(path).absolute().glob("**/*.xplane.pb"), key=lambda x: x.stat().st_mtime)
  profile_name = profiles[-1].parts[-2]
  port, use_xprof = _tb_port, True
  if use_xprof:
    url = "http://localhost:{port}/data/plugin/profile/trace_viewer@;run={name};tag=trace_viewer@"  # xprof version
  else:
    url = "http://localhost:{port}/?run={name}&tag=trace_viewer"  # tensorboard version
  print(url.format(port=port, name=profile_name))
