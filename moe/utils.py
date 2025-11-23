import contextlib
import dataclasses
import os
import random
from functools import partial
from pathlib import Path
from subprocess import Popen

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.sharding import Sharding

zip_ = zip
zip = partial(zip_, strict=True)


@partial(
  jax.tree_util.register_dataclass,
  data_fields=["input_offsets", "send_sizes", "output_offsets", "recv_sizes"],
  meta_fields=[],
)
@dataclasses.dataclass
class RA2AMeta:
  """Holds sizes and offsets for ragged all-to-all communication."""
  input_offsets: jax.Array
  send_sizes: jax.Array
  output_offsets: jax.Array
  recv_sizes: jax.Array


def empty(shape, dtype, out_sharding=None):
  """Create an empty array with the given shape and dtype, and the given sharding."""
  if out_sharding is None:
    out_shape, out_specs = jax.ShapeDtypeStruct(shape, dtype), pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(lambda *args: None, out_shape=out_shape, out_specs=out_specs)()

  spec = out_sharding.spec if isinstance(out_sharding, Sharding) else out_sharding
  assert len(spec) <= len(shape)
  shard_axes = tuple(spec) + (None,) * (len(shape) - len(spec))

  if isinstance(out_sharding, Sharding):
    decorator = partial(jax.shard_map, mesh=out_sharding.mesh, out_specs=out_sharding.spec, check_vma=False)
  else:
    decorator = partial(jax.shard_map, out_specs=out_sharding, check_vma=False)

  @decorator
  def _():
    local_shape = [(s // jax.lax.axis_size(a)) if a is not None else s for s, a in zip(shape, shard_axes)]
    out_shape, out_specs = jax.ShapeDtypeStruct(local_shape, dtype), pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(lambda *args: None, out_shape=out_shape, out_specs=out_specs)()

  return _()


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


@partial(jax.jit, static_argnames=("max_size",))
def add_indices(idx_list: jax.Array, sizes: jax.Array, max_size: int, fill_value: int = 2 ** 31 - 1):
  """One way to add indices from a list in desired counts, filling the rest with a fill value."""
  start_idx = jnp.cumsum(sizes) - sizes
  end_idx = jnp.cumsum(sizes)
  iota = jnp.arange(idx_list.size * max_size)[None, :]
  mask = (iota >= start_idx[:, None]) & (iota < end_idx[:, None])
  # the result is a list having sizes[0] of idx_list[0], sizes[1] of idx_list[1] and so on
  # the unfilled values are filled with fill_value
  return jnp.sum(idx_list[:, None] * mask, axis=0) + ~jnp.any(mask, axis=0) * fill_value


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


@partial(jax.tree_util.register_dataclass, meta_fields=[], data_fields=[
  "group_counts", "group_counts_with_padding", "group_idx", "group_idx_with_padding", "sort_idx", "isort_idx"
])
@dataclasses.dataclass
class PaddedGroupPaddedMetadata:
  group_idx: jax.Array
  group_idx_with_padding: jax.Array
  group_counts: jax.Array
  group_counts_with_padding: jax.Array
  sort_idx: jax.Array
  isort_idx: jax.Array


def compute_padded_group_gather(group_idx: jax.Array, groups: int, multiple: int) -> PaddedGroupPaddedMetadata:
  """Compute metadata for sorting tokens according to group_idx with padding to make groups divisible by `multiple`."""

  assert multiple >= 1
  group_counts = jnp.bincount(group_idx, length=groups)

  if multiple != 1:
    padding_idxs = add_indices(jnp.arange(groups), -group_counts % multiple, max_size=multiple - 1)
    group_idx_with_padding = jnp.concat([group_idx, padding_idxs], axis=0)
    group_counts_with_padding = group_counts + (-group_counts % multiple)
  else:
    group_idx_with_padding, group_counts_with_padding = group_idx, group_counts
  sort_idx = jnp.argsort(group_idx_with_padding)
  isort_idx = jnp.argsort(sort_idx)[:group_idx.shape[0]]

  return PaddedGroupPaddedMetadata(
    group_idx, group_idx_with_padding, group_counts, group_counts_with_padding, sort_idx, isort_idx
  )


@partial(jax.custom_vjp, nondiff_argnames=("mode", "empty_buffer_for_scatter"))
def unique_gather(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, mode: str, empty_buffer_for_scatter: bool = True):
  """Gather (unique indices): Backwards pass is gather/scatter, avoiding costly scatter-add."""
  assert mode in ("gather", "scatter")
  return x[idx, ...]


def unique_gather_fwd(x: jax.Array, idx: jax.Array, inv_idx: jax.Array, mode: str, empty_buffer_for_scatter: bool):
  print(f"{empty_buffer_for_scatter=}, {mode=}")
  static = dict(mode=mode, empty_buffer_for_scatter=empty_buffer_for_scatter)
  return unique_gather(x, idx, inv_idx, **static), (x.shape, inv_idx,)


def unique_gather_bwd(mode: str, empty_buffer_for_scatter: bool, res, g):
  (x_shape, inv_idx,) = res
  if mode == "gather":
    grad = g[inv_idx, ...]
  else:  # scatter
    # TODO(rdyro): check if this gather optimization actually outperforms scatter
    if g.shape[0] == x_shape[0]:  # shortcut if input/output shape matches
      grad = g[jnp.argsort(inv_idx), ...]
    else:  # otherwise really use scatter
      buf = jax.lax.empty(x_shape, dtype=g.dtype) if empty_buffer_for_scatter else jnp.zeros(x_shape, dtype=g.dtype)
      grad = buf.at[inv_idx, ...].set(g, mode="drop")
  return (grad, None, None)


unique_gather.defvjp(unique_gather_fwd, unique_gather_bwd)


_tb_process, _tb_port = None, None


@contextlib.contextmanager
def profile(path="/tmp/profiles"):
  global _tb_process, _tb_port
  if _tb_process is None:
    devnull = open(os.devnull, "w")
    _tb_port = 52432 + random.randint(0, 1000)
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
