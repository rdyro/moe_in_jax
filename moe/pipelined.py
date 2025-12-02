import dataclasses
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .utils import RA2AMeta, add_indices, compute_padded_group_gather, unique_gather, register_jax_dataclass
from .core import SENTINEL_VALUE, MoEMeta, MoEInfo, MoEConfig, maybe_pad_size
from . import sc_kernels as sc

from .ra2a import make_split_ra2a


@register_jax_dataclass(meta_fields=["compute_meta", "load_fn", "compute_fn", "unload_fn"])
@dataclasses.dataclass
class MoEMethods:
  compute_meta: Callable[[jax.Array], MoEMeta]
  load_fn: Callable[[jax.Array, MoEMeta], jax.Array]
  compute_fn: Callable[[jax.Array, MoEMeta], jax.Array]
  unload_fn: Callable[[jax.Array, MoEMeta], jax.Array]


def _create_pipelined_moe(
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    *,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(),
) -> MoEMethods:
  assert config.gathers in ("builtin", "custom", "custom_sc")

  ######################################################################################################################
  # metadata computation ###############################################################################################
  ######################################################################################################################

  def compute_meta(all_idxs: jax.Array):
    shard_idx, num_shards = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
    experts_per_shard = experts_num // num_shards
    assert all_idxs.ndim in (1, 2), f"{jax.typeof(all_idxs)=} should be stacked (expert_shards, -1) or tiled (-1,)"

    with jax.named_scope("compute_metadata"):
      if all_idxs.ndim == 1:
        all_idxs = all_idxs.reshape((num_shards, -1))
      assert all_idxs.shape[-1] % experts_per_tok == 0
      actual_token_num = all_idxs.shape[-1]

      all_sizes = jnp.bincount(all_idxs[shard_idx, :], length=experts_num)
      all_sizes = jax.lax.all_gather(all_sizes, axis_name, axis=0, tiled=False)
      all_shard_sizes = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard)), axis=-1)

      # padding send groups to be aligned to a multiple, this allows a sublane-aligned 2D ra2a on TPU
      if config.multiple != 1:
        fill_experts = -all_shard_sizes % config.multiple
        add_indices_fn = jax.vmap(
            partial(add_indices, max_size=config.multiple - 1, fill_value=SENTINEL_VALUE), (None, 0)
        )
        last_expert_per_shard = jnp.arange(num_shards) * experts_per_shard + (experts_per_shard - 1)
        fill_indices = add_indices_fn(last_expert_per_shard, fill_experts)
        all_idxs = jnp.concat([all_idxs, fill_indices], axis=1)
        all_shard_sizes = all_shard_sizes + fill_experts
        all_sizes = all_sizes.reshape((num_shards, num_shards, experts_per_shard))
        all_sizes = all_sizes.at[..., -1].add(fill_experts).reshape((num_shards, num_shards * experts_per_shard))

      # compute the ra2a communication #################################################################################

      # all_sizes = batch_bincount_fn(all_idxs // experts_per_shard)  # after padding
      all_input_offsets = jnp.cumsum(all_shard_sizes, axis=-1) - all_shard_sizes  # cumsum from 0
      all_output_offsets = jnp.cumsum(all_shard_sizes, axis=0) - all_shard_sizes  # cumsum from 0
      send_sizes, recv_sizes = all_shard_sizes[shard_idx, :], all_shard_sizes[:, shard_idx]
      input_offsets, output_offsets = all_input_offsets[shard_idx, :], all_output_offsets[shard_idx, :]
      preamble = RA2AMeta(input_offsets, send_sizes, output_offsets, recv_sizes)
      inv_input_offsets = all_output_offsets[:, shard_idx]  # we send back chunks starting where we received them
      inv_output_offsets = all_input_offsets[:, shard_idx]  # we write the chunks from where they originally came
      inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
      epilogue = RA2AMeta(inv_input_offsets, inv_send_sizes, inv_output_offsets, inv_recv_sizes)

      # compute within expert shard sort; receiving several locally sorted chunks ######################################

      # get gather indices for pre-ra2a by-expert-shard organization
      local_ra2a_sort = jnp.argsort(all_idxs[shard_idx, :])
      local_ra2a_isort = jnp.argsort(local_ra2a_sort)[:actual_token_num]

      all_local_expert_idxs = jax.lax.all_gather(all_idxs[shard_idx, :][local_ra2a_sort], axis_name)

      # compute local expert idxs after the transfer (technically a ra2a, but more efficient via AG and dynamic slice)
      def _update_fn(i, local_expert_idxs):
        update = jnp.roll(all_local_expert_idxs[i, :], -all_input_offsets[i, shard_idx])
        return jax.lax.dynamic_update_slice_in_dim(local_expert_idxs, update, all_output_offsets[i, shard_idx], 0)

      buffer_size = round(all_idxs.shape[-1] * config.safety_factor)  # batch * expert_per_token * safety factor
      buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)
      local_expert_idxs = jax.lax.empty((buffer_size + all_local_expert_idxs.shape[-1],), jnp.int32)
      local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:buffer_size]
      local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
      local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

      # compute the local permutation
      local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
      local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
      local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=config.multiple,
                                                  group_counts=local_group_counts)
      local_group_counts = local_permute.group_counts_with_padding

      info = MoEInfo(all_idxs.shape[-1] // experts_per_tok, experts_per_tok, experts_num)
      return MoEMeta(info, local_ra2a_sort, local_ra2a_isort, preamble, epilogue, local_permute)

  ######################################################################################################################
  # compute ############################################################################################################
  ######################################################################################################################

  def load_fn():
    def prepare_fn(x, meta: MoEMeta):
      assert meta.info.batch_size == x.shape[0], f"Expected {meta.info.batch_size=}, but got {x.shape[0]=}."
      # step 1: gather local tokens for every expert per token
      with jax.named_scope("tokens_to_experts_gather"):
        if config.gathers == "custom":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = unique_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        elif config.gathers == "custom_sc":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = sc.unique_sc_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        else:
          x_sort = x[meta.local_ra2a_sort // experts_per_tok, ...]

      # step 2: communicate expert-gathered-tokens to their corresponding expert shards
      with jax.named_scope("ra2a_tokens"):
        total_recv_size = jnp.sum(meta.preamble.recv_sizes)
        # TODO(rdyro): check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor

        # batch * expert_per_token * safety factor
        buffer_size = round(meta.info.batch_size * meta.info.experts_per_tok * config.safety_factor)
        buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)

        buffer = jax.lax.empty((buffer_size,) + x.shape[1:], dtype=x.dtype)

        # y = ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
        # return _start_fn(x_sort, buffer, *dataclasses.astuple(meta.preamble))
        return (x_sort, buffer, *dataclasses.astuple(meta.preamble))

    def finalize_fn(y, meta: MoEMeta):
      # y = _wait_fn(future)
      # step 3: gather tokens locally so they're expert-contiguous
      with jax.named_scope("local_gather_before"):
        if config.gathers == "custom":
          y = unique_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        elif config.gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        else:
          y = y[meta.local_permute.sort_idx, ...]
      return y

    return prepare_fn, finalize_fn

  def compute_fn(y: jax.Array, meta: MoEMeta, *args):
    # step 4: perform gmm computation
    local_group_counts = meta.local_permute.group_counts_with_padding
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts, *args)
    return y

  def unload_fn():
    def prepare_fn(y, meta: MoEMeta):
      # step 5: unpermute tokens locally to organize them into chunks in which they arrived
      with jax.named_scope("local_gather_after"):
        if config.gathers == "custom":
          y = unique_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        elif config.gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        else:
          y = y[meta.local_permute.isort_idx, ...]

      # step 6: communincate the chunks back to their origins
      with jax.named_scope("ra2a_results"):
        out = jax.lax.empty((meta.info.batch_size * meta.info.experts_per_tok, *y.shape[1:]), dtype=y.dtype)
      # to be communicated as ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      return (y, out, *dataclasses.astuple(meta.epilogue))

    def finalize_fn(x_sort, meta: MoEMeta):
      # x_sort comes from ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)

      # step 7: gather so each token repeats are next to each other
      with jax.named_scope("expert_to_tokens_gather"):
        if config.gathers == "custom":
          y = unique_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        elif config.gathers == "custom_sc":
          y = sc.unique_sc_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        else:
          y = x_sort[meta.local_ra2a_isort, ...]

      # step 8: weigh by expert weights
      with jax.named_scope("reduction_across_experts"):
        y = y.reshape((meta.info.batch_size, experts_per_tok) + y.shape[1:])
        y = jnp.sum(y, axis=1)
      return y
    return prepare_fn, finalize_fn

  return MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)


########################################################################################################################

def _overlap_fn(y1, y2, meta1, meta2, meta3, x_next, *extra_args, axis_name: str, moe_methods, i, splits):
  """A function to overlap communication with the communication block."""
  fut1, fut3, y1_next, y3_next = None, None, None, None
  if 0 <= i < splits:
    prepare_fn1, finalize_fn1 = moe_methods.load_fn()
    fut1 = prepare_fn1(x_next, meta1)
    # fut1 = tuple(fut1) + dataclasses.astuple(meta1.preamble)

  if 2 <= i < splits + 2:
    prepare_fn3, finalize_fn3 = moe_methods.unload_fn()
    fut3 = prepare_fn3(y2, meta3)
    # fut3 = tuple(fut3) + dataclasses.astuple(meta3.epilogue)

  ra2a_split = make_split_ra2a(moe_methods.compute_fn if 1 <= i < splits + 1 else None)

  (y1_next, y3_next), y2_next = ra2a_split((fut1, fut3), (y1, meta2, *extra_args), axis_name=axis_name)

  y1_next = finalize_fn1(y1_next, meta1) if 0 <= i < splits else y1_next
  y3_next = finalize_fn3(y3_next, meta3) if 2 <= i < splits + 2 else y3_next

  return y1_next, y2_next, y3_next


def run_moe_pipelined_shard_map(
    all_idxs: jax.Array, x: jax.Array,
    *extra_args,
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(), splits: int = 1,
):
  moe_methods = _create_pipelined_moe(
    compute_block, axis_name=axis_name, experts_per_tok=experts_per_tok, experts_num=experts_num, config=config
  )
  axis_size = jax.lax.axis_size(axis_name)
  assert x.shape[0] % splits == 0
  assert all_idxs.shape[0] % (splits * axis_size) == 0

  x_ = x.reshape((splits, x.shape[0] // splits, *x.shape[1:]))
  all_idxs_ = all_idxs.reshape((axis_size, splits, all_idxs.size // (splits * axis_size)))

  all_metas = [moe_methods.compute_meta(all_idxs_[:, i, ...]) for i in range(splits)]
  x_next = x_[0, ...]
  y1s, y2s, y3s = [], [], []
  for i in range(splits + 2):
    overlap_fn_ = partial(_overlap_fn, moe_methods=moe_methods, i=i, splits=splits, axis_name=axis_name)

    meta1 = all_metas[i] if 0 <= i < splits else 0
    y1, meta2 = (y1s[i - 1], all_metas[i - 1]) if 1 <= i < splits + 1 else (None, None)
    y2, meta3 = (y2s[i - 2], all_metas[i - 2]) if 2 <= i < splits + 2 else (None, None)
    y1, y2, y3 = overlap_fn_(y1, y2, meta1, meta2, meta3, x_next, *extra_args)
    x_next = x_[i + 1, ...] if (i < splits - 1) else None

    y1, y2, y3, x_next = jax.lax.optimization_barrier((y1, y2, y3, x_next))

    y1s.append(y1) if y1 is not None else None
    y2s.append(y2) if y2 is not None else None
    y3s.append(y3) if y3 is not None else None
  return jnp.concat(y3s, axis=0)


# @partial(jax.jit, static_argnames=("splits",))
# def custom_moe(all_idxs: jax.Array, x: jax.Array, *extra_args: tuple[jax.Array, ...], splits: int=1):
def run_moe_pipelined(
    all_idxs: jax.Array, x: jax.Array, *extra_args,
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(), splits: int = 1,
):
  # opts = dict(axis_name=axis_name, experts_per_tok=experts_per_tok, experts_num=g, gathers="custom")
  # moe_methods = _create_moe(compute_block, **opts)

  extra_specs = jax.tree.map(lambda x: jax.typeof(x).sharding.spec, extra_args)

  fn = partial(
    run_moe_pipelined_shard_map,
    compute_block=compute_block, axis_name=axis_name, experts_per_tok=experts_per_tok, experts_num=experts_num,
    config=config, splits=splits
  )
  fn = jax.shard_map(fn, in_specs=(P(), P(axis_name), *extra_specs), out_specs=P(axis_name), check_vma=False)
  return fn(all_idxs, x, *extra_args)

  # def inner(x, all_idxs, *extra_args):
  #  axis_size = jax.lax.axis_size(axis_name)
  #  assert x.shape[0] % splits == 0
  #  assert all_idxs.shape[0] % (splits * axis_size) == 0

  #  x_ = x.reshape((splits, x.shape[0] // splits, *x.shape[1:]))
  #  all_idxs_ = all_idxs.reshape((axis_size, splits, all_idxs.size // (splits * axis_size)))

  #  all_metas = [moe_methods.compute_meta(all_idxs_[:, i, ...]) for i in range(splits)]
  #  x_next = x_[0, ...]
  #  y1s, y2s, y3s = [], [], []
  #  for i in range(splits + 2):
  #    overlap_fn_ = partial(_overlap_fn, moe_methods, i, splits)

  #    meta1 = all_metas[i] if 0 <= i < splits else 0
  #    y1, meta2 = (y1s[i - 1], all_metas[i - 1]) if 1 <= i < splits + 1 else (None, None)
  #    y2, meta3 = (y2s[i - 2], all_metas[i - 2]) if 2 <= i < splits + 2 else (None, None)
  #    y1, y2, y3 = overlap_fn_(y1, y2, meta1, meta2, meta3, x_next, *extra_args)
  #    x_next = x_[i + 1, ...] if (i < splits - 1) else None

  #    y1, y2, y3, x_next = jax.lax.optimization_barrier((y1, y2, y3, x_next))

  #    y1s.append(y1) if y1 is not None else None
  #    y2s.append(y2) if y2 is not None else None
  #    y3s.append(y3) if y3 is not None else None
  #  return jnp.concat(y3s, axis=0)

  return fn(x, all_idxs, *extra_args)
