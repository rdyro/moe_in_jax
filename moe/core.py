import dataclasses
import warnings
from functools import partial
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import sc_kernels as sc
from .ra2a_simulator import ragged_all_to_all as ag_ra2a
from .utils import (PaddedGroupPaddedMetadata, RA2AMeta, add_indices,
                    compute_padded_group_gather, register_jax_dataclass,
                    unique_gather, tpu_sublane_size)

SENTINEL_VALUE = 2 ** 31 - 1
SPARSECORE_PAD_SIZE = 1024


class RaggedAllToCallCallable(Protocol):
  def __call__(self, operand: jax.Array, output: jax.Array, input_offsets: jax.Array, send_sizes: jax.Array,
               output_offsets: jax.Array, recv_sizes: jax.Array, *, axis_name: str) -> jax.Array:
    ...


class ComputeBlockCallable(Protocol):
  def __call__(self, tokens: jax.Array, local_group_sizes: jax.Array, *extra_args) -> jax.Array:
    ...


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class MoEConfig:
  multiple: int = 1
  gathers: Literal["builtin", "custom", "custom_sc"] = "builtin"
  safety_factor: float = 1.25
  ra2a: RaggedAllToCallCallable | None = None
  pad_buffers_to_multiple: int | None = SPARSECORE_PAD_SIZE


class MoEInfo(NamedTuple):
  batch_size: int
  experts_per_tok: int
  num_experts: int


@register_jax_dataclass(meta_fields=["info"])
@dataclasses.dataclass
class MoEMeta:
  info: MoEInfo
  local_ra2a_sort: jax.Array
  local_ra2a_isort: jax.Array
  preamble: RA2AMeta
  epilogue: RA2AMeta
  local_permute: PaddedGroupPaddedMetadata


def maybe_pad_size(size: int, pad_to_multiple: int | None):
  if pad_to_multiple is None:
    return size
  return ((size + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple


RA2A_SHAPE_SAFE_FNS = (None, jax.lax.ragged_all_to_all, ag_ra2a)


def run_moe_shard_map(
    all_idxs: jax.Array, x: jax.Array, *extra_args,
    compute_block: ComputeBlockCallable | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(),
):
  assert x.ndim >= 2, f"Tokens must be at least 2D, but got x = {jax.typeof(x)}"
  if x.ndim < 3 and config.multiple != 1:
    warnings.warn("Padding groups for a 2D ra2a is not currently well tested, proceed at your own risk.")
  if x.ndim == 2 and config.multiple % tpu_sublane_size() and config.ra2a not in RA2A_SHAPE_SAFE_FNS:
    raise ValueError("You're attempting to ragged-all-to-all a 2D tensor via a pallas call that exploits the assumption"
                     f" that send chunks are aligned to sublanes, but {config.multiple=} and {tpu_sublane_size()=}.")

  shard_idx, num_shards = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
  experts_per_shard = experts_num // num_shards
  experts_per_tok_ = all_idxs.size // x.shape[0] // num_shards  # because all_idxs is replicated
  assert experts_per_tok_ == experts_per_tok, (
    f"Declared {experts_per_tok=} does not match apparent {all_idxs.size // x.shape[0] // num_shards=}."
    f" {all_idxs.shape=} and {x.shape=}"
  )
  assert all_idxs.ndim in (1, 2), f"{jax.typeof(all_idxs)=} should be stacked (expert_shards, -1) or tiled (-1,)"
  config = dataclasses.replace(config, ra2a=config.ra2a or jax.lax.ragged_all_to_all)

  ####################################################################################################################
  # metadata computation #############################################################################################
  ####################################################################################################################

  with jax.named_scope("compute_metadata"):
    if all_idxs.ndim == 1:
      all_idxs = all_idxs.reshape((num_shards, -1))
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

    # batch * expert_per_token * safety factor
    buffer_size = round(x.shape[0] * experts_per_tok * config.safety_factor)
    buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)

    local_expert_idxs = jax.lax.empty((buffer_size + all_local_expert_idxs.shape[-1],), jnp.int32)
    local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:buffer_size]
    local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
    local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

    # compute the local permutation
    local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
    local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
    local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=1,
                                                group_counts=local_group_counts)
    local_group_counts = local_permute.group_counts_with_padding

    info = MoEInfo(x.shape[0], experts_per_tok, experts_num)
    meta = MoEMeta(info, local_ra2a_sort, local_ra2a_isort, preamble, epilogue, local_permute)

  ####################################################################################################################
  # compute ##########################################################################################################
  ####################################################################################################################

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
    total_recv_size = jnp.sum(recv_sizes)
    # check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor
    buffer = jax.lax.empty((buffer_size,) + x.shape[1:], dtype=x.dtype)
    y = config.ra2a(x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)

  # step 3: gather tokens locally so they're expert-contiguous
  with jax.named_scope("local_gather_before"):
    if config.gathers == "custom":
      y = unique_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
    elif config.gathers == "custom_sc":
      y = sc.unique_sc_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
    else:
      y = y[meta.local_permute.sort_idx, ...]

  # step 4: perform gmm computation
  with jax.named_scope("compute"):
    if compute_block is not None:
      # y = compute_block(y, local_group_counts)
      y = compute_block(y, meta.local_permute.group_counts_with_padding, *extra_args)

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
    out = jax.lax.empty((all_idxs.shape[-1],) + x.shape[1:], dtype=x.dtype)
    x_sort = config.ra2a(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)

  # step 7: gather so each token repeats are next to each other
  with jax.named_scope("expert_to_tokens_gather"):
    if config.gathers == "custom":
      y = unique_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
    elif config.gathers == "custom_sc":
      y = sc.unique_sc_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
    else:
      y = x_sort[meta.local_ra2a_isort, ...]

  # step 8: reduce across experts
  with jax.named_scope("reduction_across_experts"):
    y = y.reshape((x.shape[0], experts_per_tok) + x.shape[1:])
    y = jnp.sum(y, 1)

  return y


def run_moe(
    all_idxs: jax.Array, x: jax.Array, *extra_args,
    compute_block: ComputeBlockCallable | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(),
):
  out_specs = P(axis_name, *[None for _ in range(x.ndim - 1)])
  fn = partial(run_moe_shard_map, compute_block=compute_block, axis_name=axis_name, experts_per_tok=experts_per_tok,
               experts_num=experts_num, config=config)
  fn = jax.shard_map(fn, out_specs=out_specs, check_vma=False)
  return fn(all_idxs, x, *extra_args)


def run_moe_ag_shard_map(
    all_idxs: jax.Array, x: jax.Array, *extra_args,
    compute_block: ComputeBlockCallable | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(),
):
  shard_idx, num_shards = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
  experts_per_shard = experts_num // num_shards
  experts_per_tok_ = all_idxs.size // x.shape[0] // num_shards  # because all_idxs is replicated
  assert experts_per_tok_ == experts_per_tok, (
    f"Declared {experts_per_tok=} does not match apparent {all_idxs.size // x.shape[0] // num_shards=}."
    f" {all_idxs.shape=} and {x.shape=}"
  )
  assert all_idxs.ndim in (1, 2), f"{jax.typeof(all_idxs)=} should be stacked (expert_shards, -1) or tiled (-1,)"

  ####################################################################################################################
  # metadata computation #############################################################################################
  ####################################################################################################################

  with jax.named_scope("all_gather_tokens"):
    all_x = jax.lax.all_gather(x, axis_name, axis=0, tiled=True)

  with jax.named_scope("compute_metadata"):
    if all_idxs.ndim != 1:
      all_idxs = all_idxs.reshape(-1)
    valid_mask = (all_idxs >= shard_idx * experts_per_shard) & (all_idxs < (shard_idx + 1) * experts_per_shard)
    valid_idxs = jnp.where(valid_mask, all_idxs, SENTINEL_VALUE)
    total_local_size = jnp.sum(valid_mask)
    # jax.debug.print("total_local_size = {}", total_local_size)
    all_sizes = jnp.bincount(all_idxs[shard_idx, :], length=experts_num)
    all_sizes = jax.lax.all_gather(all_sizes, axis_name, axis=0, tiled=False)
    group_sizes = jnp.sum(all_sizes.reshape((num_shards, num_shards, -1))[:, shard_idx, :], 0)

    # batch * expert_per_token * safety factor
    buffer_size = round(x.shape[0] * experts_per_tok * config.safety_factor)
    buffer_size = maybe_pad_size(buffer_size, config.pad_buffers_to_multiple)
    local_sort = jnp.argsort(valid_idxs)[:buffer_size]
    local_isort = jnp.argsort(local_sort)[:buffer_size]

  ####################################################################################################################
  # compute ##########################################################################################################
  ####################################################################################################################

  # step 3: gather tokens locally so they're expert-contiguous
  with jax.named_scope("local_gather_before"):
    y = all_x[local_sort // experts_per_tok, ...]

  # step 4: perform gmm computation
  with jax.named_scope("compute"):
    if compute_block is not None:
      y = compute_block(y, group_sizes, *extra_args)
    mask = jnp.expand_dims(jnp.arange(y.shape[0]) < total_local_size, tuple(range(1, y.ndim)))
    y = jnp.where(mask, y, 0)

  # step 5: unpermute tokens locally to organize them into chunks in which they arrived
  with jax.named_scope("local_scatter_after"):
    y = jnp.zeros(all_x.shape, dtype=all_x.dtype).at[local_sort // experts_per_tok, ...].add(y)

  with jax.named_scope("reduce-scatter"):
    y = jax.lax.psum_scatter(y, axis_name=axis_name, scatter_dimension=0, tiled=True)

  return y


def run_moe_ag(
    all_idxs: jax.Array, x: jax.Array, *extra_args,
    compute_block: ComputeBlockCallable | None = None,
    axis_name: str, experts_per_tok: int, experts_num: int, config: MoEConfig = MoEConfig(),
):
  out_specs = P(axis_name, *[None for _ in range(x.ndim - 1)])
  fn = partial(run_moe_ag_shard_map, compute_block=compute_block, axis_name=axis_name, experts_per_tok=experts_per_tok,
               experts_num=experts_num, config=config)
  fn = jax.shard_map(fn, out_specs=out_specs, check_vma=False)
  return fn(all_idxs, x, *extra_args)
