import dataclasses
from functools import partial
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .utils import RA2AMeta, add_indices, compute_padded_group_gather


class RaggedAllToCallCallable(Protocol):
  def __call__(self, operand: jax.Array, output: jax.Array, input_offsets: jax.Array, send_sizes: jax.Array,
               output_offsets: jax.Array, recv_sizes: jax.Array, *, axis_name: str) -> jax.Array:
    ...


SENTINEL_VALUE = 2 ** 31 - 1


def _get_gather_dims(x):
  return dict(
      dimension_numbers=jax.lax.GatherDimensionNumbers(
          offset_dims=tuple(range(1, x.ndim)), collapsed_slice_dims=(0,), start_index_map=(0,)
      ),
      slice_sizes=(1, *x.shape[1:])
  )


def run_moe(x: jax.Array, all_idxs: jax.Array,
            compute_block: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
            reduce_block: Callable[[jax.Array], jax.Array] | None = None,
            ragged_all_to_all: RaggedAllToCallCallable = jax.lax.ragged_all_to_all,
            *, axis_name: str, experts_num: int, safety_factor: int = 2, multiple: int = 1,
            custom_gathers: bool = False):

  out_specs = P(axis_name, *[None for _ in range(x.ndim - 1)])

  @partial(jax.shard_map, out_specs=out_specs, check_vma=False)
  def fn(x: jax.Array, all_idxs: jax.Array):
    shard_idx, num_shards = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
    experts_per_shard = experts_num // num_shards
    experts_per_tok = all_idxs.size // x.shape[0] // num_shards  # because all_idxs is replicated
    assert all_idxs.ndim in (1, 2), f"{jax.typeof(all_idxs)=} should be stacked (expert_shards, -1) or tiled (-1,)"

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

      if multiple != 1:  # padding send groups to be aligned to a multiple, this allows a sublane-aligned 2D ra2a on TPU
        fill_experts = -all_shard_sizes % multiple
        add_indices_fn = jax.vmap(partial(add_indices, max_size=multiple - 1, fill_value=SENTINEL_VALUE), (None, 0))
        last_expert_per_shard = jnp.arange(num_shards) * experts_per_shard + (experts_per_shard - 1)
        fill_indices = add_indices_fn(last_expert_per_shard, fill_experts)
        all_idxs = jnp.concat([all_idxs, fill_indices], axis=1)
        all_shard_sizes = all_shard_sizes + fill_experts
        all_sizes = all_sizes.reshape((num_shards, num_shards, experts_per_shard)).at[..., -1].add(
            fill_experts).reshape(all_sizes.shape)

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

      bef = x.shape[0] * experts_per_tok * safety_factor  # batch * expert_per_token * safety factor
      local_expert_idxs = jax.lax.empty((bef + all_local_expert_idxs.shape[-1],), jnp.int32)
      local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:bef]
      local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
      local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

      # compute the local permutation
      local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
      local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
      # local_group_counts = jnp.sum(
      #   jax.lax.dynamic_slice_in_dim(all_sizes, experts_per_shard * shard_idx, experts_per_shard, axis=-1), 0
      # )
      # local_group_counts = jnp.bincount(local_expert_idxs_, length=experts_per_shard)
      local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=multiple,
                                                  group_counts=local_group_counts)

      # local_sort = jnp.argsort(local_expert_idxs)
      # local_isort = jnp.argsort(local_sort)
      # local_group_sizes = jnp.bincount(
      #   jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE),
      #   length=experts_per_shard
      # )
      local_group_counts = local_permute.group_counts_with_padding

    ####################################################################################################################
    # compute ##########################################################################################################
    ####################################################################################################################

    # step 1: gather local tokens for every expert per token
    with jax.named_scope("tokens_to_experts_gather"):
      x_sort = x[local_ra2a_sort // experts_per_tok, ...]

    # step 2: communicate expert-gathered-tokens to their corresponding expert shards
    with jax.named_scope("ra2a_tokens"):
      total_recv_size = jnp.sum(recv_sizes)
      # check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor
      buffer = jax.lax.empty((x.shape[0] * experts_per_tok * safety_factor,) + x.shape[1:], dtype=x.dtype)
      y = ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(preamble), axis_name=axis_name)

    # step 3: gather tokens locally so they're expert-contiguous
    with jax.named_scope("local_gather_before"):
      if custom_gathers:
        # y = unique_gather(y, local_permute.sort_idx, local_permute.isort_idx, mode="gather")
        y = jax.lax.gather(y, local_permute.sort_idx[:, None], **_get_gather_dims(y), unique_indices=True)
      else:
        y = y[local_permute.sort_idx, ...]

    # step 4: perform gmm computation
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts)

    # step 5: unpermute tokens locally to organize them into chunks in which they arrived
    with jax.named_scope("local_gather_after"):
      if custom_gathers:
        # y = unique_gather(y, local_permute.isort_idx, local_permute.isort_idx, mode="scatter")
        y = jax.lax.gather(y, local_permute.isort_idx[:, None], **_get_gather_dims(y), unique_indices=True)
      else:
        y = y[local_permute.isort_idx, ...]

    # step 6: communincate the chunks back to their origins
    with jax.named_scope("ra2a_results"):
      out = jax.lax.empty((all_idxs.shape[-1],) + x.shape[1:], dtype=x.dtype)
      x_sort = ragged_all_to_all(y, out, *dataclasses.astuple(epilogue), axis_name=axis_name)

    # step 7: gather so each token repeats are next to each other
    with jax.named_scope("expert_to_tokens_gather"):
      if custom_gathers:
        # y = unique_gather(x_sort, local_ra2a_isort, local_ra2a_isort, mode="scatter")
        y = jax.lax.gather(x_sort, local_ra2a_isort[:, None], **_get_gather_dims(x_sort), unique_indices=True)
      else:
        y = x_sort[local_ra2a_isort, ...]
      y = y.reshape((x.shape[0], experts_per_tok) + x.shape[1:])

    # step 8: weigh by expert weights
    with jax.named_scope("reduction_across_experts"):
      if reduce_block is not None:
        y = reduce_block(y)

    return y

  return fn(x, all_idxs)
