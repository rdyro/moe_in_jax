import dataclasses
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .utils import RA2AMeta, add_indices, compute_padded_group_gather, unique_gather, register_jax_dataclass
from .core import RaggedAllToCallCallable, SENTINEL_VALUE, MoEMeta, MoEInfo, GathersType, SPARSECORE_PAD_SIZE
from . import sc_kernels as sc

from .ra2a import make_ra2a_3d, ra2a_split


@register_jax_dataclass(meta_fields=["compute_meta", "load_fn", "compute_fn", "unload_fn"])
@dataclasses.dataclass
class MoEMethods:
  compute_meta: Callable[[jax.Array], MoEMeta]
  load_fn: Callable[[jax.Array, MoEMeta], jax.Array]
  compute_fn: Callable[[jax.Array, MoEMeta], jax.Array]
  unload_fn: Callable[[jax.Array, MoEMeta], jax.Array]


def create_moe(
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    ragged_all_to_all: RaggedAllToCallCallable = jax.lax.ragged_all_to_all,
    *,
    axis_name: str,
    experts_per_tok: int, experts_num: int,
    safety_factor: float = 1.2, multiple: int = 1, gathers: GathersType = "builtin"
) -> MoEMethods:
  assert gathers in ("builtin", "custom", "custom_sc")

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

      if multiple != 1:  # padding send groups to be aligned to a multiple, this allows a sublane-aligned 2D ra2a on TPU
        fill_experts = -all_shard_sizes % multiple
        add_indices_fn = jax.vmap(partial(add_indices, max_size=multiple - 1, fill_value=SENTINEL_VALUE), (None, 0))
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

      bef = round(all_idxs.shape[-1] * safety_factor)  # batch * expert_per_token * safety factor
      bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE
      local_expert_idxs = jax.lax.empty((bef + all_local_expert_idxs.shape[-1],), jnp.int32)
      local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:bef]
      local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
      local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

      # compute the local permutation
      local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
      local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
      local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=multiple,
                                                  group_counts=local_group_counts)
      local_group_counts = local_permute.group_counts_with_padding

      info = MoEInfo(all_idxs.shape[-1] // experts_per_tok, experts_per_tok, experts_num)
      return MoEMeta(info, local_ra2a_sort, local_ra2a_isort, preamble, epilogue, local_permute)

  ######################################################################################################################
  # compute ############################################################################################################
  ######################################################################################################################
  # _start_fn, _wait_fn = make_ra2a_3d(axis_name=axis_name)

  def load_fn(extra_input, x: jax.Array, meta: MoEMeta):
    assert meta.info.batch_size == x.shape[0], f"Expected {meta.info.batch_size=}, but got {x.shape[0]=}."
    # step 1: gather local tokens for every expert per token
    with jax.named_scope("tokens_to_experts_gather"):
      if gathers == "custom":
        x_sort = jnp.repeat(x, experts_per_tok, axis=0)
        x_sort = unique_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
      elif gathers == "custom_sc":
        x_sort = jnp.repeat(x, experts_per_tok, axis=0)
        x_sort = sc.unique_sc_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
      else:
        x_sort = x[meta.local_ra2a_sort // experts_per_tok, ...]

    # step 2: communicate expert-gathered-tokens to their corresponding expert shards
    with jax.named_scope("ra2a_tokens"):
      total_recv_size = jnp.sum(meta.preamble.recv_sizes)
      # TODO(rdyro): check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor

      # batch * expert_per_token * safety factor
      bef = round(meta.info.batch_size * meta.info.experts_per_tok * safety_factor)
      bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE

      buffer = jax.lax.empty((bef,) + x.shape[1:], dtype=x.dtype)
      # y = ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
      extra_input, y = ra2a_split(extra_input, x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
      # _, y = ra2a_split(2, x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
      print("hello")

    # step 3: gather tokens locally so they're expert-contiguous
    with jax.named_scope("local_gather_before"):
      if gathers == "custom":
        y = unique_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
      elif gathers == "custom_sc":
        y = sc.unique_sc_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
      else:
        y = y[meta.local_permute.sort_idx, ...]
    return extra_input, y

  def compute_fn(y: jax.Array, meta: MoEMeta, *args):
    # step 4: perform gmm computation
    local_group_counts = meta.local_permute.group_counts_with_padding
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts, *args)
    return y

  # def unload_fn(y: jax.Array, meta: MoEMeta):
  def unload_fn(extra_input, y: jax.Array, meta: MoEMeta):
    # step 5: unpermute tokens locally to organize them into chunks in which they arrived
    with jax.named_scope("local_gather_after"):
      if gathers == "custom":
        y = unique_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
      elif gathers == "custom_sc":
        y = sc.unique_sc_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
      else:
        y = y[meta.local_permute.isort_idx, ...]

    # step 6: communincate the chunks back to their origins
    with jax.named_scope("ra2a_results"):
      out = jax.lax.empty((meta.info.batch_size * meta.info.experts_per_tok, *y.shape[1:]), dtype=y.dtype)
      # x_sort = ragged_all_to_all(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      extra_input, x_sort = ra2a_split(extra_input, y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      # _, x_sort = ra2a_split(2, y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)

    # step 7: gather so each token repeats are next to each other
    with jax.named_scope("expert_to_tokens_gather"):
      if gathers == "custom":
        y = unique_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
      elif gathers == "custom_sc":
        y = sc.unique_sc_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
      else:
        y = x_sort[meta.local_ra2a_isort, ...]

    # step 8: weigh by expert weights
    with jax.named_scope("reduction_across_experts"):
      y = y.reshape((meta.info.batch_size, experts_per_tok) + y.shape[1:])
      y = jnp.sum(y, axis=1)

    return extra_input, y

  return MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)

########################################################################################################################
########################################################################################################################
########################################################################################################################


def create_moe2(
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    ragged_all_to_all: RaggedAllToCallCallable = jax.lax.ragged_all_to_all,
    *,
    axis_name: str,
    experts_per_tok: int, experts_num: int,
    safety_factor: float = 1.2, multiple: int = 1, gathers: GathersType = "builtin"
) -> MoEMethods:
  assert gathers in ("builtin", "custom", "custom_sc")

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

      if multiple != 1:  # padding send groups to be aligned to a multiple, this allows a sublane-aligned 2D ra2a on TPU
        fill_experts = -all_shard_sizes % multiple
        add_indices_fn = jax.vmap(partial(add_indices, max_size=multiple - 1, fill_value=SENTINEL_VALUE), (None, 0))
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

      bef = round(all_idxs.shape[-1] * safety_factor)  # batch * expert_per_token * safety factor
      bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE
      local_expert_idxs = jax.lax.empty((bef + all_local_expert_idxs.shape[-1],), jnp.int32)
      local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:bef]
      local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
      local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

      # compute the local permutation
      local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
      local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
      local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=multiple,
                                                  group_counts=local_group_counts)
      local_group_counts = local_permute.group_counts_with_padding

      info = MoEInfo(all_idxs.shape[-1] // experts_per_tok, experts_per_tok, experts_num)
      return MoEMeta(info, local_ra2a_sort, local_ra2a_isort, preamble, epilogue, local_permute)

  ######################################################################################################################
  # compute ############################################################################################################
  ######################################################################################################################
  _start_fn, _wait_fn = make_ra2a_3d(axis_name=axis_name)

  def load_fn():
    def start_fn(x, meta: MoEMeta):
      assert meta.info.batch_size == x.shape[0], f"Expected {meta.info.batch_size=}, but got {x.shape[0]=}."
      # step 1: gather local tokens for every expert per token
      with jax.named_scope("tokens_to_experts_gather"):
        if gathers == "custom":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = unique_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        elif gathers == "custom_sc":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = sc.unique_sc_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        else:
          x_sort = x[meta.local_ra2a_sort // experts_per_tok, ...]

      # step 2: communicate expert-gathered-tokens to their corresponding expert shards
      with jax.named_scope("ra2a_tokens"):
        total_recv_size = jnp.sum(meta.preamble.recv_sizes)
        # TODO(rdyro): check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor

        # batch * expert_per_token * safety factor
        bef = round(meta.info.batch_size * meta.info.experts_per_tok * safety_factor)
        bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE

        buffer = jax.lax.empty((bef,) + x.shape[1:], dtype=x.dtype)
        # y = ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
        return _start_fn(x_sort, buffer, *dataclasses.astuple(meta.preamble))

    def wait_fn(future, meta: MoEMeta):
      y = _wait_fn(future)
      # step 3: gather tokens locally so they're expert-contiguous
      with jax.named_scope("local_gather_before"):
        if gathers == "custom":
          y = unique_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        else:
          y = y[meta.local_permute.sort_idx, ...]
      return y

    return start_fn, wait_fn

  def compute_fn(y: jax.Array, meta: MoEMeta, *args):
    # step 4: perform gmm computation
    local_group_counts = meta.local_permute.group_counts_with_padding
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts, *args)
    return y

  # def unload_fn(y: jax.Array, meta: MoEMeta):
  def unload_fn():
    def start_fn(y, meta: MoEMeta):
      # step 5: unpermute tokens locally to organize them into chunks in which they arrived
      with jax.named_scope("local_gather_after"):
        if gathers == "custom":
          y = unique_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        else:
          y = y[meta.local_permute.isort_idx, ...]

      # step 6: communincate the chunks back to their origins
      with jax.named_scope("ra2a_results"):
        out = jax.lax.empty((meta.info.batch_size * meta.info.experts_per_tok, *y.shape[1:]), dtype=y.dtype)
        # x_sort = ragged_all_to_all(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      return _start_fn(y, out, *dataclasses.astuple(meta.epilogue))

    def wait_fn(future, meta: MoEMeta):
      x_sort = _wait_fn(future)
      # step 7: gather so each token repeats are next to each other
      with jax.named_scope("expert_to_tokens_gather"):
        if gathers == "custom":
          y = unique_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        else:
          y = x_sort[meta.local_ra2a_isort, ...]

      # step 8: weigh by expert weights
      with jax.named_scope("reduction_across_experts"):
        y = y.reshape((meta.info.batch_size, experts_per_tok) + y.shape[1:])
        y = jnp.sum(y, axis=1)
      return y
    return start_fn, wait_fn

  return MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)

########################################################################################################################
########################################################################################################################
########################################################################################################################


def create_moe3(
    compute_block: Callable[[jax.Array, jax.Array | None], jax.Array] | None = None,
    ragged_all_to_all: RaggedAllToCallCallable = jax.lax.ragged_all_to_all,
    *,
    axis_name: str,
    experts_per_tok: int, experts_num: int,
    safety_factor: float = 1.2, multiple: int = 1, gathers: GathersType = "builtin"
) -> MoEMethods:
  assert gathers in ("builtin", "custom", "custom_sc")

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

      if multiple != 1:  # padding send groups to be aligned to a multiple, this allows a sublane-aligned 2D ra2a on TPU
        fill_experts = -all_shard_sizes % multiple
        add_indices_fn = jax.vmap(partial(add_indices, max_size=multiple - 1, fill_value=SENTINEL_VALUE), (None, 0))
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

      bef = round(all_idxs.shape[-1] * safety_factor)  # batch * expert_per_token * safety factor
      bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE
      local_expert_idxs = jax.lax.empty((bef + all_local_expert_idxs.shape[-1],), jnp.int32)
      local_expert_idxs = jax.lax.fori_loop(0, num_shards, _update_fn, local_expert_idxs)[:bef]
      local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
      local_expert_idxs = jnp.where(local_pack_mask, local_expert_idxs, SENTINEL_VALUE)

      # compute the local permutation
      local_expert_idxs_ = jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, SENTINEL_VALUE)
      local_group_counts = jnp.sum(all_sizes.reshape((num_shards, num_shards, experts_per_shard))[:, shard_idx, :], 0)
      local_permute = compute_padded_group_gather(local_expert_idxs_, experts_per_shard, multiple=multiple,
                                                  group_counts=local_group_counts)
      local_group_counts = local_permute.group_counts_with_padding

      info = MoEInfo(all_idxs.shape[-1] // experts_per_tok, experts_per_tok, experts_num)
      return MoEMeta(info, local_ra2a_sort, local_ra2a_isort, preamble, epilogue, local_permute)

  ######################################################################################################################
  # compute ############################################################################################################
  ######################################################################################################################
  _start_fn, _wait_fn = make_ra2a_3d(axis_name=axis_name)

  def load_fn():
    def start_fn(x, meta: MoEMeta):
      assert meta.info.batch_size == x.shape[0], f"Expected {meta.info.batch_size=}, but got {x.shape[0]=}."
      # step 1: gather local tokens for every expert per token
      with jax.named_scope("tokens_to_experts_gather"):
        if gathers == "custom":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = unique_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        elif gathers == "custom_sc":
          x_sort = jnp.repeat(x, experts_per_tok, axis=0)
          x_sort = sc.unique_sc_gather(x_sort, meta.local_ra2a_sort, meta.local_ra2a_isort, ad_mode="gather")
        else:
          x_sort = x[meta.local_ra2a_sort // experts_per_tok, ...]

      # step 2: communicate expert-gathered-tokens to their corresponding expert shards
      with jax.named_scope("ra2a_tokens"):
        total_recv_size = jnp.sum(meta.preamble.recv_sizes)
        # TODO(rdyro): check that (total_recv_size / x.shape[0] * experts_per_token) < safety_factor

        # batch * expert_per_token * safety factor
        bef = round(meta.info.batch_size * meta.info.experts_per_tok * safety_factor)
        bef = ((bef + SPARSECORE_PAD_SIZE - 1) // SPARSECORE_PAD_SIZE) * SPARSECORE_PAD_SIZE

        buffer = jax.lax.empty((bef,) + x.shape[1:], dtype=x.dtype)

        # y = ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(meta.preamble), axis_name=axis_name)
        # return _start_fn(x_sort, buffer, *dataclasses.astuple(meta.preamble))
        return (x_sort, buffer)

    def wait_fn(y, meta: MoEMeta):
      # y = _wait_fn(future)
      # step 3: gather tokens locally so they're expert-contiguous
      with jax.named_scope("local_gather_before"):
        if gathers == "custom":
          y = unique_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.sort_idx, meta.local_permute.isort_idx, ad_mode="gather")
        else:
          y = y[meta.local_permute.sort_idx, ...]
      return y

    return start_fn, wait_fn

  def compute_fn(y: jax.Array, meta: MoEMeta, *args):
    # step 4: perform gmm computation
    local_group_counts = meta.local_permute.group_counts_with_padding
    with jax.named_scope("compute"):
      if compute_block is not None:
        y = compute_block(y, local_group_counts, *args)
    return y

  # def unload_fn(y: jax.Array, meta: MoEMeta):
  def unload_fn():
    def start_fn(y, meta: MoEMeta):
      # step 5: unpermute tokens locally to organize them into chunks in which they arrived
      with jax.named_scope("local_gather_after"):
        if gathers == "custom":
          y = unique_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(y, meta.local_permute.isort_idx, meta.local_permute.isort_idx, ad_mode="scatter")
        else:
          y = y[meta.local_permute.isort_idx, ...]

      # step 6: communincate the chunks back to their origins
      with jax.named_scope("ra2a_results"):
        out = jax.lax.empty((meta.info.batch_size * meta.info.experts_per_tok, *y.shape[1:]), dtype=y.dtype)
        # x_sort = ragged_all_to_all(y, out, *dataclasses.astuple(meta.epilogue), axis_name=axis_name)
      # return _start_fn(y, out, *dataclasses.astuple(meta.epilogue))
      return (y, out)

    def wait_fn(x_sort, meta: MoEMeta):
      # x_sort = _wait_fn(future)
      # step 7: gather so each token repeats are next to each other
      with jax.named_scope("expert_to_tokens_gather"):
        if gathers == "custom":
          y = unique_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        elif gathers == "custom_sc":
          y = sc.unique_sc_gather(x_sort, meta.local_ra2a_isort, meta.local_ra2a_isort, ad_mode="scatter")
        else:
          y = x_sort[meta.local_ra2a_isort, ...]

      # step 8: weigh by expert weights
      with jax.named_scope("reduction_across_experts"):
        y = y.reshape((meta.info.batch_size, experts_per_tok) + y.shape[1:])
        y = jnp.sum(y, axis=1)
      return y
    return start_fn, wait_fn

  return MoEMethods(compute_meta, load_fn, compute_fn, unload_fn)
