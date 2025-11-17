from functools import partial
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .utils import empty


@partial(
  jax.tree_util.register_dataclass,
  data_fields=["input_offsets", "send_sizes", "output_offsets", "recv_sizes"],
  meta_fields=[],
)
@dataclasses.dataclass
class RA2AMeta:
  input_offsets: jax.Array
  send_sizes: jax.Array
  output_offsets: jax.Array
  recv_sizes: jax.Array


def make_compute_metadata(axis_name, experts_num, safety_factor: int = 2):
  @partial(jax.shard_map, out_specs=P("x", None, None), check_vma=False)
  def compute_metadata(x: jax.Array, all_idxs: jax.Array):
    shard_idx, num_shards = jax.lax.axis_index(axis_name), jax.lax.axis_size(axis_name)
    experts_per_shard = experts_num // num_shards
    experts_per_tok = all_idxs.size // x.shape[0] // num_shards  # because all_idxs is replicated

    if all_idxs.ndim == 1:
      all_idxs = all_idxs.reshape((num_shards, -1))

    # compute the ra2a communication ###################################################################################

    all_sizes = jax.vmap(partial(jnp.bincount, length=num_shards))(all_idxs // experts_per_shard)
    all_input_offsets = jnp.cumsum(all_sizes, axis=-1) - all_sizes  # cumsum from 0
    all_output_offsets = jnp.cumsum(all_sizes, axis=0) - all_sizes  # cumsum from 0
    send_sizes, recv_sizes = all_sizes[shard_idx, :], all_sizes[:, shard_idx]
    input_offsets, output_offsets = all_input_offsets[shard_idx, :], all_output_offsets[shard_idx, :]
    preamble = RA2AMeta(input_offsets, send_sizes, output_offsets, recv_sizes)
    inv_input_offsets = all_output_offsets[:, shard_idx]  # we send back chunks starting where we received them
    inv_output_offsets = all_input_offsets[:, shard_idx]  # we write the chunks from where they originally came
    inv_send_sizes, inv_recv_sizes = recv_sizes, send_sizes
    epilogue = RA2AMeta(inv_input_offsets, inv_send_sizes, inv_output_offsets, inv_recv_sizes)

    # compute within expert shard sort; receiving several locally sorted chunks ########################################

    # get gather indices for pre-ra2a by-expert-shard organization
    local_ra2a_sort = jnp.argsort(all_idxs[shard_idx, :])
    local_ra2a_isort = jnp.argsort(local_ra2a_sort)

    all_expert_idxs = jax.lax.all_gather(all_idxs[shard_idx, :][local_ra2a_sort], axis_name)

    local_expert_idxs = empty((x.shape[0] * experts_per_tok * safety_factor,), jnp.int32)

    def update_fn(i, local_expert_idxs):
      update = jnp.roll(all_expert_idxs[i, :], -all_input_offsets[i, shard_idx])
      return jax.lax.dynamic_update_slice_in_dim(local_expert_idxs, update, all_output_offsets[i, shard_idx], 0)

    local_expert_idxs = jax.lax.fori_loop(0, num_shards, update_fn, local_expert_idxs)

    # ra2a_sort = jnp.argsort(all_idxs, axis=-1)
    # all_expert_idxs = jnp.take_along_axis(all_idxs, ra2a_sort, axis=-1)  # expensive
    # all_shard_assignment = all_expert_idxs // experts_per_shard
    # local_mask = ((all_shard_assignment >= shard_idx) & (all_shard_assignment < (shard_idx + 1))).reshape(-1)
    # local_pack_idx = jnp.where(local_mask, size=local_mask.size, fill_value=0)  # expensive
    # local_expert_idxs = all_expert_idxs.reshape(-1)[local_pack_idx]  # expensive
    # local_expert_idxs = local_expert_idxs[:x.shape[0] * experts_per_tok * safety_factor]
    # mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
    # jax.debug.print("diff = {}", jnp.sum(jnp.abs(jnp.where(mask, local_expert_idxs - local_expert_idxs_, 0))))

    local_pack_mask = jnp.arange(local_expert_idxs.size) < jnp.sum(recv_sizes)
    local_sort = jnp.argsort(jnp.where(local_pack_mask, local_expert_idxs, 2**30))
    local_isort = jnp.argsort(local_sort)
    local_group_sizes = jnp.bincount(
      jnp.where(local_pack_mask, local_expert_idxs - shard_idx * experts_per_shard, 2**30), length=experts_per_shard
    )

    # perform the actual communication and computation #################################################################

    # step 1: gather local tokens for every expert per token
    x_sort = x[local_ra2a_sort // experts_per_tok, ...]

    # step 2: communicate expert-gathered-tokens to their corresponding expert shards
    # buffer = jnp.empty((x.shape[0] * experts_per_tok * safety_factor,) + x.shape[1:], dtype=x.dtype)
    buffer = empty((x.shape[0] * experts_per_tok * safety_factor,) + x.shape[1:], dtype=x.dtype)
    y = jax.lax.ragged_all_to_all(x_sort, buffer, *dataclasses.astuple(preamble), axis_name=axis_name)

    # step 3: gather tokens locally so they're expert-contiguous
    y = y[local_sort, ...]

    # step 4: perform gmm computation
    pass

    # step 5: unpermute tokens locally to organize them into chunks in which they arrived
    y = y[local_isort, ...]

    # step 6: communincate the chunks back to their origins
    # out = jnp.empty((x.shape[0] * experts_per_tok,) + x.shape[1:], dtype=x.dtype)
    out = empty((x.shape[0] * experts_per_tok,) + x.shape[1:], dtype=x.dtype)
    x_sort = jax.lax.ragged_all_to_all(y, out, *dataclasses.astuple(epilogue), axis_name=axis_name)

    # step 7: gather so each token repeats are next to each other
    x = x_sort[local_ra2a_isort, ...].reshape((x.shape[0], experts_per_tok) + x.shape[1:])

    # step 8: weigh by expert weights
    pass

    return x

  return compute_metadata
