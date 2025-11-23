"""Simple all-gather simulator for ragged_all_to_all with extensive debug callbacks."""


import jax
import jax.numpy as jnp

from .utils import RA2AMeta


def assert_fn(x):
  assert x


def ragged_all_to_all(
    x: jax.Array, out: jax.Array,
    input_offsets: jax.Array, send_sizes: jax.Array,
    output_offsets: jax.Array, recv_sizes: jax.Array,
    *, axis_name: str = "x", validate_input: bool = False
) -> jax.Array:

  meta = RA2AMeta(input_offsets=input_offsets, send_sizes=send_sizes,
                  output_offsets=output_offsets, recv_sizes=recv_sizes)

  axis_index = jax.lax.axis_index(axis_name)
  x_all = jax.lax.all_gather(x, axis_name=axis_name, tiled=False, axis=0)
  meta_all: RA2AMeta = jax.lax.all_gather(meta, axis_name=axis_name, tiled=False, axis=0)

  if validate_input:
    # check that send_sizes and recv_sizes match
    all_send_recv_match = jnp.all(meta_all.send_sizes[:, axis_index] == meta_all.recv_sizes[axis_index, :])
    jax.debug.callback(assert_fn, all_send_recv_match)

    # check actual receive sizes wrt to the output buffer
    total_recv = jnp.sum(meta_all.recv_sizes[axis_index, :])
    jax.debug.callback(assert_fn, total_recv < out.shape[0])
    max_el_pos = jnp.max(meta_all.output_offsets[:, axis_index] + meta_all.send_sizes[:, axis_index])
    jax.debug.callback(assert_fn, max_el_pos < out.shape[0])
    min_el_pos = jnp.min(meta_all.output_offsets[:, axis_index])
    jax.debug.callback(assert_fn, min_el_pos >= 0)

  def insert(i, buf):
    existing_slice = jax.lax.dynamic_slice_in_dim(buf, meta_all.output_offsets[i, axis_index], x.shape[0], axis=0)
    x_ = jnp.roll(x_all[i, ...], -meta_all.input_offsets[i, axis_index], axis=0)
    mask = jnp.arange(x.shape[0]) < meta_all.send_sizes[i, axis_index]
    mask = jnp.expand_dims(mask, axis=tuple(range(1, x.ndim)))
    new_slice = jnp.where(mask, x_, existing_slice)
    buf = jax.lax.dynamic_update_slice_in_dim(buf, new_slice, meta_all.output_offsets[i, axis_index], axis=0)
    return buf

  # make the buffer larger by x since we use x-sized slice in the dynamic update slice so that we don't wrap around
  buffer_for_update_ = jnp.concatenate([out, jnp.zeros_like(out, shape=(x.shape[0],) + out.shape[1:])], 0)
  updated_buffer = jax.lax.fori_loop(0, x_all.shape[0], insert, buffer_for_update_)
  updated_buffer = updated_buffer[:out.shape[0], ...]

  return updated_buffer
