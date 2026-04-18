from absl.testing import absltest
from absl.testing import parameterized
from functools import partial
import itertools
import time

import jax
import jax.numpy as jnp


def indices_to_mask(s: int, indices: jax.Array):
  return jnp.arange(s)[None, None, :] < indices[:, None, None]


def is_valid(chunk_sums: jax.Array, cap: int | jax.Array):
  return jnp.all(jnp.sum(chunk_sums, axis=(-1, 0)) <= cap)


def objective(chunk_sums: jax.Array, mask: jax.Array, cap: int, balance_cap: int | None = None):
  n, _, s = chunk_sums.shape
  masked_sums = jnp.where(mask, chunk_sums, 0)
  C = jnp.sum(masked_sums, axis=-1)
  indicator = jnp.all(jnp.sum(C, axis=0) <= cap)
  if balance_cap is not None:
    indicator &= jnp.all(C <= (balance_cap // n))
  return jnp.where(indicator, jnp.sum(C), -(2**31))


def fairness(chunk_sums: jax.Array, mask: jax.Array):
  masked_sums = jnp.where(mask, chunk_sums, 0)
  C = jnp.sum(masked_sums, axis=-1)
  send_totals = jnp.sum(C, axis=-1)
  recv_totals = jnp.sum(C, axis=0)
  send_fairness = jnp.max(jnp.abs(send_totals - jnp.mean(send_totals)))
  recv_fairness = jnp.max(jnp.abs(recv_totals - jnp.mean(recv_totals)))
  return -jnp.maximum(send_fairness, recv_fairness)


def steps_to_consider(s: int, up_only: bool = False):
  base = 2 if up_only else 3
  shift = 0 if up_only else -1
  total = base**s
  divs = base ** jnp.arange(s)
  return jax.lax.rem(jnp.arange(total)[:, None] // divs, base) + shift


def initialize_opt(chunk_sums: jax.Array, cap: int):
  n, _, s = chunk_sums.shape
  indices = jnp.arange(s)[:, None]
  obj_fn = lambda chunk_sums, indices: objective(chunk_sums, indices_to_mask(s, indices), cap=cap)
  obj = jax.vmap(obj_fn, in_axes=(None, 0))(chunk_sums, indices)
  best_inital = indices[jnp.argmax(obj), 0]
  return best_inital


@partial(jax.jit, static_argnames=("steps", "batch_size"))
def optimize_random(
  key: jax.Array,
  chunk_sums: jax.Array,
  cap: int,
  balance_cap: int | None = None,
  batch_size: int = 1024,
  steps: int = 8,
):
  n, _, s = chunk_sums.shape
  steps = max(min(steps, s), 1)
  samples = jax.random.randint(key, (batch_size, steps, n, n, s), 0, steps + 1, dtype=jnp.int8)
  candidates = (samples >= (jnp.arange(steps) + 1)[None, :, None, None, None]).astype(bool).reshape((-1, n, n, s))
  candidates = candidates.at[0, ...].set(True)
  sample0 = objective(chunk_sums, candidates[0, ...], cap=cap, balance_cap=balance_cap)

  objs = jax.vmap(partial(objective, cap=cap, balance_cap=balance_cap), in_axes=(None, 0))(chunk_sums, candidates)
  best_guess = candidates[jnp.argmax(objs), ...]
  return best_guess


@jax.jit
def optimize(
  chunk_sums: jax.Array,
  cap,
  balance_cap: int | None = None,
  max_it=128,
  initial_guess=None,
  fairness_cutoff: float | None = None,
):
  n, _, s = chunk_sums.shape
  if initial_guess is not None:
    indices = initial_guess
  else:
    indices = jnp.broadcast_to(initialize_opt(chunk_sums, cap=cap)[None], (n,))

  floor = 0 if initial_guess is None else initial_guess

  obj_fn = lambda chunk_sums, indices: objective(
    chunk_sums, indices_to_mask(s, indices), cap=cap, balance_cap=balance_cap
  )
  cond = lambda carry: (carry[0] < max_it) & jnp.any(carry[1] != carry[2])

  def body(carry):
    i, indices, _ = carry
    diffs = steps_to_consider(n)
    candidates = jnp.clip(indices[None, :] + diffs, floor, s)
    objs = jax.vmap(obj_fn, in_axes=(None, 0))(chunk_sums, candidates)

    objs = objs + jnp.sum((candidates - indices) / (n + 1) / 2, axis=-1)  # tie breaking
    objs = jnp.where(objs > 0, (2 * n + 1) * objs + jnp.sum(candidates - indices, axis=-1), objs)

    if fairness_cutoff is not None:
      raise NotImplementedError("This path doesn't work.")
    else:
      best_idx = jnp.argmax(objs)

    new_indices = candidates[best_idx, :]
    return (i + 1, new_indices, indices)

  it, indices, _ = jax.lax.while_loop(cond, body, (0, indices, indices - 1))
  return it, indices


@jax.jit
def optimize_fairness(
  chunk_sums: jax.Array,
  cap: int,
  balance_cap: int | None = None,
  max_it: int = 128,
  initial_guess: jax.Array | None = None,
):
  n, _, s = chunk_sums.shape
  if initial_guess is not None:
    indices = initial_guess
  else:
    indices = jnp.broadcast_to(initialize_opt(chunk_sums, cap=cap)[None], (n,))

  floor = 0 if initial_guess is None else initial_guess

  def obj_fn(chunk_sums, indices):
    mask = indices_to_mask(s, indices)
    masked_sums = jnp.where(mask, chunk_sums, 0)
    C = jnp.sum(masked_sums, axis=-1)
    indicator = jnp.all(jnp.sum(C, axis=0) <= cap)
    if balance_cap is not None:
      indicator &= jnp.all(C <= (balance_cap // n))
    return jnp.where(indicator, fairness(chunk_sums, mask), -jnp.inf)

  cond = lambda carry: (carry[0] < max_it) & jnp.any(carry[1] != carry[2])

  def body(carry):
    i, indices, _ = carry
    diffs = steps_to_consider(n, up_only=True)
    candidates = jnp.clip(indices[None, :] + diffs, floor, s)
    objs = jax.vmap(obj_fn, in_axes=(None, 0))(chunk_sums, candidates)

    progress = jnp.sum(candidates - indices, axis=-1)
    # Prefer making progress. If no progress is the only valid option, it stays.
    objs = jnp.where(
        objs != -jnp.inf,
        objs + jnp.where(progress > 0, 1e6, 0) + progress * 1e-3,
        -jnp.inf
    )

    best_idx = jnp.argmax(objs)
    new_indices = candidates[best_idx, :]
    return (i + 1, new_indices, indices)

  it, indices, _ = jax.lax.while_loop(cond, body, (0, indices, indices - 1))
  return it, indices


@jax.jit
def optimize_step_by_step(
  chunk_sums: jax.Array,
  cap: int,
  balance_cap: int | None = None,
  max_it: int = 128,
  initial_mask: jax.Array | None = None,
):
  n, _, s = chunk_sums.shape
  if initial_mask is not None:
    mask = initial_mask
  else:
    mask = jnp.zeros_like(chunk_sums, dtype=bool)

  def obj_fn(mask):
    masked_sums = jnp.where(mask, chunk_sums, 0)
    C = jnp.sum(masked_sums, axis=-1)
    indicator = jnp.all(jnp.sum(C, axis=0) <= cap)
    if balance_cap is not None:
      indicator &= jnp.all(C <= (balance_cap // n))
    return jnp.where(indicator, fairness(chunk_sums, mask), -jnp.inf)

  cond = lambda carry: (carry[0] < max_it) & jnp.any(carry[1] != carry[2])

  def body(carry):
    i, mask, _ = carry
    num_elements = n * n * s
    eyes = jnp.eye(num_elements, dtype=bool).reshape((num_elements, n, n, s))
    candidates = mask[None, ...] | eyes
    valid_flip = (~mask) & (chunk_sums > 0)
    valid_flip_flat = valid_flip.flatten()
    objs = jax.vmap(obj_fn)(candidates)
    objs = jnp.where(valid_flip_flat, objs, -jnp.inf)
    best_idx = jnp.argmax(objs)
    best_obj = objs[best_idx]
    new_mask = jnp.where(best_obj > -jnp.inf, candidates[best_idx], mask)
    return (i + 1, new_mask, mask)

  it, mask, _ = jax.lax.while_loop(cond, body, (0, mask, jnp.ones_like(mask)))
  return it, mask


class RoutingOptimizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
    ("optimize", optimize, "optimize"),
    ("optimize_fairness", optimize_fairness, "optimize_fairness"),
    ("optimize_step_by_step", optimize_step_by_step, "optimize_step_by_step"),
  )
  def test_optimization_loop(self, opt_method, method_name):
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    total = 128
    num_splits = 32
    chunk_sums = jnp.round(
      total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)

    indices, mask = None, None
    capacity = 1.5
    print(f"Optimizer: {method_name}")
    for _ in range(5):
      if method_name == "optimize_step_by_step":
        it, mask = opt_method(chunk_sums, int(128 * capacity), initial_mask=mask)
        indices = 1  # mark as not None
      else:
        it, indices = opt_method(chunk_sums, int(128 * capacity), initial_guess=indices)
        mask = indices_to_mask(chunk_sums.shape[-1], indices)
      print(f"Recv sizes = {jnp.sum(chunk_sums * mask, axis=(-1, 0))}; {it = }")

      chunk_sums = jnp.where(mask, 0, chunk_sums)
      if jnp.sum(chunk_sums) == 0:
        break

  @parameterized.named_parameters(
    ("optimize", optimize, "optimize"),
    ("optimize_fairness", optimize_fairness, "optimize_fairness"),
    ("optimize_step_by_step", optimize_step_by_step, "optimize_step_by_step"),
  )
  def test_optimization_loop_converges(self, opt_method, method_name):
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    total = 128
    num_splits = 32
    chunk_sums = jnp.round(
      total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)

    indices = None
    mask = jnp.ones_like(chunk_sums, dtype=bool)
    chunk_sums_ = chunk_sums
    capacity = 1.5
    for _ in range(64):
      if method_name == "optimize_step_by_step":
        it, mask = opt_method(chunk_sums_, int(128 * capacity), initial_mask=mask if indices is not None else None)
        indices = 1  # mark as not None
      else:
        it, indices = opt_method(chunk_sums_, int(128 * capacity), initial_guess=indices)
        mask = indices_to_mask(chunk_sums.shape[-1], indices)

      chunk_sums_ = chunk_sums_ * ~mask
      if jnp.sum(chunk_sums_) == 0:
        break

    self.assertEqual(jnp.sum(chunk_sums_), 0)

  @parameterized.named_parameters(
    ("optimize", optimize, "optimize", None),
    ("optimize_fairness", optimize_fairness, "optimize_fairness", None),
    ("optimize_step_by_step", optimize_step_by_step, "optimize_step_by_step", None),
    ("optimize_random", optimize_random, "optimize_random", {"batch_size": [16, 32], "steps": [2, 4]}),
  )
  def test_efficacy(self, opt_method, method_name, kwargs_dict):
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    total = 128
    num_splits = 32
    chunk_sums = jnp.round(
      total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)
    capacity = 1.5

    def check_efficacy(kwargs):
      def find_new_mask(key, carry):
        i, chunk_sums_, prev_mask, all_masks, indices_or_mask, _ = carry

        if method_name == "optimize_random":
          mask = opt_method(key, chunk_sums_, int(128 * capacity), **kwargs)
        elif method_name == "optimize_step_by_step":
          _, mask = opt_method(chunk_sums_, int(128 * capacity), initial_mask=indices_or_mask)
        else:
          _, indices = opt_method(chunk_sums_, int(128 * capacity), initial_guess=indices_or_mask)
          mask = indices_to_mask(chunk_sums.shape[-1], indices)
          indices_or_mask = indices

        mask = mask & (~prev_mask)
        chunk_sums_ = jnp.where(mask, 0, chunk_sums_)
        all_masks = all_masks.at[i, ...].set(mask)
        done = jnp.sum(chunk_sums_) == 0

        # update state for next iteration
        if method_name == "optimize_step_by_step":
          next_indices_or_mask = mask
        elif method_name == "optimize_random":
          next_indices_or_mask = jnp.zeros(chunk_sums.shape[0], dtype=jnp.int32)
        else:
          next_indices_or_mask = indices_or_mask

        return (i + 1, chunk_sums_, mask | prev_mask, all_masks, next_indices_or_mask, done)

      all_masks = jnp.zeros((32, *chunk_sums.shape), dtype=bool)
      cond = lambda carry: (carry[0] < 32) & (~carry[-1])

      init_indices_or_mask = jnp.zeros(chunk_sums.shape[0], dtype=jnp.int32)
      if method_name == "optimize_step_by_step":
         init_indices_or_mask = jnp.zeros_like(chunk_sums, dtype=bool)

      it, _, _, all_masks, _, _ = jax.lax.while_loop(
        cond,
        partial(find_new_mask, jax.random.key(1)),
        (0, chunk_sums, jnp.zeros_like(chunk_sums, dtype=bool), all_masks, init_indices_or_mask, False),
      )
      return it, all_masks

    if kwargs_dict is None:
      it, masks = check_efficacy({})
    else:
      dict_keys, values = zip(*kwargs_dict.items(), strict=True)
      for v in itertools.product(*values):
        kwargs = dict(zip(dict_keys, v, strict=True))
        it, masks = check_efficacy(kwargs)


if __name__ == "__main__":
  absltest.main()
