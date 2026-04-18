from functools import partial
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


def steps_to_consider(s: int):
  total = 3**s
  divs = 3 ** jnp.arange(s)
  return jax.lax.rem(jnp.arange(total)[:, None] // divs, 3) - 1


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
  # jax.debug.print("candidates = {}", jnp.mean(candidates, axis=(1, 2, 3)))
  sample0 = objective(chunk_sums, candidates[0, ...], cap=cap, balance_cap=balance_cap)
  # jax.debug.print("Candidate 0 satisfies = {}", sample0)

  objs = jax.vmap(partial(objective, cap=cap, balance_cap=balance_cap), in_axes=(None, 0))(chunk_sums, candidates)
  best_guess = candidates[jnp.argmax(objs), ...]
  return best_guess

  # steps = min(steps, s)
  # def body(_, val):
  #   key, best_guess = val
  #   key, next_key = jax.random.split(key)
  #   # perturbs = jax.random.randint(key, (batch_size, n, n, s), -1, 2, dtype=jnp.int8)
  #   perturbs = jax.random.randint(key, (batch_size, n, n, s), 0, 2, dtype=jnp.int8).astype(bool)
  #   candidates = jnp.logical_xor(best_guess[None, ...], perturbs)
  #   objs = jax.vmap(objective, in_axes=(None, 0, None, None))(chunk_sums, candidates, cap, balance_cap=balance_cap)
  #   best_obj = jnp.max(objs)
  #   best_guess = jnp.where(best_obj > 0, candidates[jnp.argmax(objs), ...], best_guess)
  #   return next_key, best_guess
  # return jax.lax.fori_loop(0, 16, body, (key, jnp.zeros((n, n, s), dtype=bool)))[1]


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


if __name__ == "__main__":
  seed = int(time.time_ns()) % (2**31)
  keys = iter(jax.random.split(jax.random.key(seed), 1024))
  num_shards = 4
  total = 128
  num_splits = 32
  chunk_sums = jnp.round(
    total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
  ).astype(jnp.int32)

  print(f"Total destination: {jnp.sum(chunk_sums, axis=(-1, 0))}")

  indices = None
  mask = jnp.ones_like(chunk_sums, dtype=bool)
  chunk_sums_ = chunk_sums
  capacity = 1.5
  for _ in range(5):
    it, indices = optimize(chunk_sums_, int(128 * capacity), initial_guess=indices)
    t = time.perf_counter()
    it, indices = optimize(chunk_sums_, int(128 * capacity), initial_guess=indices)
    mask = indices_to_mask(chunk_sums.shape[-1], indices)
    t = time.perf_counter() - t

    # t = time.perf_counter()
    # mask = jax.block_until_ready(
    #   optimize_random(next(keys), chunk_sums_, int(128 * capacity), batch_size=256, steps=8)
    # )
    # t = time.perf_counter() - t

    print(f"optimization took {t:.4e} s")
    print("send amounts =", jnp.sum(jnp.sum(chunk_sums_ * mask, axis=-1), -1))
    print("recv amounts =", jnp.sum(jnp.sum(chunk_sums_ * mask, axis=-1), 0))
    chunk_sums_ = chunk_sums_ * ~mask
    # print(it, indices)
    print("-" * 80)
    if jnp.sum(chunk_sums_) == 0:
      break

  def find_new_mask(key, carry, bs, steps):
    i, chunk_sums_, prev_mask, all_masks, _ = carry
    mask = jax.block_until_ready(optimize_random(key, chunk_sums_, int(128 * capacity), batch_size=bs, steps=steps))
    mask = mask & (~prev_mask)
    # jax.debug.print("recv amounts = {}", jnp.sum(jnp.sum(chunk_sums_ * mask, axis=-1), 0))
    chunk_sums_ = jnp.where(mask, 0, chunk_sums_)
    all_masks = all_masks.at[i, ...].set(mask)
    done = jnp.sum(chunk_sums_) == 0
    return (i + 1, chunk_sums_, mask | prev_mask, all_masks, done)

  @partial(jax.jit, static_argnames=("bs", "steps"))
  def find_masks(key, chunk_sums: jax.Array, bs, steps):
    all_masks = jnp.zeros((32, *chunk_sums.shape), dtype=bool)
    cond = lambda carry: (carry[0] < 32) & (~carry[-1])
    it, _, _, all_masks, _ = jax.lax.while_loop(
      cond,
      partial(find_new_mask, key, bs=bs, steps=steps),
      (0, chunk_sums, jnp.zeros_like(chunk_sums, dtype=bool), all_masks, False),
    )
    # masks = jax.lax.scan(find_new_mask, (chunk_sums, jnp.zeros_like(chunk_sums, dtype=bool)), None, length=32)[1]
    return it, all_masks

  @partial(jax.jit, static_argnames=("bs", "steps"))
  def test_efficacy(key, chunk_sums, bs, steps):
    keys = jax.random.split(key, 128)
    _, (its, _) = jax.lax.scan(lambda _, key: (None, find_masks(key, chunk_sums, bs, steps)), None, keys)
    return its

  it, masks = jax.block_until_ready(find_masks(next(keys), chunk_sums, 512, 8))
  t = time.perf_counter()
  it, masks = jax.block_until_ready(find_masks(next(keys), chunk_sums, 512, 8))
  t = time.perf_counter() - t
  print(f"scan took {t:.4e} s")
  print(f"recv sizes = {jnp.sum(masks * chunk_sums[None, ...], axis=(-1, -3))[:3, :]}")
  print(f"it = {it}")
  # print(1 * masks[:3, :, :, :])
  its = test_efficacy(next(keys), chunk_sums, 512, 8)
  print(jnp.bincount(its, length=8))

  @partial(jax.jit, static_argnames=("bs", "steps"))
  def experiment(key, bs, steps):
    its = test_efficacy(next(keys), chunk_sums, bs, steps)
    return jnp.bincount(its, length=8)

  key = next(keys)
  for bs in [16, 32, 64, 128, 256, 512]:
    for steps in [2, 4, 8, 16]:
      key, exp_key = jax.random.split(key)
      print(f"{bs = } {steps = } {experiment(exp_key, bs, steps)}")
