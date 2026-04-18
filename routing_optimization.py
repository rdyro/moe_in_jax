from pathlib import Path
import itertools
import time
from typing import Sequence, Any
from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import z3

jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))


def indices_to_mask(s: int, indices: jax.Array):
  return jnp.arange(s)[None, None, :] < indices[:, None, None]


def objective(chunk_sums: jax.Array, mask: jax.Array, cap: int):
  n, _, s = chunk_sums.shape
  masked_sums = jnp.where(mask, chunk_sums, 0)
  C = jnp.sum(masked_sums, axis=-1)
  violations = jnp.maximum(jnp.sum(C, axis=0) - cap, 0)
  indicator = jnp.all(violations <= 0)
  return jnp.where(indicator, jnp.sum(C), -jnp.sum(violations))


def fairness_objective(chunk_sums: jax.Array, mask: jax.Array):
  n, _, s = chunk_sums.shape
  C = jnp.sum(jnp.where(mask, chunk_sums, 0), axis=-1)
  send_totals, recv_totals = jnp.sum(C, axis=-1), jnp.sum(C, axis=0)
  send_unfairness = jnp.max(jnp.abs(n * send_totals - jnp.sum(send_totals)))
  recv_unfairness = jnp.max(jnp.abs(n * recv_totals - jnp.sum(recv_totals)))
  return jnp.maximum(send_unfairness, recv_unfairness)


def get_fairness_objs(chunk_sums: jax.Array, candidates: jax.Array, objs: jax.Array, obj_cutoff: jax.Array):
  n, _, s = chunk_sums.shape
  fairness_obj = jax.vmap(fairness_objective, in_axes=(None, 0))(chunk_sums, candidates)
  fairness_objs = jnp.where(objs >= obj_cutoff, fairness_obj, 2**31 - 1)
  return -fairness_objs


def sample_bernoulli(key: jax.Array, batch_size: int, sample_shape: Sequence[int]):
  ndim, total_shape = len(sample_shape), (batch_size,) + tuple(sample_shape)
  key1, key2 = jax.random.split(key)
  probs = jax.random.uniform(key1, (batch_size,))
  return jax.random.bernoulli(key2, jnp.expand_dims(probs, tuple(range(1, ndim + 1))), total_shape)


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


@partial(jax.jit, static_argnames=("samples",))
def optimize_random(
  key: jax.Array, chunk_sums: jax.Array, cap: int, *, samples: int = 128 * 1024, fairness_cutoff: float | None = None
):
  n, _, s = chunk_sums.shape
  candidates = sample_bernoulli(key, samples, (n, 1, s)).at[0, ...].set(True)
  sample0 = objective(chunk_sums, candidates[0, ...], cap=cap)
  objs = jax.vmap(partial(objective, cap=cap), in_axes=(None, 0))(chunk_sums, candidates)
  if fairness_cutoff is not None:
    objs = get_fairness_objs(chunk_sums, candidates, objs, fairness_cutoff * jnp.max(objs))
  best_guess = candidates[jnp.argmax(objs), ...]
  return 1, best_guess


@partial(jax.jit, static_argnames=("samples", "max_it"))
def optimize_random_iterative(
  key: jax.Array,
  chunk_sums: jax.Array,
  cap: int,
  *,
  samples: int = 1024,
  fairness_cutoff: float | None = None,
  max_it: int = 128,
  mutate_prob: float = 0.05,
):
  n, _, s = chunk_sums.shape
  key, subkey = jax.random.split(key)
  candidates = sample_bernoulli(subkey, samples, (n, 1, s)).at[0, ...].set(True)
  objs = jax.vmap(partial(objective, cap=cap), in_axes=(None, 0))(chunk_sums, candidates)

  best_idx = jnp.argmax(objs)
  best_guess = candidates[best_idx, ...]

  def cond(carry):
    i, _, _, changed = carry
    return (i < max_it) & changed

  def body(carry):
    i, key, best_guess, _ = carry
    key, subkey = jax.random.split(key)

    flip_mask = jax.random.bernoulli(subkey, mutate_prob, (samples, n, 1, s))
    candidates = jnp.where(flip_mask, ~best_guess[None, ...], best_guess[None, ...])
    candidates = candidates.at[0, ...].set(best_guess)

    objs = jax.vmap(partial(objective, cap=cap), in_axes=(None, 0))(chunk_sums, candidates)
    if fairness_cutoff is not None:
      objs = get_fairness_objs(chunk_sums, candidates, objs, fairness_cutoff * jnp.max(objs))
    new_best_idx = jnp.argmax(objs)
    new_best_guess = candidates[new_best_idx, ...]
    changed = jnp.any(best_guess != new_best_guess)

    return (i + 1, key, new_best_guess, changed)

  its, _, best_guess, _ = jax.lax.while_loop(cond, body, (0, key, best_guess, True))
  return its, best_guess


def optimize_z3_indices(
  key: jax.Array,
  chunk_sums: np.ndarray,
  cap: int,
  *,
  max_it: int = 128,
  fairness_cutoff: float | None = None,
):
  del key, fairness_cutoff
  n, _, s = chunk_sums.shape
  chunk_sums = np.array(chunk_sums)
  existing_sums = np.sum(chunk_sums, axis=-1)

  solver = z3.Optimize()

  # Variables
  # diffs[i] is the number of steps sender `i` advances.
  diffs = [z3.Int(f"d_{i}") for i in range(n)]

  # Bounds constraints on variables
  for i in range(n):
    solver.add(diffs[i] >= 0)
    solver.add(diffs[i] <= s)

  new_indices = []
  for i in range(n):
    new_indices.append(diffs[i])

  # Calculate masked values (C_vars)
  C_vars = [[0 for _ in range(n)] for _ in range(n)]
  for send in range(n):
    for recv in range(n):
      for step in range(s):
        # if chunk_sums[send, recv, step] != 0:  # making this matrix irregular hurts performance
        C_vars[send][recv] += z3.If(step < new_indices[send], int(chunk_sums[send, recv, step]), 0)

  # Constraints
  # Cap constraint
  for recv in range(n):
    recv_sum = sum(C_vars[send][recv] for send in range(n))
    solver.add(recv_sum <= cap)
    solver.add(recv_sum > 0)

  # Objectives setup
  send_totals = [sum(C_vars[send][recv] for recv in range(n)) for send in range(n)]
  recv_totals = [sum(C_vars[send][recv] for send in range(n)) for recv in range(n)]

  total_send_sum = sum(send_totals)
  total_recv_sum = sum(recv_totals)

  max_send_dev = z3.Int("max_send_dev")
  max_recv_dev = z3.Int("max_recv_dev")

  # Calculate max deviation from mean (scaled by n to stay integer)
  for send in range(n):
    solver.add(max_send_dev >= n * send_totals[send] - total_send_sum)
    solver.add(max_send_dev >= total_send_sum - n * send_totals[send])

  for recv in range(n):
    solver.add(max_recv_dev >= n * recv_totals[recv] - total_recv_sum)
    solver.add(max_recv_dev >= total_recv_sum - n * recv_totals[recv])

  max_dev = z3.Int("max_dev")
  solver.add(max_dev >= max_send_dev)
  solver.add(max_dev >= max_recv_dev)

  # Objectives
  solver.maximize(total_recv_sum)  # primary objective
  solver.minimize(max_recv_dev)  # secondary objective
  solver.set("timeout", 1000)

  indices = np.zeros(n)
  if (result := solver.check()) in (z3.sat, z3.unknown):
    try:
      model = solver.model()
      indices = np.array([model[diffs[i]].as_long() for i in range(n)], dtype="int32")
    except z3.Z3Exception:
      pass
  return (1 if result == z3.sat else -1), indices_to_mask(s, indices)


def optimize_z3_full(
  key: jax.Array,
  chunk_sums: np.ndarray | list[list[list[int]]],
  cap: int,
  *,
  max_it: int = 128,
  fairness_cutoff: float | None = None,
):
  chunk_sums = np.array(chunk_sums)
  n, _, s = chunk_sums.shape
  del key, fairness_cutoff

  solver = z3.Optimize()

  # We will search over `diffs` mask where each element can be 0 or 1
  # representing whether we flip a specific chunk from False to True
  var_mask = [[z3.Bool(f"d_{i}_{k}") for k in range(s)] for i in range(n)]

  C_vars = [[0 for _ in range(n)] for _ in range(n)]
  for send in range(n):
    for recv in range(n):
      for step in range(s):
        C_vars[send][recv] += z3.If(var_mask[send][step], 1, 0) * int(chunk_sums[send, recv, step])

  # Cap constraint
  send_totals = [sum(C_vars[send][recv] for recv in range(n)) for send in range(n)]
  recv_totals = [sum(C_vars[send][recv] for send in range(n)) for recv in range(n)]
  for recv in range(n):
    solver.add(recv_totals[recv] <= cap)

  total_send_sum = sum(send_totals)
  total_recv_sum = sum(recv_totals)

  max_send_dev = z3.Int("max_send_dev")
  max_recv_dev = z3.Int("max_recv_dev")

  for send in range(n):
    solver.add(max_send_dev >= n * send_totals[send] - total_send_sum)
    solver.add(max_send_dev >= total_send_sum - n * send_totals[send])

  for recv in range(n):
    solver.add(max_recv_dev >= n * recv_totals[recv] - total_recv_sum)
    solver.add(max_recv_dev >= total_recv_sum - n * recv_totals[recv])

  max_dev = z3.Int("max_dev")
  solver.add(max_dev >= max_send_dev)
  solver.add(max_dev >= max_recv_dev)

  solver.maximize(total_recv_sum)
  solver.minimize(max_dev)
  solver.set("timeout", 1000)

  mask = np.zeros((n, n, s), dtype=bool)
  result = solver.check()
  if result in (z3.sat, z3.unknown):
    try:
      model = solver.model()
      for send in range(n):
        for recv in range(n):
          for step in range(s):
            mask[send, recv, step] = bool(model[var_mask[send][step]])
      return 1, np.array(mask)
    except z3.Z3Exception:
      pass
  return -1, mask


def optimize_lp_full(
  key: jax.Array,
  chunk_sums: np.ndarray | list[list[list[int]]],
  cap: int,
  *,
  max_it: int = 128,
  fairness_cutoff: float | None = None,
):
  import scipy.optimize as opt
  chunk_sums = np.array(chunk_sums)
  n, _, s = chunk_sums.shape

  S = np.sum(chunk_sums, axis=1)

  c = [-float(S[i, k]) for i in range(n) for k in range(s)]
  A_ub = [[float(chunk_sums[i, j, k]) for i in range(n) for k in range(s)] for j in range(n)]
  b_ub = [float(cap) for _ in range(n)]

  res = opt.linprog(np.array(c), A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, 1), integrality=1)

  mask = np.zeros((n, n, s), dtype=bool)
  if not res.success:
    return -1, mask

  if fairness_cutoff is None:
    mask[...] = np.clip(res.x, 0, 1).reshape(n, 1, s) > 0.5
    return 1, mask

  opt_vol = -res.fun
  if opt_vol <= 0:
    return 1, mask

  # Phase 2: Maximize fairness (minimize max_dev)
  V = n * s
  c2 = [0.0] * V + [1.0]
  A_ub2 = [row + [0.0] for row in A_ub]
  b_ub2 = list(b_ub)

  # Cutoff constraint: total_vol >= cutoff * opt_vol => -total_vol <= -cutoff * opt_vol
  A_ub2.append(c + [0.0])
  b_ub2.append(-float(fairness_cutoff) * opt_vol)

  # Deviation constraints
  for i_target in range(n):
    row_pos = [float((n - 1) * S[i, k] if i == i_target else -S[i, k]) for i in range(n) for k in range(s)]
    A_ub2.append(row_pos + [-1.0])
    b_ub2.append(0.0)

    row_neg = [float(-(n - 1) * S[i, k] if i == i_target else S[i, k]) for i in range(n) for k in range(s)]
    A_ub2.append(row_neg + [-1.0])
    b_ub2.append(0.0)

  for j_target in range(n):
    row_pos = [float(n * chunk_sums[i, j_target, k] - S[i, k]) for i in range(n) for k in range(s)]
    A_ub2.append(row_pos + [-1.0])
    b_ub2.append(0.0)

    row_neg = [float(-(n * chunk_sums[i, j_target, k] - S[i, k])) for i in range(n) for k in range(s)]
    A_ub2.append(row_neg + [-1.0])
    b_ub2.append(0.0)

  bounds2 = [(0, 1)] * V + [(0, None)]
  integrality2 = [1] * V + [0]

  res2 = opt.linprog(np.array(c2), A_ub=np.array(A_ub2), b_ub=np.array(b_ub2), bounds=bounds2, integrality=integrality2)

  if res2.success:
    mask[...] = np.clip(res2.x[:-1], 0, 1).reshape(n, 1, s) > 0.5
    return 1, mask

  mask[...] = np.clip(res.x, 0, 1).reshape(n, 1, s) > 0.5
  return 1, mask


@jax.jit
def optimize_indices(key: jax.Array, chunk_sums: jax.Array, cap, *, max_it=128, fairness_cutoff: float | None = None):
  del key
  n, _, s = chunk_sums.shape
  indices = jnp.broadcast_to(initialize_opt(chunk_sums, cap=cap)[None], (n,))
  assert indices.shape == (n,)

  obj_fn = lambda chunk_sums, indices: objective(chunk_sums, indices_to_mask(s, indices), cap=cap)
  cond = lambda carry: (carry[0] < max_it) & jnp.any(carry[1] != carry[2])

  def body(carry):
    i, indices, _ = carry
    diffs = steps_to_consider(n)
    candidates = jnp.clip(indices[None, :] + diffs, 0, s)
    objs = jax.vmap(obj_fn, in_axes=(None, 0))(chunk_sums, candidates)

    objs = objs + jnp.sum((candidates - indices) / (n + 1) / 2, axis=-1)  # tie breaking
    objs = jnp.where(objs > 0, (2 * n + 1) * objs + jnp.sum(candidates - indices, axis=-1), objs)
    if fairness_cutoff is not None:
      masks = jax.vmap(partial(indices_to_mask, s))(candidates)
      objs = get_fairness_objs(chunk_sums, masks, objs, fairness_cutoff * jnp.max(objs))
    new_indices = candidates[jnp.argmax(objs), :]
    return (i + 1, new_indices, indices)

  it, indices, _ = jax.lax.while_loop(cond, body, (0, indices, indices - 1))
  return it, indices_to_mask(s, indices)


@partial(jax.jit, static_argnames=("solver", "opts"))
def find_partition(key: jax.Array, chunk_sums: jax.Array, capacity, solver=None, opts: tuple[str, Any] = ()):
  n, _, s = chunk_sums.shape
  assert chunk_sums.shape == (n, n, s)
  solver = solver if solver is not None else optimize_random_iterative

  def find_masks(carry):
    i, key, chunk_sums, prev_mask, all_masks = carry
    next_key, key = jax.random.split(key)
    it, mask = solver(key, chunk_sums, capacity, **dict(opts))
    assert mask.shape == (n, 1, s)
    mask = mask & (~prev_mask)
    chunk_sums = jnp.where(mask, 0, chunk_sums)
    all_masks = all_masks.at[i, ...].set(mask)
    return (i + 1, key, chunk_sums, mask | prev_mask, all_masks)

  all_masks = jnp.zeros((32, n, 1, s), dtype=bool)
  cond = lambda carry: (carry[0] < 32) & (jnp.sum(carry[2]) > 0)
  existing_mask = jnp.zeros((n, 1, s), dtype=bool)
  it, _, _, _, all_masks = jax.lax.while_loop(cond, find_masks, (0, key, chunk_sums, existing_mask, all_masks))
  return it, all_masks


class RoutingOptimizationTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ("optimize_indices", optimize_indices, "optimize_indices", None),
    ("optimize_z3_indices", optimize_z3_indices, "optimize_z3_indices", None),
    ("optimize_z3_full", optimize_z3_full, "optimize_z3_full", None),
    ("optimize_lp_full", optimize_lp_full, "optimize_lp_full", None),
    ("optimize_random_iterative", optimize_random_iterative, "optimize_random_iterative", None),
    ("optimize_indices_fair", optimize_indices, "optimize_indices", 0.95),
    ("optimize_z3_indices_fair", optimize_z3_indices, "optimize_z3_indices", 0.95),
    ("optimize_z3_full_fair", optimize_z3_full, "optimize_z3_full", 0.95),
    ("optimize_random_fair", optimize_random, "optimize_random", 0.95),
    ("optimize_lp_full_fair", optimize_lp_full, "optimize_lp_full_fair", 0.95),
    ("optimize_random_iterative_fair", optimize_random_iterative, "optimize_random_iterative", 0.95),
  )
  def test_optimization_loop_perf(self, opt_method, method_name, fairness_cutoff):
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    total = 128
    num_splits = 16
    chunk_sums = jnp.round(
      total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)

    it, var, mask, capacity = 1, None, None, 1.5
    for i in range(10):
      # with jax.profiler.trace(f"/tmp/profile_{method_name}"):
      t = time.perf_counter()
      it, mask = opt_method(next(keys), chunk_sums, int(128 * capacity), fairness_cutoff=fairness_cutoff)
      jax.block_until_ready((mask, var))
      t = time.perf_counter() - t
      # if method_name == "optimize_lp_full":
      #   breakpoint()

      assert (mask.shape[0], mask.shape[2]) == (chunk_sums.shape[0], chunk_sums.shape[2])
      C = np.sum(chunk_sums * mask, axis=-1)
      print(f"recv sizes = {np.sum(C, 0)}; send sizes = {np.sum(C, -1)}; {int(it) = } time = {t:.4e} s")

      chunk_sums = np.where(mask, 0, chunk_sums)
      if np.sum(chunk_sums) == 0:
        print(f"Optimizer: {method_name}_{'fair' if fairness_cutoff is not None else ''} converges in {i + 1} its")
        break

  @parameterized.named_parameters(
    ("optimize_indices", optimize_indices, "optimize_indices"),
    ("optimize_z3_indices", optimize_z3_indices, "optimize_z3_indices"),
    ("optimize_z3_full", optimize_z3_full, "optimize_z3_full"),
    ("optimize_lp_full", optimize_lp_full, "optimize_lp_full"),
    ("optimize_random", optimize_random, "optimize_random"),
    ("optimize_random_iterative", optimize_random_iterative, "optimize_random_iterative"),
  )
  def test_optimization_loop_converges(self, opt_method, method_name):
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    total = 128
    num_splits = 16
    chunk_sums = jnp.round(
      total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)

    it, indices, capacity, mask = 1, None, 1.5, None
    for i in range(10):
      it, mask = opt_method(next(keys), chunk_sums, int(128 * capacity))
      assert (mask.shape[0], mask.shape[2]) == (chunk_sums.shape[0], chunk_sums.shape[2])
      C = jnp.sum(chunk_sums * mask, axis=-1)
      print(f"Recv sizes = {jnp.sum(chunk_sums * mask, axis=(-1, 0))}; {int(it) = }")
      chunk_sums = jnp.where(mask, 0, chunk_sums)
      if jnp.sum(chunk_sums) == 0:
        print(f"Optimizer: {method_name} converges in {i + 1} iterations")
        break
    self.assertEqual(jnp.sum(chunk_sums), 0)

  @parameterized.named_parameters(
    ("optimize_indices", optimize_indices, "optimize_indices", None),
    ("optimize_lp_full", optimize_lp_full, "optimize_lp_full", None),
    ("optimize_lp_full_fair", optimize_lp_full, "optimize_lp_full", {"fairness_cutoff": [0.95]}),
    ("optimize_random", optimize_random, "optimize_random", {"samples": [128, 256]}),
    (
      "optimize_random_iterative",
      optimize_random_iterative,
      "optimize_random_iterative",
      {"samples": [128, 256], "mutate_prob": [0.05, 0.1], "max_it": [8, 16, 32]},
    ),
  )
  def test_efficacy(self, opt_method, method_name, kwargs_dict):
    WITH_PROFILING = False
    seed = int(time.time_ns()) % (2**31)
    keys = iter(jax.random.split(jax.random.key(seed), 1024))
    num_shards = 4
    per_shard_total = 128
    num_splits = 16
    chunk_sums = jnp.round(
      per_shard_total * jax.nn.softmax(jax.random.gumbel(next(keys), (num_shards, num_shards, num_splits)), axis=-1)
    ).astype(jnp.int32)
    capacity = per_shard_total * 1.5

    TRIALS = 32

    @partial(jax.jit, static_argnames=("kwargs",))
    def check_efficacy(kwargs):
      kwargs = dict(kwargs)

      def find_its(key, _):
        new_key, key = jax.random.split(key)
        mask = jnp.zeros_like(chunk_sums, dtype=bool)
        all_masks = jnp.zeros((32,) + chunk_sums.shape, dtype=bool)
        opts = tuple(kwargs.items())
        it, all_masks = find_partition(key, chunk_sums, capacity=capacity, solver=opt_method, opts=opts)
        return new_key, it

      _, its = jax.lax.scan(find_its, jax.random.key(0), None, length=TRIALS)
      return jnp.stack([jnp.arange(8), jnp.bincount(its, length=8)], 0)

    if WITH_PROFILING:
      import tune_jax

      tune_jax.logger.setLevel("INFO")
      if kwargs_dict is None:
        fn = tune_jax.tune(partial(check_efficacy, ()))
        it_hist = fn()
        elapsed = fn.timing_results[0].t_mean / TRIALS
        print(f"For {method_name} histogram is:\n{it_hist}\nin {elapsed:.4e} s")
      else:
        dict_keys, values = zip(*kwargs_dict.items(), strict=True)
        for v in itertools.product(*values):
          kwargs = tuple(zip(dict_keys, v, strict=True))
          fn = tune_jax.tune(partial(check_efficacy, kwargs))
          it_hist = fn()
          elapsed = fn.timing_results[0].t_mean / TRIALS
          print(f"For {method_name} with {kwargs} histogram is:\n{it_hist}\nin {elapsed:.4e} s")
    else:
      if kwargs_dict is None:
        it_hist = check_efficacy()
        print(f"For {method_name} histogram is:\n{it_hist}")
      else:
        dict_keys, values = zip(*kwargs_dict.items(), strict=True)
        for v in itertools.product(*values):
          kwargs = tuple(zip(dict_keys, v, strict=True))
          it_hist = check_efficacy(kwargs)
          print(f"For {method_name} with {kwargs} histogram is:\n{it_hist}")


if __name__ == "__main__":
  absltest.main()
