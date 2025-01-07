import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, NamedTuple
import time
from collections import defaultdict


# TIMING HELPER FUNCTIONS
class TimeCount:
  def __init__(self):
    self.total_time:float = 0
    self.n_calls:int = 0
  def __str__(self):
    return f"(total_time={self.total_time:14.8f}, n_calls={self.n_calls})"
  def __repr__(self):
    return str(self)
def maketimer():
  return defaultdict(lambda: TimeCount())
def printtimer(times):
  key_len = max(len(k) for k in times)
  kv = sorted([(k,v) for k,v in times.items()], reverse=True, key=lambda t: t[1].total_time)
  for k,v in kv:
    print(f'{k:{key_len}}: {v}')
class Timer:
    def __init__(self, times, name):
        self._record = times
        self._name = name
        self._s = None
        self._e = None
    def __enter__(self):
        self._s = time.time()
    def __exit__(self, *args, **kwargs):
        self._e = time.time()
        self._record[self._name].total_time += self._e - self._s
        self._record[self._name].n_calls += 1
def timeblock(times, name):
    return Timer(times, name)


# REPLAY FUNCTIONS
def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
  """Calculate a list of probabilities from logits using the logsumexp trick to avoid under/overflow."""
  return np.exp(logits - logsumexp(logits))
def logsumexp(x: np.ndarray) -> np.ndarray:
  """Calculates a safe logsumexp operation using the logsumexp trick to avoid under/overflow."""
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))


SEED = 0
MAX_REPLAY_SIZE = int(1e6)
NUM_SAMPLES = int(1e6)
BATCH_SIZE = 5
times = maketimer()

rng_key = jax.random.PRNGKey(SEED)
logits = np.array([0], dtype=np.float32)

for i in range(1, NUM_SAMPLES):
    # add a new logit
    if len(logits) < NUM_SAMPLES:
      with timeblock(times, 'add-new-logit'):
        logits = np.pad(logits, (0, 1))
        logits[-1] = logsumexp(logits) - jnp.log(len(logits)) + ((i % 5) * ((i % 2) * 2 - 1))
    if len(logits) < BATCH_SIZE:
      continue
    # make probabilities
    with timeblock(times, 'make-probabilities'):
      probs = probabilities_from_logits(logits)
    # split the key
    with timeblock(times, 'split-key'):
      rng_key, key = jax.random.split(rng_key)
    # sample
    with timeblock(times, 'sample-with-probs'):
      indices = jax.random.choice(key, len(logits), shape=(BATCH_SIZE,), p=probs)
    with timeblock(times, 'sample-uniform'):
       indices = jax.random.choice(key, len(logits), shape=(BATCH_SIZE,), replace=False)
    if i % 50 == 0:
      print(f'completed iteration {i}, {(i / NUM_SAMPLES * 100):6.2f}%')
    if i % 300 == 0:
      printtimer(times)

"""
sample-uniform    : (total_time=  862.60822821, n_calls=1497)
sample-with-probs : (total_time=  770.46436572, n_calls=1497)
add-new-logit     : (total_time=    1.66802549, n_calls=1500)
split-key         : (total_time=    0.61532307, n_calls=1497)
make-probabilities: (total_time=    0.09771490, n_calls=1497)
"""

print('DONE')
printtimer(times)