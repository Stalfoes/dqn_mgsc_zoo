import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, TypeVar, Generic
import time
from collections import defaultdict

from dqn_zoo.replay import Transition, compress_array


def encoder(transition):
  return transition._replace(
    s_tm1=compress_array(transition.s_tm1),
    s_t=compress_array(transition.s_t),
  )
def make_dummy_transition(state, next_state):
  return encoder(
    Transition(
      s_tm1=np.full((84, 84, 4), state, dtype=np.uint8),
      a_tm1=int(state % 3),
      r_t=float(0.1 * (state % 3)),
      discount_t=float(0.95),
      s_t=np.full((84, 84, 4), next_state, dtype=np.uint8)
    )
  )


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

# RNGKey HELPER
def PRNGKeyIter(seed):
  rng_key = jax.random.PRNGKey(seed)
  while True:
    rng_key, _ = jax.random.split(rng_key)
    yield rng_key

# REPLAY FUNCTIONS
#@jax.jit
def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
  """Calculate a list of probabilities from logits using the logsumexp trick to avoid under/overflow."""
  return np.exp(logits - logsumexp(logits))
#@jax.jit
def logsumexp(x: np.ndarray) -> np.ndarray:
  """Calculates a safe logsumexp operation using the logsumexp trick to avoid under/overflow."""
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))


ItemType = TypeVar('ItemType')
class CircularBuffer(Generic[ItemType]):
  def __init__(self, capacity:int):
    self._list = [None] * capacity
    self._capacity = capacity
    self._size = 0
    self._left_head = 0
    self._right_head = 0
  @property
  def capacity(self) -> int:
    return self._capacity
  @property
  def size(self) -> int:
    return self._size
  def is_full(self) -> bool:
    return self._size == self._capacity
  def add(self, item:ItemType) -> None:
    if self.is_full():
      raise BufferError("Buffer is full and cannot be added to. Pop an item first.")
    self._list[self._right_head] = item
    self._right_head = self._right_head + 1
    # self._right_head = (self._right_head + 1) % self.capacity
    if self._right_head == self._capacity:
      self._right_head = 0
    self._size += 1
  def popleft(self) -> ItemType:
    if self._size == 0:
      raise BufferError("Buffer is empty and cannot be popped from. Add an item first.")
    item = self._list[self._left_head]
    # self._list[self._left_head] = None
    self._left_head = self._left_head + 1
    # self._left_head = (self._left_head + 1) % self.capacity
    if self._left_head == self._capacity:
      self._left_head = 0
    self._size -= 1
    return item
  def __getitem__(self, key:int) -> ItemType:
    if self._size == 0:
      raise BufferError("Buffer is empty and cannot be indexed. Add an item first.")
    idx = (self._left_head + key) % self._capacity
    return self._list[idx]
  def __str__(self):
    left = self._left_head
    right = self._right_head
    if left == right and self.is_full():
      right += self.capacity
    if right < left:
      right += self.capacity
    return str([self._list[i % self.capacity] for i in range(self._left_head, self._right_head)])
  def __repr__(self):
    return repr(self._list)
  def __len__(self):
    return self.size

class CircularLogitBuffer:
  def __init__(self, capacity:int):
    self._logits = np.full((capacity,), -np.inf, dtype=np.float32)
    self._capacity = capacity
    self._size = 0
    self._left_head = 0
    self._right_head = 0
  @property
  def capacity(self) -> int:
    return self._capacity
  @property
  def size(self) -> int:
    return self._size
  def is_full(self) -> bool:
    return self._size == self._capacity
  def add(self, item:Optional[np.float32]=None) -> None:
    if self.is_full():
      raise BufferError("Buffer is full and cannot be added to. Pop an item first.")
    if item is None:
      if self._size == 0:
        item = 0
      else:
        item = logsumexp(self._logits) - np.log(self._size)
    self._logits[self._right_head] = item
    # self._right_head = (self._right_head + 1) % self.capacity
    self._right_head = self._right_head + 1
    if self._right_head == self._capacity:
      self._right_head = 0
    self._size += 1
  def popleft(self) -> np.float32:
    if self._size == 0:
      raise BufferError("Buffer is empty and cannot be popped from. Add an item first.")
    item = self._logits[self._left_head]
    self._logits[self._left_head] = -np.inf
    # self._left_head = (self._left_head + 1) % self.capacity
    self._left_head = self._left_head + 1
    if self._left_head == self._capacity:
      self._left_head = 0
    self._size -= 1
    return item
  def __getitem__(self, key):
    if self._size == 0:
      raise BufferError("Buffer is empty and cannot be indexed. Add an item first.")
    if (key >= self._size).any():
      raise KeyError(f"Buffer is not large enough to index at position {key}. Must be in [0,{self.size-1}).")
    idx = (self._left_head + key) % self._capacity
    return self._logits[idx]
  def __str__(self):
    return repr(self._logits)
  def __repr__(self):
    return repr(self._logits)
  def as_probs(self) -> np.ndarray:
    return probabilities_from_logits(self._logits)
  def sample(self, rng_key:jnp.ndarray, size:int) -> jnp.ndarray:
    if self._size < size:
      raise BufferError(f"Cannot sample from buffer with length {self._size} when requested sample size was {size}.")
    absolute_indices = jax.random.choice(rng_key, self._capacity, shape=(size,), p=self.as_probs())
    indices = (absolute_indices - self._left_head) % self._capacity
    return indices
  def sample_uniform(self, rng_key:jnp.ndarray, size:int, replace:bool=True):
    if self._size < size:
      raise BufferError(f"Cannot sample from buffer with length {self._size} when requested sample size was {size}.")
    relative_indices = jax.random.choice(rng_key, self._size, shape=(size,), replace=replace)
    return relative_indices

def test_basic():
  SEED = 0
  MAX_REPLAY_SIZE = int(10)
  BATCH_SIZE = 5

  rng_key = jax.random.PRNGKey(SEED)
  logits = CircularLogitBuffer(capacity=MAX_REPLAY_SIZE)

  logits.add(5)
  print(logits)
  print('\t', logits.as_probs())
  logits.add(7)
  print(logits)
  print('\t', logits.as_probs())
  logits.popleft()
  print(logits)
  logits.add(0)
  for _ in range(logits.size, logits.capacity):
    logits.add()
  print(logits)
  print('\t', logits.as_probs())
  print('\t', logits.sample(rng_key, BATCH_SIZE))
  print('\t', logits.sample_uniform(rng_key, BATCH_SIZE))

  print('\n------------------')
  logits = CircularLogitBuffer(capacity=MAX_REPLAY_SIZE)
  logits.add()
  logits.popleft()

  for _ in range(logits.size, logits.capacity):
    logits.add()
  print(logits)
  logits._logits[logits._left_head] = 99
  print(logits)
  print('\t', logits.sample(rng_key, BATCH_SIZE))

def test_timing():
  from collections import deque, OrderedDict
  SEED = 0
  REPLAY_SIZE = int(1e6)
  N_SAMPLES = int(1e3)
  N_REPLACEMENTS = int(1e3)
  BATCH_SIZE = 5

  TIMER = maketimer()
  RNG_KEY = PRNGKeyIter(SEED)
  
  # DEQUE
  # with timeblock(TIMER, 'deque-init'):
  #   deque_replay = deque()
  #   np_logits = np.array((), dtype=np.float32)
  # for i in range(REPLAY_SIZE):
  #   transition = make_dummy_transition(i % 200, (i % 200) + 1)
    # with timeblock(TIMER, 'deque-add-transition'):
    #   deque_replay.append(transition)
    # with timeblock(TIMER, 'np-add-logit'):
    #   if len(np_logits) == 0:
    #     new_logit = 0
    #   else:
    #     new_logit = logsumexp(np_logits) - np.log(len(np_logits))
    #   np_logits = np.pad(np_logits, (0, 1))
    #   np_logits[len(np_logits) - 1] = new_logit
  # for i in range(N_SAMPLES):
  #   with timeblock(TIMER, 'deque-sample'):
  #     probs = probabilities_from_logits(np_logits)
  #     indices = jax.random.choice(next(RNG_KEY), REPLAY_SIZE, shape=(BATCH_SIZE,), p=probs)
  #     for idx in indices:
  #       _ = deque_replay[idx]
  # for i in range(N_REPLACEMENTS):
  #   new_item = make_dummy_transition(i % 200, (i % 200) + 1)
  #   with timeblock(TIMER, 'deque-replace'):
  #     deque_replay.popleft()
  #     deque_replay.append(new_item)
  # for i in range(N_REPLACEMENTS):
  #   with timeblock(TIMER, 'np-logit-replace'):
  #     np_logits = np.delete(np_logits, len(np_logits) - 1)
  #     new_logit = logsumexp(np_logits) - np.log(len(np_logits))
  #     np_logits = np.pad(np_logits, (0, 1))
  #     np_logits[len(np_logits) - 1] = new_logit
  # deque_replay.clear(); del deque_replay
  # TIMER['deque-replace'].total_time += TIMER['np-logit-replace'].total_time
  # print('done timing deque')
  
  # ORDERED-DICT
  # with timeblock(TIMER, 'ordered-dict-init'):
  #   ordered_dict_replay = OrderedDict()
  #   current_id = 0
  # for i in range(REPLAY_SIZE):
  #   transition = make_dummy_transition(i % 200, (i % 200) + 1)
  #   with timeblock(TIMER, 'ordered-dict-add-transition'):
  #     ordered_dict_replay[i] = transition
  # for i in range(N_SAMPLES):
  #   with timeblock(TIMER, 'ordered-dict-sample'):
  #     probs = probabilities_from_logits(np_logits)
  #     indices = jax.random.choice(next(RNG_KEY), REPLAY_SIZE, shape=(BATCH_SIZE,), p=probs)
  #     for idx in indices:
  #       _ = ordered_dict_replay[int(idx)]
  # for i in range(N_REPLACEMENTS):
  #   new_item = make_dummy_transition(i % 200, (i % 200) + 1)
  #   with timeblock(TIMER, 'ordered-dict-replace'):
  #     ordered_dict_replay.popitem(last=False)
  #     ordered_dict_replay[i + REPLAY_SIZE] = new_item
  # TIMER['ordered-dict-replace'].total_time += TIMER['np-logit-replace'].total_time
  # ordered_dict_replay.clear(); del ordered_dict_replay
  # print('done timing ordereddict')
  
  # LIST
  # with timeblock(TIMER, 'list-init'):
  #   list_replay = []
  # for i in range(REPLAY_SIZE):
  #   transition = make_dummy_transition(i % 200, (i % 200) + 1)
  #   with timeblock(TIMER, 'list-add-transition'):
  #     list_replay.append(transition)
  # for i in range(N_SAMPLES):
  #   with timeblock(TIMER, 'list-sample'):
  #     probs = probabilities_from_logits(np_logits)
  #     indices = jax.random.choice(next(RNG_KEY), REPLAY_SIZE, shape=(BATCH_SIZE,), p=probs)
  #     for idx in indices:
  #       _ = list_replay[idx]
  # for i in range(N_REPLACEMENTS):
  #   new_item = make_dummy_transition(i % 200, (i % 200) + 1)
  #   with timeblock(TIMER, 'list-replace'):
  #     list_replay.pop(0)
  #     list_replay.append(new_item)
  # TIMER['list-replace'].total_time += TIMER['np-logit-replace'].total_time
  # list_replay.clear(); del list_replay
  # print('done timing list')

  # del np_logits

  # CIRCULAR REPLAY
  with timeblock(TIMER, 'circular-init'):
    circular_replay:CircularBuffer[Transition] = CircularBuffer(REPLAY_SIZE)
    circular_logits = CircularLogitBuffer(REPLAY_SIZE)
  for i in range(REPLAY_SIZE):
    transition = make_dummy_transition(i % 200, (i % 200) + 1)
    with timeblock(TIMER, 'circular-replay-add-transition'):
      circular_replay.add(transition)
    with timeblock(TIMER, 'circular-replay-add-logit'):
      circular_logits.add()
  for i in range(N_SAMPLES):
    with timeblock(TIMER, 'circular-replay-sample'):
      indices = circular_logits.sample(next(RNG_KEY), BATCH_SIZE)
      for idx in indices:
        _ = circular_replay[idx]
  for i in range(N_REPLACEMENTS):
    new_item = make_dummy_transition(i % 200, (i % 200) + 1)
    with timeblock(TIMER, 'circular-replay-replace'):
      circular_replay.popleft()
      circular_replay.add(new_item)
      circular_logits.popleft()
      circular_logits.add()
  circular_replay._list.clear(); del circular_replay; del circular_logits
  print('done timing circularbuffer')

  printtimer(TIMER)
  """
  circular-replay-add-logit     : (total_time= 1944.93601847, n_calls=1000000)
  np-add-logit                  : (total_time= 1542.17577267, n_calls=1000000)
  deque-sample                  : (total_time=   11.94535089, n_calls=1000)
  circular-replay-sample        : (total_time=    8.10895872, n_calls=1000)
  list-sample                   : (total_time=    7.83584642, n_calls=1000)
  ordered-dict-sample           : (total_time=    7.74131107, n_calls=1000)
  circular-replay-add-transition: (total_time=    3.48490024, n_calls=1000000)
  list-replace                  : (total_time=    3.03555465, n_calls=1000)
  ordered-dict-replace          : (total_time=    2.67604661, n_calls=1000)
  deque-replace                 : (total_time=    2.67538714, n_calls=1000)
  np-logit-replace              : (total_time=    2.67439938, n_calls=1000)
  circular-replay-replace       : (total_time=    1.93055844, n_calls=1000)
  deque-add-transition          : (total_time=    0.96062350, n_calls=1000000)
  ordered-dict-add-transition   : (total_time=    0.64717484, n_calls=1000000)
  list-add-transition           : (total_time=    0.49294543, n_calls=1000000)
  circular-init                 : (total_time=    0.00531530, n_calls=1)
  deque-init                    : (total_time=    0.00001478, n_calls=1)
  ordered-dict-init             : (total_time=    0.00000668, n_calls=1)
  list-init                     : (total_time=    0.00000381, n_calls=1)

  CIRCULAR
    circular-init                 : (total_time=    0.00531530, n_calls=1)          
    circular-replay-add-transition: (total_time=    3.48490024, n_calls=1000000)    highest
    circular-replay-add-logit     : (total_time= 1944.93601847, n_calls=1000000)    highest
    circular-replay-replace       : (total_time=    1.93055844, n_calls=1000)       lowest
    circular-replay-sample        : (total_time=    8.10895872, n_calls=1000)
  CIRCULAR w/ slight changes
    circular-init                 : (total_time=    0.00533438, n_calls=1)
    circular-replay-add-transition: (total_time=    2.31454062, n_calls=1000000)    lower             highest
    circular-replay-add-logit     : (total_time= 1902.81359863, n_calls=1000000)    lower             highest
    circular-replay-replace       : (total_time=    1.90660238, n_calls=1000)        slightly lower    lowest
    circular-replay-sample        : (total_time=   12.05650997, n_calls=1000)       higher but should be the exact same
  DEQUE
    deque-init                    : (total_time=    0.00001478, n_calls=1)
    deque-add-transition          : (total_time=    0.96062350, n_calls=1000000)
    np-add-logit                  : (total_time= 1542.17577267, n_calls=1000000)    equal
    deque-replace                 : (total_time=    2.67538714, n_calls=1000)
      np-logit-replace              : (total_time=    2.67439938, n_calls=1000)
    deque-sample                  : (total_time=   11.94535089, n_calls=1000)       highest
  ORDERED-DICT
    ordered-dict-init             : (total_time=    0.00000668, n_calls=1)
    ordered-dict-add-transition   : (total_time=    0.64717484, n_calls=1000000)
    np-add-logit                  : (total_time= 1542.17577267, n_calls=1000000)    equal
    ordered-dict-replace          : (total_time=    2.67604661, n_calls=1000)
      np-logit-replace              : (total_time=    2.67439938, n_calls=1000)
    ordered-dict-sample           : (total_time=    7.74131107, n_calls=1000)       lowest
  LIST
    list-init                     : (total_time=    0.00000381, n_calls=1)
    list-add-transition           : (total_time=    0.49294543, n_calls=1000000)    lowest
    np-add-logit                  : (total_time= 1542.17577267, n_calls=1000000)    equal
    list-replace                  : (total_time=    3.03555465, n_calls=1000)       highest
      np-logit-replace              : (total_time=    2.67439938, n_calls=1000) 
    list-sample                   : (total_time=    7.83584642, n_calls=1000)
  """

if __name__ == '__main__':
  # test_basic()
  test_timing()