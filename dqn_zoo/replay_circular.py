# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Replay components for DQN-type agents."""

# pylint: disable=g-bad-import-order

import collections
import typing
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import dm_env
import numpy as np
import snappy

import jax
import jax.numpy as jnp

from dqn_zoo import parts

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(typing.NamedTuple):
  s_tm1: Optional[np.ndarray]
  a_tm1: Optional[parts.Action]
  r_t: Optional[float]
  discount_t: Optional[float]
  s_t: Optional[np.ndarray]


# DEBUGGING operations
def check_valid_pytree(pytree) -> None:
  leaves, _ = jax.tree_util.tree_flatten(pytree)
  for leaf in leaves:
    check_valid_array(leaf)

def check_valid_array(arr: np.ndarray) -> None:
  if jnp.isnan(arr).any():
    raise ValueError(f"Array contains a NaN. {arr=}, {arr.shape=}, {np.isnan(arr).any()=}, {(arr == np.inf).any()=}, {(arr == -np.inf).any()=}, {arr.max()=}, {arr.min()=}")
  if not jnp.isfinite(arr).all():
    raise ValueError(f"Array contains not-finite values. {arr=}, {arr.shape=}, {np.isnan(arr).any()=}, {(arr == np.inf).any()=}, {(arr == -np.inf).any()=}, {arr.max()=}, {arr.min()=}")

def check_valid_value(value) -> None:
  if jnp.isnan(value):
    raise ValueError(f"Value is a NaN. {value=}, {np.isnan(value)=}, {(value == np.inf)=}, {(value == -np.inf)=}")
  if not jnp.isfinite(value):
    raise ValueError(f"Value is not finite. {value=}, {np.isnan(value)=}, {(value == np.inf)=}, {(value == -np.inf)=}")


# PROBABILITY CALCULATING OPERATIONS
# The issue with JIT-compiling these operations is that we call them with different length logits
# So it would re-compile like a million times
def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
  """Calculate a list of probabilities from logits using the logsumexp trick to avoid under/overflow."""
  return np.exp(logits - logsumexp(logits))

def logsumexp(x: np.ndarray) -> np.ndarray:
  """Calculates a safe logsumexp operation using the logsumexp trick to avoid under/overflow."""
  c = x.max()
  return c + np.log(np.sum(np.exp(x - c)))

# If these equal the functions above, then the functions above also get JIT compiled!
def JNPprobabilities_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
  """Calculate a list of probabilities from logits using the logsumexp trick to avoid under/overflow."""
  return jnp.exp(logits - JNPlogsumexp(logits))

def JNPlogsumexp(x: jnp.ndarray) -> jnp.ndarray:
  """Calculates a safe logsumexp operation using the logsumexp trick to avoid under/overflow."""
  c = x.max()
  return c + jnp.log(jnp.sum(jnp.exp(x - c)))


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
  def __len__(self):
    return self._size
  def get_state(self) -> Mapping[str,Any]:
    state = {
      'list': self._list,
      'capacity': self._capacity,
      'size': self._size,
      'left_head': self._left_head,
      'right_head': self._right_head,
    }
    return state
  def set_state(self, state:Mapping[str, Any]) -> None:
    self._list = state['list']
    self._capacity = state['capacity']
    self._size = state['size']
    self._left_head = state['left_head']
    self._right_head = state['right_head']

class CircularLogitBuffer:
  def __init__(self, capacity:int, random_state:np.random.Generator):
  # def __init__(self, capacity:int, rng_key:jnp.ndarray):
    self._logits = np.full((capacity,), -np.inf, dtype=np.float32)
    self._capacity = capacity
    self._size = 0
    self._left_head = 0
    self._right_head = 0
    self._rng_state = random_state
    # self._rng_key = rng_key
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
  def __setitem__(self, key, item):
    if (key >= self._size).any():
      raise KeyError(f"Buffer is not large enough to index at position {key}. Must be in [0,{self.size-1}).")
    idx = (self._left_head + key) % self._capacity
    self._logits[idx] = item
  def as_probs(self) -> np.ndarray:
    return probabilities_from_logits(self._logits)
  def sample(self, size:int) -> jnp.ndarray:
    if self._size < size:
      raise BufferError(f"Cannot sample from buffer with length {self._size} when requested sample size was {size}.")
    absolute_indices = self._rng_state.choice(self._capacity, size=size, p=self.as_probs())
    # self._rng_key, key = jax.random.split(self._rng_key)
    # probs = self.as_probs()
    # try:
    #   absolute_indices = jax.random.choice(key, self._capacity, shape=(size,), p=probs)
    # except Exception as e:
    #   print(f'FAILURE NON-UNIFORM\n=========================\n\t{self._rng_key=}\n\t{type(self._rng_key)=}\n\t{key=}\n\t{type(key)=}\n\t{self._capacity=}\n\t{type(self._capacity)=}\n\t{size=}\n\t{type(size)=}\n\t{probs=}\n\t{type(probs)=}\n=========================')
    #   raise e
    indices = (absolute_indices - self._left_head) % self._capacity
    return indices
  def sample_uniform(self, size:int, replace:bool=True) -> jnp.ndarray:
    if self._size < size:
      raise BufferError(f"Cannot sample from buffer with length {self._size} when requested sample size was {size}.")
    relative_indices = self._rng_state.choice(self._size, size=size, replace=replace)
    # self._rng_key, key = jax.random.split(self._rng_key)
    # try:
    #   relative_indices = jax.random.choice(key, self._size, shape=(size,), replace=replace)
    # except Exception as e:
    #   print(f'FAILURE UNIFORM\n=========================\n\t{self._rng_key=}\n\t{type(self._rng_key)=}\n\t{key=}\n\t{type(key)=}\n\t{self._size=}\n\t{type(self._size)=}\n\t{size=}\n\t{type(size)=}\n\t{replace=}\n\t{type(replace)=}\n=========================')
    #   raise e
    return relative_indices
  def get_state(self) -> Mapping[str, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
      'capacity': self._capacity,
      'logits': self._logits,
      'size': self._size,
      'left_head': self._left_head,
      'right_head': self._right_head,
      'rng_state': self._rng_state,
      # 'rng_key': self._rng_key,
    }
  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._capacity = state['capacity']
    self._logits = state['logits']
    self._size = state['size']
    self._left_head = state['left_head']
    self._right_head = state['right_head']
    self._rng_state = state['rng_state']
    # self._rng_key = state['rng_key']


class UniformDistribution:
  """Provides uniform sampling of user-defined integer IDs."""

  def __init__(self, random_state: np.random.RandomState):
    self._random_state = random_state
    self._ids = []  # IDs in a contiguous indexable format for sampling.
    self._id_to_index = {}  # User ID -> index into self._ids.

  def add(self, ids: Sequence[int]) -> None:
    """Adds new IDs."""
    for i in ids:
      if i in self._id_to_index:
        raise IndexError('Cannot add ID %d, it already exists.' % i)

    for i in ids:
      idx = len(self._ids)
      self._id_to_index[i] = idx
      self._ids.append(i)

  def remove(self, ids: Sequence[int]) -> None:
    """Removes existing IDs."""
    for i in ids:
      if i not in self._id_to_index:
        raise IndexError('Cannot remove ID %d, it does not exist.' % i)

    for i in ids:
      idx = self._id_to_index[i]
      # Swap ID to be removed with ID at the end of self._ids.
      self._ids[idx], self._ids[-1] = self._ids[-1], self._ids[idx]
      self._id_to_index[self._ids[idx]] = idx  # Update index for swapped ID.
      self._id_to_index.pop(self._ids.pop())  # Remove ID from data structures.

  def sample(self, size: int) -> np.ndarray:
    """Returns sample of IDs, uniformly sampled."""
    indices = self._random_state.randint(self.size, size=size)
    ids = np.fromiter(
        (self._ids[idx] for idx in indices), dtype=np.int64, count=len(indices)
    )
    return ids

  def ids(self) -> Iterable[int]:
    """Returns an iterable of all current IDs."""
    return self._id_to_index.keys()

  @property
  def size(self) -> int:
    """Number of IDs currently tracked."""
    return len(self._ids)

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
        'ids': self._ids,
        'id_to_index': self._id_to_index,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._ids = state['ids']
    self._id_to_index = state['id_to_index']

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if len(self._ids) != len(self._id_to_index):
      return False, 'ids and id_to_index should be the same size.'
    if len(self._ids) != len(set(self._ids)):
      return False, 'IDs should be unique.'
    if len(self._id_to_index.values()) != len(set(self._id_to_index.values())):
      return False, 'Indices should be unique.'
    for i in self._ids:
      if self._ids[self._id_to_index[i]] != i:
        return False, 'ID %d should map to itself.' % i
    # Indices map to themselves because of previous check and uniqueness.
    return True, ''


class TransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with LIFO storage for flat named tuples."""

  def __init__(
      self,
      capacity: int,
      structure: ReplayStructure,
      random_state: np.random.RandomState,
      encoder: Optional[Callable[[ReplayStructure], Any]] = None,
      decoder: Optional[Callable[[Any], ReplayStructure]] = None,
  ):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._distribution = UniformDistribution(random_state=random_state)
    self._storage = collections.OrderedDict()  # ID -> item.
    self._t = 0  # Used to generate unique IDs for each item.

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    if self.size == self._capacity:
      oldest_id, _ = self._storage.popitem(last=False)
      self._distribution.remove([oldest_id])

    item_id = self._t
    self._distribution.add([item_id])
    self._storage[item_id] = self._encoder(item)
    self._t += 1

  def get(self, ids: Sequence[int]) -> Iterable[ReplayStructure]:
    """Retrieves items by IDs."""
    for i in ids:
      yield self._decoder(self._storage[i])

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay uniformly, with replacement."""
    ids = self._distribution.sample(size)
    samples = self.get(ids)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  def ids(self) -> Iterable[int]:
    """Get IDs of stored transitions, for testing."""
    return self._storage.keys()

  @property
  def size(self) -> int:
    """Number of items currently contained in the replay."""
    return len(self._storage)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        # Serialize OrderedDict as a simpler, more common data structure.
        'storage': list(self._storage.items()),
        't': self._t,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = collections.OrderedDict(state['storage'])
    self._t = state['t']
    self._distribution.set_state(state['distribution'])

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if self._t < len(self._storage):
      return False, 't should be >= storage size.'
    if set(self._storage.keys()) != set(self._distribution.ids()):
      return False, 'IDs in storage and distribution do not match.'
    return self._distribution.check_valid()


class ReservoirTransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with Reservoir Sampling storage for flat named tuples."""

  def __init__(
      self,
      capacity: int,
      structure: ReplayStructure,
      random_state: np.random.RandomState,
      encoder: Optional[Callable[[ReplayStructure], Any]] = None,
      decoder: Optional[Callable[[Any], ReplayStructure]] = None,
  ):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._distribution = UniformDistribution(random_state=random_state)
    self._storage = collections.OrderedDict()  # ID -> item.
    self._t = 0  # Used to generate unique IDs for each item.

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    if self.size == self._capacity:
      # Perform Algorithm R
      j = self._random_state.randint(0, self._t)
      if j < self.size:
        # kick and replace
        item_id = j
        self._storage[item_id] = self._encoder(item)
      else:
        pass # do nothing, don't add it or replace anything
    else:
      # Fill the buffer up to capacity
      item_id = self._t
      self._distribution.add([item_id])
      self._storage[item_id] = self._encoder(item)
    self._t += 1

  def get(self, ids: Sequence[int]) -> Iterable[ReplayStructure]:
    """Retrieves items by IDs."""
    for i in ids:
      yield self._decoder(self._storage[i])

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay uniformly, with replacement."""
    ids = self._distribution.sample(size)
    samples = self.get(ids)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  def ids(self) -> Iterable[int]:
    """Get IDs of stored transitions, for testing."""
    return self._storage.keys()

  @property
  def size(self) -> int:
    """Number of items currently contained in the replay."""
    return len(self._storage)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        # Serialize OrderedDict as a simpler, more common data structure.
        'storage': list(self._storage.items()),
        't': self._t,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = collections.OrderedDict(state['storage'])
    self._t = state['t']
    self._distribution.set_state(state['distribution'])

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if self._t < len(self._storage):
      return False, 't should be >= storage size.'
    if set(self._storage.keys()) != set(self._distribution.ids()):
      return False, 'IDs in storage and distribution do not match.'
    return self._distribution.check_valid()


def _power(base, exponent) -> np.ndarray:
  """Same as usual power except `0 ** 0` is zero."""
  # By default 0 ** 0 is 1 but we never want indices with priority zero to be
  # sampled, even if the priority exponent is zero.
  base = np.asarray(base)
  return np.where(base == 0.0, 0.0, base**exponent)


def importance_sampling_weights(
    probabilities: np.ndarray,
    uniform_probability: float,
    exponent: float,
    normalize: bool,
  ) -> np.ndarray:
  """Calculates importance sampling weights from given sampling probabilities.

  Args:
    probabilities: Array of sampling probabilities for a subset of items. Since
      this is a subset the probabilites will typically not sum to `1`.
    uniform_probability: Probability of sampling an item if uniformly sampling.
    exponent: Scalar that controls the amount of importance sampling correction
      in the weights. Where `1` corrects fully and `0` is no correction
      (resulting weights are all `1`).
    normalize: Whether to scale all weights so that the maximum weight is `1`.
      Can be enabled for stability since weights will only scale down.

  Returns:
    Importance sampling weights that can be used to scale the loss. These have
    the same shape as `probabilities`.
  """
  if not 0.0 <= exponent <= 1.0:
    raise ValueError('Require 0 <= exponent <= 1.')
  if not 0.0 <= uniform_probability <= 1.0:
    raise ValueError('Expected 0 <= uniform_probability <= 1.')

  weights = (uniform_probability / probabilities) ** exponent
  if normalize:
    weights /= np.max(weights)
  if not np.isfinite(weights).all():
    raise ValueError('Weights are not finite: %s.' % weights)
  return weights


class SumTree:
  """A binary tree where non-leaf nodes are the sum of child nodes.

  Leaf nodes contain non-negative floats and are set externally. Non-leaf nodes
  are the sum of their children. This data structure allows O(log n) updates and
  O(log n) queries of which index corresponds to a given sum. The main use
  case is sampling from a multinomial distribution with many probabilities
  which are updated a few at a time.
  """

  def __init__(self):
    """Initializes an empty `SumTree`."""
    # When there are n values, the storage array will have size 2 * n. The first
    # n elements are non-leaf nodes (ignoring the very first element), with
    # index 1 corresponding to the root node. The next n elements are leaf nodes
    # that contain values. A non-leaf node with index i has children at
    # locations 2 * i, 2 * i + 1.
    self._size = 0
    self._storage = np.zeros(0, dtype=np.float64)
    self._first_leaf = 0  # Boundary between non-leaf and leaf nodes.

  def resize(self, size: int) -> None:
    """Resizes tree, truncating or expanding with zeros as needed."""
    self._initialize(size, values=None)

  def get(self, indices: Sequence[int]) -> np.ndarray:
    """Gets values corresponding to given indices."""
    indices = np.asarray(indices)
    if not ((0 <= indices) & (indices < self.size)).all():
      raise IndexError('index out of range, expect 0 <= index < %s' % self.size)
    return self.values[indices]

  def set(self, indices: Sequence[int], values: Sequence[float]) -> None:
    """Sets values at the given indices."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.0).any():
      raise ValueError('value must be finite and positive.')
    self.values[indices] = values
    storage = self._storage
    for idx in np.asarray(indices) + self._first_leaf:
      parent = idx // 2
      while parent > 0:
        # At this point the subtree with root parent is consistent.
        storage[parent] = storage[2 * parent] + storage[2 * parent + 1]
        parent //= 2

  def set_all(self, values: Sequence[float]) -> None:
    """Sets many values all at once, also setting size of the sum tree."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.0).any():
      raise ValueError('Values must be finite positive numbers.')
    self._initialize(len(values), values)

  def query(self, targets: Sequence[float]) -> Sequence[int]:
    """Finds smallest indices where `target <` cumulative value sum up to index.

    Args:
      targets: The target sums.

    Returns:
      For each target, the smallest index such that target is strictly less than
      the cumulative sum of values up to and including that index.

    Raises:
      ValueError: if `target >` sum of all values or `target < 0` for any
        of the given targets.
    """
    return [self._query_single(t) for t in targets]

  def root(self) -> float:
    """Returns sum of values."""
    return self._storage[1] if self.size > 0 else np.nan

  @property
  def values(self) -> np.ndarray:
    """View of array containing all (leaf) values in the sum tree."""
    return self._storage[self._first_leaf : self._first_leaf + self.size]

  @property
  def size(self) -> int:
    """Number of (leaf) values in the sum tree."""
    return self._size

  @property
  def capacity(self) -> int:
    """Current sum tree capacity (exceeding it will trigger resizing)."""
    return self._first_leaf

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves sum tree state as a dictionary (e.g. for serialization)."""
    return {
        'size': self._size,
        'storage': self._storage,
        'first_leaf': self._first_leaf,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets sum tree state from a (potentially de-serialized) dictionary."""
    self._size = state['size']
    self._storage = state['storage']
    self._first_leaf = state['first_leaf']

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if len(self._storage) != 2 * self._first_leaf:
      return False, 'first_leaf should be half the size of storage.'
    if not 0 <= self.size <= self.capacity:
      return False, 'Require 0 <= self.size <= self.capacity.'
    if len(self.values) != self.size:
      return False, 'Number of values should be equal to the size.'
    storage = self._storage
    for i in range(1, self._first_leaf):
      if storage[i] != storage[2 * i] + storage[2 * i + 1]:
        return False, 'Non-leaf node %d should be sum of child nodes.' % i
    return True, ''

  def _initialize(self, size: int, values: Optional[Sequence[float]]) -> None:
    """Resizes storage and sets new values if supplied."""
    assert size >= 0
    assert values is None or len(values) == size

    if size < self.size:  # Keep storage and values, zero out extra values.
      if values is None:
        new_values = self.values[:size]  # Truncate existing values.
      else:
        new_values = values
      self._size = size
      self._set_values(new_values)
      # self._first_leaf remains the same.
    elif size <= self.capacity:  # Reuse same storage, but size increases.
      self._size = size
      if values is not None:
        self._set_values(values)
      # self._first_leaf remains the same.
      # New activated leaf nodes are already zero and sum nodes already correct.
    else:  # Allocate new storage.
      new_capacity = 1
      while new_capacity < size:
        new_capacity *= 2
      new_storage = np.empty((2 * new_capacity,), dtype=np.float64)
      if values is None:
        new_values = self.values
      else:
        new_values = values
      self._storage = new_storage
      self._first_leaf = new_capacity
      self._size = size
      self._set_values(new_values)

  def _set_values(self, values: Sequence[float]) -> None:
    """Sets values assuming storage has enough capacity and update sums."""
    # Note every part of the storage is set here.
    assert len(values) <= self.capacity
    storage = self._storage
    storage[self._first_leaf : self._first_leaf + len(values)] = values
    storage[self._first_leaf + len(values) :] = 0
    for i in range(self._first_leaf - 1, 0, -1):
      storage[i] = storage[2 * i] + storage[2 * i + 1]
    storage[0] = 0.0  # Unused.

  def _query_single(self, target: float) -> int:
    """Queries a single target, see query for more detailed documentation."""
    if not 0.0 <= target < self.root():
      raise ValueError('Require 0 <= target < total sum.')

    storage = self._storage
    idx = 1  # Root node.
    while idx < self._first_leaf:
      # At this point we always have target < storage[idx].
      assert target < storage[idx]
      left_idx = 2 * idx
      right_idx = left_idx + 1
      left_sum = storage[left_idx]
      if target < left_sum:
        idx = left_idx
      else:
        idx = right_idx
        target -= left_sum

    assert idx < 2 * self.capacity
    return idx - self._first_leaf


class PrioritizedDistribution:
  """Distribution for weighted sampling of user-defined integer IDs."""

  def __init__(
      self,
      priority_exponent: float,
      uniform_sample_probability: float,
      random_state: np.random.RandomState,
      min_capacity: int = 0,
      max_capacity: Optional[int] = None,
  ):
    if priority_exponent < 0.0:
      raise ValueError('Require priority_exponent >= 0.')
    self._priority_exponent = priority_exponent
    if not 0.0 <= uniform_sample_probability <= 1.0:
      raise ValueError('Require 0 <= uniform_sample_probability <= 1.')
    if max_capacity is not None and max_capacity < min_capacity:
      raise ValueError('Require max_capacity >= min_capacity.')
    if min_capacity < 0:
      raise ValueError('Require min_capacity >= 0.')
    self._uniform_sample_probability = uniform_sample_probability
    self._max_capacity = max_capacity
    self._sum_tree = SumTree()
    self._sum_tree.resize(min_capacity)
    self._random_state = random_state
    self._id_to_index = {}  # User ID -> sum tree index.
    self._index_to_id = {}  # Sum tree index -> user ID.
    # Unused sum tree indices that can be allocated to new user IDs.
    self._inactive_indices = list(range(min_capacity))
    # Currently used sum tree indices, needed for uniform sampling.
    self._active_indices = []
    # Maps an active index to its location in active_indices_, for removal.
    self._active_indices_location = {}

  def ensure_capacity(self, capacity: int) -> None:
    """Ensures sufficient capacity, a no-op if capacity is already enough."""
    if self._max_capacity is not None and capacity > self._max_capacity:
      raise ValueError(
          'capacity %d cannot exceed max_capacity %d'
          % (capacity, self._max_capacity)
      )
    if capacity <= self._sum_tree.size:
      return  # There is already sufficient capacity.
    self._inactive_indices.extend(range(self._sum_tree.size, capacity))
    self._sum_tree.resize(capacity)

  def add_priorities(
      self, ids: Sequence[int], priorities: Sequence[float]
  ) -> None:
    """Add priorities for new IDs."""
    for i in ids:
      if i in self._id_to_index:
        raise IndexError('ID %d already exists.' % i)

    new_size = self.size + len(ids)
    if self._max_capacity is not None and new_size > self._max_capacity:
      raise ValueError('Cannot add IDs as max capacity would be exceeded.')

    # Expand to accommodate new IDs if needed.
    if new_size > self.capacity:
      candidate_capacity = max(new_size, 2 * self.capacity)
      if self._max_capacity is None:
        new_capacity = candidate_capacity
      else:
        new_capacity = min(self._max_capacity, candidate_capacity)
      self.ensure_capacity(new_capacity)

    # Assign unused indices to IDs.
    indices = []
    for i in ids:
      idx = self._inactive_indices.pop()
      self._active_indices_location[idx] = len(self._active_indices)
      self._active_indices.append(idx)
      self._id_to_index[i] = idx
      self._index_to_id[idx] = i
      indices.append(idx)

    # Set priorities on sum tree.
    self._sum_tree.set(indices, _power(priorities, self._priority_exponent))

  def remove_priorities(self, ids: Sequence[int]) -> None:
    """Remove priorities associated with given IDs."""
    indices = []
    for i in ids:
      try:
        idx = self._id_to_index[i]
      except IndexError as err:
        raise IndexError('Cannot remove ID %d, it does not exist.' % i) from err
      indices.append(idx)

    for i, idx in zip(ids, indices):
      del self._id_to_index[i]
      del self._index_to_id[idx]
      # Swap index to be removed with index at the end.
      j = self._active_indices_location[idx]
      self._active_indices[j], self._active_indices[-1] = (
          self._active_indices[-1],
          self._active_indices[j],
      )
      # Update location for the swapped index.
      self._active_indices_location[self._active_indices[j]] = j
      # Remove index from data structures.
      self._active_indices_location.pop(self._active_indices.pop())

    self._inactive_indices.extend(indices)
    self._sum_tree.set(indices, np.zeros((len(indices),), dtype=np.float64))

  def update_priorities(
      self, ids: Sequence[int], priorities: Sequence[float]
  ) -> None:
    """Updates priorities for existing IDs."""
    indices = []
    for i in ids:
      if i not in self._id_to_index:
        raise IndexError('ID %d does not exist.' % i)
      indices.append(self._id_to_index[i])
    self._sum_tree.set(indices, _power(priorities, self._priority_exponent))

  def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns sample of IDs with corresponding probabilities."""
    if self.size == 0:
      raise RuntimeError('No IDs to sample.')
    uniform_indices = [
        self._active_indices[j]
        for j in self._random_state.randint(self.size, size=size)
    ] # randomly sample `size` number of indices

    if self._sum_tree.root() == 0.0: # the sum of values in the tree is 0
      prioritized_indices = uniform_indices # therefore we use the uniformly sampled indices
    else: # the sum of values in the tree is != 0
      targets = self._random_state.uniform(size=size) * self._sum_tree.root() # multiply uniform numbers (0-1) by the sum of the tree
      prioritized_indices = np.asarray(self._sum_tree.query(targets)) # query the numbers in the tree and get their indices

    usp = self._uniform_sample_probability # the uniform sample probability
    indices = np.where(
        self._random_state.uniform(size=size) < usp, 
        uniform_indices,
        prioritized_indices,
    ) # replace indices randomly according to `usp` with the uniformly sampled index and otherwise the prioritized sampled index 

    uniform_prob = np.asarray(1.0 / self.size)  # np.asarray is for pytype.
    priorities = self._sum_tree.get(indices) # get the priorities at the indices

    if self._sum_tree.root() == 0.0:
      prioritized_probs = np.full_like(priorities, fill_value=uniform_prob)
    else:
      prioritized_probs = priorities / self._sum_tree.root()

    sample_probs = (1.0 - usp) * prioritized_probs + usp * uniform_prob
    ids = np.fromiter(
        (self._index_to_id[idx] for idx in indices),
        dtype=np.int64,
        count=len(indices),
    )
    return ids, sample_probs

  def get_exponentiated_priorities(self, ids: Sequence[int]) -> Sequence[float]:
    """Returns priority ** priority_exponent for the given indices."""
    indices = np.fromiter(
        (self._id_to_index[i] for i in ids), dtype=np.int64, count=len(ids)
    )
    return self._sum_tree.get(indices)

  def ids(self) -> Iterable[int]:
    """Returns an iterable of all current IDs."""
    return self._id_to_index.keys()

  @property
  def capacity(self) -> int:
    """Number of IDs that can be stored until memory needs to be allocated."""
    return self._sum_tree.size

  @property
  def size(self) -> int:
    """Number of IDs currently tracked."""
    return len(self._id_to_index)

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
        'sum_tree': self._sum_tree.get_state(),
        'id_to_index': self._id_to_index,
        'index_to_id': self._index_to_id,
        'inactive_indices': self._inactive_indices,
        'active_indices': self._active_indices,
        'active_indices_location': self._active_indices_location,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._sum_tree.set_state(state['sum_tree'])
    self._id_to_index = state['id_to_index']
    self._index_to_id = state['index_to_id']
    self._inactive_indices = state['inactive_indices']
    self._active_indices = state['active_indices']
    self._active_indices_location = state['active_indices_location']

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if len(self._id_to_index) != len(self._index_to_id):
      return False, 'ID to index maps are not the same size.'
    for i in self._id_to_index:
      if self._index_to_id[self._id_to_index[i]] != i:
        return False, 'ID %d should map to itself.' % i
    # Indices map to themselves because of previous check and uniqueness.
    if len(set(self._inactive_indices)) != len(self._inactive_indices):
      return False, 'Inactive indices should be unique.'
    if len(set(self._active_indices)) != len(self._active_indices):
      return False, 'Active indices should be unique.'
    if set(self._active_indices) != set(self._index_to_id.keys()):
      return False, 'Active indices should match index to ID mapping keys.'
    all_indices = self._inactive_indices + self._active_indices
    if sorted(all_indices) != list(range(self._sum_tree.size)):
      return False, 'Inactive and active indices should partition all indices.'
    if len(self._active_indices) != len(self._active_indices_location):
      return False, 'Active indices and their location should be the same size.'
    for j, i in enumerate(self._active_indices):
      if j != self._active_indices_location[i]:
        return False, (
            'Active index location %d not correct for index %d.' % (j, i)
        )

    return self._sum_tree.check_valid()


class MGSCDistribution:
  """Distribution for weighted sampling of user-defined integer IDs."""

  def __init__(
      self,
      rng_key: jnp.ndarray,
      min_capacity: int = 0,
      max_capacity: Optional[int] = None,
  ):
    if max_capacity is not None and max_capacity < min_capacity:
      raise ValueError('Require max_capacity >= min_capacity.')
    if min_capacity < 0:
      raise ValueError('Require min_capacity >= 0.')
    self._max_capacity = max_capacity
    self._buffer = CircularLogitBuffer(capacity=max_capacity)
    self._rng_key = rng_key

  def ensure_capacity(self, capacity: int) -> None:
    pass

  def add_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
    """Add priorities for new IDs."""
    for i in indices:
      if 0 <= i and i < len(self._logits):
        raise IndexError('ID %d already exists.' % i)

    new_size = self.size + len(indices)
    if self._max_capacity is not None and new_size > self._max_capacity:
      raise ValueError('Cannot add IDs as max capacity would be exceeded.')

    # Add new priorities
    self._logits = np.pad(self._logits, (0, len(indices)))
    self._logits[indices] = priorities
    self._buffer.add(prior)

  @property
  def probabilities(self) -> np.ndarray:
    return probabilities_from_logits(self._logits)

  def remove_priorities(self, indices: Sequence[int]) -> None:
    """Remove priorities associated with given IDs."""
    self._logits = np.delete(self._logits, indices)

  def update_priorities(self, indices: Sequence[int], priorities: np.ndarray) -> None:
    """Updates priorities for existing IDs in order of what's specified."""
    self._logits[np.asarray(indices)] = priorities

  def sample(self, size: int) -> np.ndarray:
    """Returns sample of indices."""
    if self.size == 0:
      raise RuntimeError('No items to sample.')
    probs = self.probabilities
    # ids_in_order = [self._index_to_id[idx] for idx in range(len(self._logits))]
    self._rng_key, key = jax.random.split(self._rng_key)
    indices = jax.random.choice(key, self.size, shape=(size,), p=probs)
    # or np.arange to sample idx then convert to ids
    # maybe jax.random.choice is faster as well?
    # or is the python native random choice faster?
    return indices

  def uniform_sample(self, size: int, replace:bool=True) -> np.ndarray:
    """Returns uniform sample of indices."""
    if self.size == 0:
      raise RuntimeError('No items to sample.')
    self._rng_key, key = jax.random.split(self._rng_key)
    indices = jax.random.choice(key, self.size, shape=(size,), replace=replace)
    return indices

  def get_priorities(self, indices: Sequence[int]) -> np.ndarray:
    """Returns a numpy array of the requested logits for the ids specified in order of the ids."""
    return self._logits[np.asarray(indices)]

  @property
  def capacity(self) -> int:
    """Number of items that can be stored until memory needs to be allocated."""
    return len(self._logits)

  @property
  def size(self) -> int:
    """Number of items currently tracked."""
    return len(self._logits)

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
        'rng_key': self._rng_key,
        'logits': self._logits,
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._logits = state['logits']

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    return True


class MGSCFiFoTransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with LIFO storage for flat named tuples."""

  def __init__(
      self,
      capacity: int,
      structure: ReplayStructure,
      random_state: np.random.Generator,
      # rng_key: jnp.ndarray,
      encoder: Optional[Callable[[ReplayStructure], Any]] = None,
      decoder: Optional[Callable[[Any], ReplayStructure]] = None,
  ):
    self._capacity = capacity
    self._structure = structure
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._distribution = CircularLogitBuffer(capacity=capacity, random_state=random_state)
    # self._distribution = CircularLogitBuffer(capacity=capacity, rng_key=rng_key)
    self._storage = CircularBuffer(capacity=capacity)
    self._t = 0  # counter of elements added

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    if self._storage.is_full():
      self._distribution.popleft()
      self._storage.popleft()

    self._distribution.add()
    self._storage.add(self._encoder(item))
    self._t += 1

  def get(self, indices: Sequence[int]) -> Iterable[ReplayStructure]:
    """Retrieves items by IDs."""
    for i in indices:
      yield self._decoder(self._storage[i])

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay according to the distribution."""
    indices = self._distribution.sample(size)
    return self.stack_transitions(indices)  # pytype: disable=not-callable

  def stack_transitions(self, indices: Sequence[int]) -> ReplayStructure:
    samples = self.get(indices)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  # def transitions_and_logits(self) -> Tuple[ReplayStructure, np.ndarray]:
  #   """Get the transitions and their associated logits."""
  #   indices = np.arange(self.size)
  #   return self.stack_transitions(indices), self._distribution._logits

  def batch_of_ids_transitions_and_logits(self, size: int) -> Tuple[Sequence[int], ReplayStructure, Sequence[float]]:
    """Return a batch of transitions sampled uniformly (not according to the distribution) without replacement."""
    indices = self._distribution.sample_uniform(size, replace=False)
    transitions = self.stack_transitions(indices)
    logits = self._distribution[np.asarray(indices)]
    return indices, transitions, logits
  
  def update_priorities(self, indices:Sequence[int], priorities:Sequence[float]):
    """Update the priorities of the associated IDs."""
    self._distribution[np.asarray(indices)] = priorities

  @property
  def size(self) -> int:
    """Number of items currently contained in the replay."""
    return len(self._storage)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        # Serialize OrderedDict as a simpler, more common data structure.
        'storage': self._storage.get_state(),
        't': self._t,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage.set_state(state['storage'])
    self._t = state['t']
    self._distribution.set_state(state['distribution'])

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if self._t < len(self._storage):
      return False, 't should be >= storage size.'
    # if set(self._storage.keys()) != set(self._distribution.ids()):
      # return False, 'IDs in storage and distribution do not match.'
    return self._distribution.check_valid()


class PrioritizedTransitionReplay(Generic[ReplayStructure]):
  """Prioritized replay, with LIFO storage for flat named tuples.

  This is the proportional variant as described in
  http://arxiv.org/abs/1511.05952.
  """

  def __init__(
      self,
      capacity: int,
      structure: ReplayStructure,
      priority_exponent: float,
      importance_sampling_exponent: Callable[[int], float],
      uniform_sample_probability: float,
      normalize_weights: bool,
      random_state: np.random.RandomState,
      encoder: Optional[Callable[[ReplayStructure], Any]] = None,
      decoder: Optional[Callable[[Any], ReplayStructure]] = None,
  ):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)
    self._distribution = PrioritizedDistribution(
        min_capacity=capacity,
        max_capacity=capacity,
        priority_exponent=priority_exponent,
        uniform_sample_probability=uniform_sample_probability,
        random_state=random_state,
    )
    self._importance_sampling_exponent = importance_sampling_exponent
    self._normalize_weights = normalize_weights
    self._storage = collections.OrderedDict()  # ID -> item.
    self._t = 0  # Used to allocate IDs.

  def add(self, item: ReplayStructure, priority: float) -> None:
    """Adds a single item with a given priority to the replay buffer."""
    if self.size == self._capacity:
      oldest_id, _ = self._storage.popitem(last=False)
      self._distribution.remove_priorities([oldest_id])

    item_id = self._t
    self._distribution.add_priorities([item_id], [priority])
    self._storage[item_id] = self._encoder(item)
    self._t += 1

  def get(self, ids: Sequence[int]) -> Iterable[ReplayStructure]:
    """Retrieves items by IDs."""
    for i in ids:
      yield self._decoder(self._storage[i])

  def sample(
      self,
      size: int,
  ) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
    """Samples a batch of transitions."""
    ids, probabilities = self._distribution.sample(size)
    weights = importance_sampling_weights(
        probabilities,
        uniform_probability=1.0 / self.size,
        exponent=self.importance_sampling_exponent,
        normalize=self._normalize_weights,
    )
    samples = self.get(ids)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    # pytype: disable=not-callable
    return type(self._structure)(*stacked), ids, weights
    # pytype: enable=not-callable

  def update_priorities(
      self, ids: Sequence[int], priorities: Sequence[float]
  ) -> None:
    """Updates IDs with given priorities."""
    priorities = np.asarray(priorities)
    self._distribution.update_priorities(ids, priorities)

  @property
  def size(self) -> int:
    """Number of elements currently contained in replay."""
    return len(self._storage)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (maximum number of items that can be stored)."""
    return self._capacity

  @property
  def importance_sampling_exponent(self):
    """Importance sampling exponent at current step."""
    return self._importance_sampling_exponent(self._t)

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        # Serialize OrderedDict as a simpler, more common data structure.
        'storage': list(self._storage.items()),
        't': self._t,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = collections.OrderedDict(state['storage'])
    self._t = state['t']
    self._distribution.set_state(state['distribution'])

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if self._t < len(self._storage):
      return False, 't should be >= storage size.'
    if set(self._storage.keys()) != set(self._distribution.ids()):
      return False, 'IDs in storage and distribution do not match.'
    return self._distribution.check_valid()


class TransitionAccumulator:
  """Accumulates timesteps to form transitions."""

  def __init__(self):
    self.reset()

  def step(
      self, timestep_t: dm_env.TimeStep, a_t: parts.Action
  ) -> Iterable[Transition]:
    """Accumulates timestep and resulting action, maybe yield a transition."""
    if timestep_t.first():
      self.reset()

    if self._timestep_tm1 is None:
      if not timestep_t.first():
        raise ValueError('Expected FIRST timestep, got %s.' % str(timestep_t))
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      return  # Empty iterable.
    else:
      transition = Transition(
          s_tm1=self._timestep_tm1.observation,
          a_tm1=self._a_tm1,
          r_t=timestep_t.reward,
          discount_t=timestep_t.discount,
          s_t=timestep_t.observation,
      )
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      yield transition

  def reset(self) -> None:
    """Resets the accumulator. Following timestep is expected to be `FIRST`."""
    self._timestep_tm1 = None
    self._a_tm1 = None


def _build_n_step_transition(transitions):
  """Builds a single n-step transition from n 1-step transitions."""
  r_t = 0.0
  discount_t = 1.0
  for transition in transitions:
    r_t += discount_t * transition.r_t
    discount_t *= transition.discount_t

  # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
  return Transition(
      s_tm1=transitions[0].s_tm1,
      a_tm1=transitions[0].a_tm1,
      r_t=r_t,
      discount_t=discount_t,
      s_t=transitions[-1].s_t,
  )


class NStepTransitionAccumulator:
  """Accumulates timesteps to form n-step transitions.

  Let `t` be the index of a timestep within an episode and `T` be the index of
  the final timestep within an episode. Then given the step type of the timestep
  passed into `step()` the accumulator will:
  *   `FIRST`: yield nothing.
  *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
      `s_{t - n} -> s_t`.
  *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
      away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
      These are `min(n, T)`-step, ..., `1`-step transitions.
  """

  def __init__(self, n):
    self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
    self.reset()

  def step(
      self, timestep_t: dm_env.TimeStep, a_t: parts.Action
  ) -> Iterable[Transition]:
    """Accumulates timestep and resulting action, yields transitions."""
    if timestep_t.first():
      self.reset()

    # There are no transitions on the first timestep.
    if self._timestep_tm1 is None:
      assert self._a_tm1 is None
      if not timestep_t.first():
        raise ValueError('Expected FIRST timestep, got %s.' % str(timestep_t))
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      return  # Empty iterable.

    self._transitions.append(
        Transition(
            s_tm1=self._timestep_tm1.observation,
            a_tm1=self._a_tm1,
            r_t=timestep_t.reward,
            discount_t=timestep_t.discount,
            s_t=timestep_t.observation,
        )
    )

    self._timestep_tm1 = timestep_t
    self._a_tm1 = a_t

    if timestep_t.last():
      # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
      while self._transitions:
        yield _build_n_step_transition(self._transitions)
        self._transitions.popleft()
    else:
      # Wait for n transitions before yielding anything.
      if len(self._transitions) < self._transitions.maxlen:
        return  # Empty iterable.

      assert len(self._transitions) == self._transitions.maxlen

      # This is the typical case, yield a single n-step transition.
      yield _build_n_step_transition(self._transitions)

  def reset(self) -> None:
    """Resets the accumulator. Following timestep is expected to be FIRST."""
    self._transitions.clear()
    self._timestep_tm1 = None
    self._a_tm1 = None


def compress_array(array: np.ndarray) -> CompressedArray:
  """Compresses a numpy array with snappy."""
  return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed: CompressedArray) -> np.ndarray:
  """Uncompresses a numpy array with snappy given its shape and dtype."""
  compressed_array, shape, dtype = compressed
  byte_string = snappy.uncompress(compressed_array)
  return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
