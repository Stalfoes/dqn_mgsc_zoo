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

"""DQN agent class."""

# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping

from absl import logging
import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

import time

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay_circular_nonsense as replay_lib


def safe_div(a, b):
  if b == 0:
    return 0
  return a, b


# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)


class MGSCDqn(parts.Agent):
  """Deep Q-Network agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.MGSCFiFoTransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      grad_error_bound: float,
      rng_key: parts.PRNGKey,
      meta_optimizer: optax.GradientTransformation,
      meta_batch_size: int,
      nonsense_transition_ratio: int,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period
    self._meta_batch_size = meta_batch_size
    self._nonsense_transition_ratio = nonsense_transition_ratio
    self._logits_vs_time = {
      'nonsense': {
        'max': -float('inf'),
        'min': float('inf'),
        'entry': {
          'sum': 0,
          'num': 0
        }, '1st_quarter': {
          'sum': 0,
          'num': 0
        }, '2nd_quarter': {
          'sum': 0,
          'num': 0
        }, '3rd_quarter': {
          'sum': 0,
          'num': 0
        }, 'exit': {
          'sum': 0,
          'num': 0
        }
      },
      'real': {
        'max': -float('inf'),
        'min': float('inf'),
        'entry': {
          'sum': 0,
          'num': 0
        }, '1st_quarter': {
          'sum': 0,
          'num': 0
        }, '2nd_quarter': {
          'sum': 0,
          'num': 0
        }, '3rd_quarter': {
          'sum': 0,
          'num': 0
        }, 'exit': {
          'sum': 0,
          'num': 0
        }
      }
    }

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(
        network_rng_key, sample_network_input[None, ...]
    )
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)
    self._meta_opt_state = meta_optimizer.init(jnp.zeros((self._meta_batch_size,), dtype=jnp.float32))

    # To help with the meta-gradient-learn step
    self._last_transitions = []

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}

    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def norm_of_pytree(params, target_params):
      squared_difference = jax.tree_util.tree_map(lambda p, t: jnp.sum((p - t) ** 2), params, target_params)
      reduced = jax.tree_util.tree_reduce(lambda a, b: a + b, squared_difference) # just calls reduce on the leaves
      return reduced

    def loss_fn_no_mean(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      _, online_key, target_key = jax.random.split(rng_key, 3)
      q_tm1 = network.apply(
          online_params, online_key, transitions.s_tm1
      ).q_values
      q_target_t = network.apply(
          target_params, target_key, transitions.s_t
      ).q_values
      td_errors = _batch_q_learning(
          q_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          q_target_t,
      ) # an array of floats of length `len(transitions.s_tm1)`
      td_errors = rlax.clip_gradient(
          td_errors, -grad_error_bound, grad_error_bound
      ) # an array of floats of length `len(transitions.s_tm1)`
      losses = rlax.l2_loss(td_errors) # this is just squaring and multiplying by 2. This is is an array of floats of the length of transitions
      return losses

    def loss_fn(online_params, target_params, transitions, rng_key):
      losses = loss_fn_no_mean(online_params, target_params, transitions, rng_key)
      loss = jnp.mean(losses) # Take the mean of the array of floats
      return loss

    def batch_single_transition(transition): # Is this an expensive operation?
      return replay_lib.Transition(
        s_tm1=jnp.asarray(transition.s_tm1)[None, ...],
        a_tm1=jnp.asarray(transition.a_tm1)[None, ...],
        r_t=jnp.asarray(transition.r_t)[None, ...],
        discount_t=jnp.asarray(transition.discount_t)[None, ...],
        s_t=jnp.asarray(transition.s_t)[None, ...]
      )

    def weighted_grads(online_params, target_params, transition, probability, rng_key):
      """The function to VMAP over which computes the gradient of each transition and multiplies by its probabilty.
      """
      transition = batch_single_transition(transition)
      grad = jax.grad(loss_fn)(online_params, target_params, transition, rng_key)
      weighted_grad = jax.tree_util.tree_map(lambda leaf: probability * leaf, grad)
      return weighted_grad

    def meta_loss_fn(logits, transitions, online_params, target_params, online_transition, opt_state, rng_key):
      rng_key, exp_loss_key, target_loss_key = jax.random.split(rng_key, 3)
      
      online_transition = batch_single_transition(online_transition)
      
      probabilities = replay_lib.JNPprobabilities_from_logits(logits)

      weighted_gradients = jax.vmap(weighted_grads, (None,None,0,0,None))(online_params, target_params, transitions, probabilities, exp_loss_key)
      summed_weighted_grads = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), weighted_gradients)
      
      updates, new_opt_state = optimizer.update(summed_weighted_grads, opt_state)
      expected_online_params = optax.apply_updates(online_params, updates)

      gradients_of_expected_params = jax.grad(loss_fn)(
          expected_online_params, online_params, online_transition, target_loss_key
      )

      # It's possible that since RMS is prop is too high-order of an optimizer, Adam going over that leads to instability
      target_updates, new_opt_state = optimizer.update(gradients_of_expected_params, new_opt_state)
      target_online_params = optax.apply_updates(expected_online_params, target_updates)

      loss = norm_of_pytree(expected_online_params, target_online_params)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)( # calling the jax.grad(loss_fn)(...) is expensive
          online_params, target_params, transitions, update_key
      )
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates) # is a for loop, but only over leaves of the parameters which is relatively short
      return rng_key, new_opt_state, new_online_params

    def meta_update(rng_key, opt_state, meta_opt_state, online_params, target_params, transitions, logits, online_transition):
      gradients_of_meta_params = jax.grad(meta_loss_fn)(
          logits, transitions, online_params, target_params, online_transition, opt_state, rng_key
      )
      meta_params_updates, new_meta_opt_state = meta_optimizer.update(gradients_of_meta_params, meta_opt_state)
      new_meta_params = optax.apply_updates(logits, meta_params_updates)
      return rng_key, new_meta_opt_state, new_meta_params

    self._update = jax.jit(update)
    self._meta_update = jax.jit(meta_update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
          seed=policy_key
      )
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      if self._action is None:
        raise RuntimeError('Cannot repeat if action has never been selected.')
      action = self._action
    else:
      action = self._action = self._act(timestep)
      transitions = [t for t in self._transition_accumulator.step(timestep, action)]
      self._last_transitions = self._last_transitions + transitions # this might just be a call-by-reference which makes the replay adding become NULL

    # Meta-prioritization learn step
    if (self._frame_t % self._learn_period == 0) and (self._replay.size >= max(self._meta_batch_size, self._min_replay_capacity)):
      if len(self._last_transitions) != 0:
        trans = self._last_transitions[-1] # TODO -- should we just be taking the most recent transition? maybe we can perform multiple updates?
        self._meta_prioritization_learn(trans)
        self._last_transitions.clear()
      else:
        print(f"Skipping a META LEARNING train step because the _last_transitions buffer was empty on frame {self._frame_t}...")
    if not (timestep is None):
      # Add states to replay buffer
      for transition in transitions:
        if self._replay._t % self._nonsense_transition_ratio == 0 and self._replay._t > 0:
          # Jumble the transition up to make it more likely to be nonsensical
          jumbled_transition = transition._replace(
            s_tm1 = self._replay._storage[0].s_tm1,
            a_tm1 = self._replay._storage[self._replay._storage.size - 1].a_tm1,
            r_t = self._replay._storage[self._replay._storage.size // 4].r_t,
            discount_t = self._replay._storage[self._replay._storage.size // 3].discount_t,
            s_t = self._replay._storage[self._replay._storage.size // 2].s_t,
          )
          self._replay.add(jumbled_transition, nonsense=True)
        else:
          self._replay.add(transition)
      # Update the nonsense statistics
      for idx in range(0, self._replay._storage.size):
        rel_idx = self._replay._storage._rel_idx_to_abs(idx)
        nonsense_real_key = 'nonsense' if self._replay._storage._is_nonsense[rel_idx] else 'real'
        logit_value = self._replay._distribution[idx]
        self._logits_vs_time[nonsense_real_key]['max'] = max(logit_value, self._logits_vs_time[nonsense_real_key]['max'])
        self._logits_vs_time[nonsense_real_key]['min'] = min(logit_value, self._logits_vs_time[nonsense_real_key]['min'])
        if idx == 0:
          self._logits_vs_time[nonsense_real_key]['exit']['sum'] += logit_value
          self._logits_vs_time[nonsense_real_key]['exit']['num'] += 1
        elif idx == (self._replay._storage.size // 4):
          self._logits_vs_time[nonsense_real_key]['3rd_quarter']['sum'] += logit_value
          self._logits_vs_time[nonsense_real_key]['3rd_quarter']['num'] += 1
        elif idx == (self._replay._storage.size // 2):
          self._logits_vs_time[nonsense_real_key]['2nd_quarter']['sum'] += logit_value
          self._logits_vs_time[nonsense_real_key]['2nd_quarter']['num'] += 1
        elif idx == (3 * (self._replay._storage.size // 4)):
          self._logits_vs_time[nonsense_real_key]['1st_quarter']['sum'] += logit_value
          self._logits_vs_time[nonsense_real_key]['1st_quarter']['num'] += 1
        elif idx == self._replay._storage.size - 1:
          self._logits_vs_time[nonsense_real_key]['entry']['sum'] += logit_value
          self._logits_vs_time[nonsense_real_key]['entry']['num'] += 1

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def logit_value_stats(self) -> list[tuple[str, Any, str]]:
    return [
      ('nonsense_max', self._logits_vs_time['nonsense']['max'], '%.3f'),
      ('nonsense_min', self._logits_vs_time['nonsense']['min'], '%.3f'),
      ('real_max', self._logits_vs_time['real']['max'], '%.3f'),
      ('real_min', self._logits_vs_time['real']['min'], '%.3f'),
      ('nonsense_entry_avg', safe_div(self._logits_vs_time['nonsense']['entry']['sum'], self._logits_vs_time['nonsense']['entry']['num']), '%.3f'),
      ('nonsense_1st_quarter_avg', safe_div(self._logits_vs_time['nonsense']['1st_quarter']['sum'], self._logits_vs_time['nonsense']['1st_quarter']['num']), '%.3f'),
      ('nonsense_2nd_quarter_avg', safe_div(self._logits_vs_time['nonsense']['2nd_quarter']['sum'], self._logits_vs_time['nonsense']['2nd_quarter']['num']), '%.3f'),
      ('nonsense_3rd_quarter_avg', safe_div(self._logits_vs_time['nonsense']['3rd_quarter']['sum'], self._logits_vs_time['nonsense']['3rd_quarter']['num']), '%.3f'),
      ('nonsense_exit_avg', safe_div(self._logits_vs_time['nonsense']['exit']['sum'], self._logits_vs_time['nonsense']['exit']['num']), '%.3f'),
      ('real_entry_avg', safe_div(self._logits_vs_time['real']['entry']['sum'], self._logits_vs_time['real']['entry']['num']), '%.3f'),
      ('real_1st_quarter_avg', safe_div(self._logits_vs_time['real']['1st_quarter']['sum'], self._logits_vs_time['real']['1st_quarter']['num']), '%.3f'),
      ('real_2nd_quarter_avg', safe_div(self._logits_vs_time['real']['2nd_quarter']['sum'], self._logits_vs_time['real']['2nd_quarter']['num']), '%.3f'),
      ('real_3rd_quarter_avg', safe_div(self._logits_vs_time['real']['3rd_quarter']['sum'], self._logits_vs_time['real']['3rd_quarter']['num']), '%.3f'),
      ('real_exit_avg', safe_div(self._logits_vs_time['real']['exit']['sum'], self._logits_vs_time['real']['exit']['num']), '%.3f'),
    ]

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None
    self._last_transitions = []

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self._select_action(
        self._rng_key, self._online_params, s_t, self.exploration_epsilon
    )
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _meta_prioritization_learn(self, online_transition) -> None:
    """Performs an expected update on the online parameters and updates the meta-parameters with the target."""
    logging.log_first_n(logging.INFO, 'Begin meta-prioritization learning', 1)
    # Draw a meta batch
    # s = time.time()
    indices, transitions, logits = self._replay.batch_of_ids_transitions_and_logits(self._meta_batch_size)
    # e = time.time()
    # self.times['replay-sample-meta-batch']['total_time'] += e - s
    # self.times['replay-sample-meta-batch']['num_calls'] += 1

    # Perform an update
    # s = time.time()
    self._rng_key, self._meta_opt_state, new_meta_params = self._meta_update(
        self._rng_key,
        self._opt_state,
        self._meta_opt_state,
        self._online_params,
        self._target_params,
        transitions,
        logits,
        online_transition
    )
    # e = time.time()
    # self.times['meta-update']['total_time'] += e - s
    # self.times['meta-update']['num_calls'] += 1

    # if self._debugging_num_meta_update_calls % 75 == 0:
      # print(f"Logits on update={self._debugging_num_meta_update_calls}: {new_meta_params=}, {len(new_meta_params)=}, {self._replay._distribution._logits.min()=}, {self._replay._distribution._logits.mean()=}, {self._replay._distribution._logits.max()=}")
    # if jnp.isnan(new_meta_params).any():
      # raise ValueError(f"New batched logits contain a NaN. Occurs on update_number={self._debugging_num_meta_update_calls} {new_meta_params=}, {len(new_meta_params)=}, {np.isnan(new_meta_params).any()=}")
    
    # s = time.time()
    self._replay.update_priorities(indices, new_meta_params) # these are nan's sometimes!
    # e = time.time()
    # self.times['replay-update']['total_time'] += e - s
    # self.times['replay-update']['num_calls'] += 1
    
    # self._debugging_num_meta_update_calls += 1

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    # s = time.time()
    transitions = self._replay.sample(self._batch_size)
    # e = time.time()
    # self.times['replay-sample-batch']['total_time'] += e - s
    # self.times['replay-sample-batch']['num_calls'] += 1
    
    # s = time.time()
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
    )
    # e = time.time()
    # self.times['update']['total_time'] += e - s
    # self.times['update']['num_calls'] += 1

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[str, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values()
    )
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[str, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
        'meta_opt_state': self._meta_opt_state,
        'logits_vs_time': self._logits_vs_time
    }
    return state

  def set_state(self, state: Mapping[str, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
    self._meta_opt_state = state['meta_opt_state']
    self._logits_vs_time = state['logits_vs_time']
