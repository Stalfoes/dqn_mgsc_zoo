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

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

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
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(
        network_rng_key, sample_network_input[None, ...]
    )
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)
    self._meta_opt_state = meta_optimizer.init(jnp.zeros((replay.capacity,), dtype=jnp.float32)) # ERROR: logits here is [] but eventually grows, but the meta_opt_state is not set up for that kind of thing

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
      chex.assert_trees_all_equal_shapes(params, target_params)
      l2_norms = jax.tree_util.tree_map(lambda t, e: jnp.linalg.norm(t - e, ord=None) ** 2, target_params, params)
      l2_norms_list, _ = jax.tree_util.tree_flatten(l2_norms)
      reduced = jnp.sum(jnp.array(l2_norms_list))
      return reduced

    def loss_fn_no_mean(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      chex.assert_trees_all_equal_shapes(online_params, target_params)
      _, online_key, target_key = jax.random.split(rng_key, 3)
      q_tm1 = network.apply( # ValueError: 'sequential/sequential_1/linear/w' with retrieved shape (3136, 512) does not match shape=[448, 512] dtype=dtype('float32')
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
      losses = rlax.l2_loss(td_errors) # # an array of floats of length `len(transitions.s_tm1)`. THIS IS JUST A SQUARE TIMES 1/2
      return losses

    def loss_fn(online_params, target_params, transitions, rng_key):
      losses = loss_fn_no_mean(online_params, target_params, transitions, rng_key)
      loss = jnp.mean(losses) # should be just a single float
      return loss
    
    def exp_loss_fn(online_params, target_params, transitions, probabilities, rng_key):
      losses = loss_fn_no_mean(online_params, target_params, transitions, rng_key)
      return jnp.dot(probabilities, losses)

    def batch_single_transition(transition):
      return type(transition)(*[
          jnp.array(transition[i])[None, ...] for i in range(len(transition))
      ])

    def meta_loss_fn(logits, transitions, online_params, target_params, online_transition, opt_state, rng_key):
      rng_key, exp_loss_key, target_loss_key = jax.random.split(rng_key, 3)
      online_transition = batch_single_transition(online_transition)
      e_to_logits = jnp.power(jnp.e, logits)
      probabilities = e_to_logits / jnp.sum(e_to_logits)
      truncated_probabilities = probabilities[:len(transitions.s_tm1)]

      summed_weighted_grads = jax.grad(exp_loss_fn)(online_params, target_params, transitions, truncated_probabilities, exp_loss_key)
      updates, new_opt_state = optimizer.update(summed_weighted_grads, opt_state) # RMS Prop
      expected_online_params = optax.apply_updates(online_params, updates)

      d_loss_d_expected_params = jax.grad(loss_fn)(
          expected_online_params, online_params, online_transition, target_loss_key
      )
      target_updates, new_opt_state = optimizer.update(d_loss_d_expected_params, new_opt_state) # RMS Prop
      target_online_params = optax.apply_updates(expected_online_params, target_updates)

      loss = norm_of_pytree(expected_online_params, target_online_params)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(
          online_params, target_params, transitions, update_key
      )
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    def meta_update(rng_key, opt_state, meta_opt_state, online_params, target_params, transitions, logits, online_transition):
      zero_padded_logits = jnp.array(logits + [0] * (replay.capacity - len(logits)), dtype=jnp.float32)
      d_loss_d_meta_params = jax.grad(meta_loss_fn)(
          zero_padded_logits, transitions, online_params, target_params, online_transition, opt_state, rng_key
      )
      meta_updates, new_meta_opt_state = meta_optimizer.update(d_loss_d_meta_params, meta_opt_state)
      new_meta_params_zero_padded = optax.apply_updates(zero_padded_logits, meta_updates)
      new_meta_params = list(new_meta_params_zero_padded[:len(logits)])
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
    if (self._frame_t % self._learn_period == 0) and (self._replay.size >= self._min_replay_capacity):
      if len(self._last_transitions) != 0:
        trans = self._last_transitions[-1] # TODO -- should we just be taking the most recent transition? maybe we can perform multiple updates?
        self._meta_prioritization_learn(trans)
        self._last_transitions.clear()
      else:
        print(f"Skipping a META LEARNING train step because the _last_transitions buffer was empty on frame {self._frame_t}...")
    if not (timestep is None):
      # Add states to replay buffer
      for transition in transitions:
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

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
    print("=======================\n", online_transition, "=======================")
    def shape_of(thing):
      if hasattr(thing, 'shape'):
        return (thing.shape, thing.dtype)
      elif hasattr(thing, '__len__'):
        return ((len(thing),), type(thing[0]))
      else:
        return type(thing)
    def pytree_structure(tree):
      leaves, structure = jax.tree_util.tree_flatten(tree)
      leaves = [shape_of(leaf) for leaf in leaves]
      return jax.tree_util.tree_unflatten(structure, leaves)
    print("=======================\n", type(online_transition)(*[shape_of(online_transition[i]) for i in range(len(online_transition))]),"=======================")
    print("=======================\n", pytree_structure(self._opt_state),"=======================")
    print("=======================\n", pytree_structure(self._meta_opt_state),"=======================")
    quit()
    transitions, logits = self._replay.transitions_and_logits()
    self._rng_key, self._meta_opt_state, new_meta_params = self._meta_update(
        self._rng_key,
        self._opt_state,
        self._meta_opt_state,
        self._online_params,
        self._target_params,
        transitions,
        logits,
        online_transition)
    self._replay._distribution._logits = new_meta_params

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
    )

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
