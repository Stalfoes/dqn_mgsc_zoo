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

"""A DQN agent training on Atari.

From the paper "Human Level Control Through Deep Reinforcement Learning"
http://www.nature.com/articles/nature14236.
"""

# pylint: disable=g-bad-import-order

import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import chex
import dm_env
import haiku as hk
import jax
from jax import config
import numpy as np
import optax

from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib


# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping

import distrax
import jax.numpy as jnp
import rlax

# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)

class TestMGSCDqn(parts.Agent):
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
      assert len(transitions.s_tm1) > 0, f'Transitions was of length={len(transitions.s_tm1)} when expected > 0'
      # print(f'\tTransitions was of length={len(transitions.s_tm1)}')
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
      # print(f"{losses.shape=}")
      chex.assert_shape(losses, (len(transitions.s_tm1),)) # chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses) # should be just a single float
      # print(f"{loss.shape=}")
      return loss
    
    def exp_loss_fn(online_params, target_params, transitions, probabilities, rng_key):
      losses = loss_fn_no_mean(online_params, target_params, transitions, rng_key)
      return jnp.dot(probabilities, losses)

    def batch_single_transition(transition):
      return type(transition)(*[
          jnp.array(transition[i])[None, ...] for i in range(len(transition))
      ])

    def expected_sum(online_params, target_params, transition, probability, rng_key):
      """Collect the gradient and multiply by the probability"""
      chex.assert_trees_all_equal_shapes(online_params, target_params)
      batched_transition = batch_single_transition(transition)
      grad = jax.grad(loss_fn)(online_params, target_params, batched_transition, rng_key)
      weighted_grad = jax.tree_util.tree_map(lambda leaf: probability * leaf, grad)
      return weighted_grad

    def meta_loss_fn(logits, transitions, online_params, target_params, online_transition, opt_state, rng_key):
      rng_key, exp_loss_key, target_loss_key = jax.random.split(rng_key, 3)
      # online_transition = type(online_transition)(*[
      #     jnp.array(online_transition[i])[None, ...] for i in range(len(online_transition))
      # ])
      online_transition = batch_single_transition(online_transition)
      # print("batched online transition")
      # Unbatch the transitions so we can compute the gradient for each individual transition
      # unbatched_transitions = [
      #     type(online_transition)(*[
      #         jnp.array(transitions[i][t], dtype=transitions[i].dtype)[None, ...] for i in range(len(transitions))
      #     ])
      #     for t in range(len(transitions.s_tm1))
      # ]
      
      assert len(logits) == replay.capacity, f"Logits were of length={len(logits)} when expected {replay.capacity}"
      # print(f"\tLogits are of length={len(logits)}")
      e_to_logits = jnp.power(jnp.e, logits)
      probabilities = e_to_logits / jnp.sum(e_to_logits)
      assert len(probabilities) == replay.capacity, f"{len(probabilities)} != {replay.capacity}"
      truncated_probabilities = probabilities[:len(transitions.s_tm1)]
      assert len(truncated_probabilities) == len(transitions.s_tm1), f"{len(truncated_probabilities)} != {len(transitions.s_tm1)}"

      # print("calculated probabilities")

      # weighted_grads = jax.vmap(expected_sum, (None,None,0,0,None))(online_params, target_params, transitions, truncated_probabilities, exp_loss_key)
      # weighted_grads_leaves, _ = jax.tree_util.tree_flatten(weighted_grads)
      # assert weighted_grads_leaves[0].shape[0] == len(transitions.s_tm1), f"weighted_grads was length={weighted_grads_leaves[0].shape[0]} but expected {len(transitions.s_tm1)}"
      # # chex.assert_trees_all_equal_shapes(weighted_grads[0], online_params)
      # # summed_weighted_grads = jax.tree_util.tree_map(lambda *v: sum(v), *weighted_grads)
      # summed_weighted_grads = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), weighted_grads)
      summed_weighted_grads = jax.grad(exp_loss_fn)(online_params, target_params, transitions, truncated_probabilities, exp_loss_key)
      chex.assert_trees_all_equal_shapes(summed_weighted_grads, online_params)
      updates, new_opt_state = optimizer.update(summed_weighted_grads, opt_state) # RMS Prop
      chex.assert_trees_all_equal_shapes(updates, online_params)
      expected_online_params = optax.apply_updates(online_params, updates)
      chex.assert_trees_all_equal_shapes(expected_online_params, online_params)

      # print("calculated the expected parameters")

      d_loss_d_expected_params = jax.grad(loss_fn)(
          expected_online_params, online_params, online_transition, target_loss_key
      )
      chex.assert_trees_all_equal_shapes(d_loss_d_expected_params, online_params)
      target_updates, new_opt_state = optimizer.update(d_loss_d_expected_params, new_opt_state) # RMS Prop
      chex.assert_trees_all_equal_shapes(target_updates, online_params)
      target_online_params = optax.apply_updates(expected_online_params, target_updates)
      chex.assert_trees_all_equal_shapes(target_online_params, online_params)

      # print("calculated the target parameters")

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
      assert len(logits) > 0, f"logits was length={len(logits)} but expected >0"
      zero_padded_logits = jnp.array(logits + [0] * (replay.capacity - len(logits)), dtype=jnp.float32)
      assert len(zero_padded_logits) == replay.capacity, f"{len(zero_padded_logits)} != {replay.capacity}"
      d_loss_d_meta_params = jax.grad(meta_loss_fn)(
          zero_padded_logits, transitions, online_params, target_params, online_transition, opt_state, rng_key
      )
      # print("DONE META LOSS FUNCTION GRADIENT")
      # print(f"{len(d_loss_d_meta_params)=}", f"{len(logits)=}", f"{replay.capacity=}")
      assert len(d_loss_d_meta_params) == replay.capacity, f"d_loss_d_meta_params was length={len(d_loss_d_meta_params)} but expected {replay.capacity}"
      meta_updates, new_meta_opt_state = meta_optimizer.update(d_loss_d_meta_params, meta_opt_state)
      # print(f"{meta_updates=}")
      assert len(meta_updates) == replay.capacity, f"meta_updates was length={len(meta_updates)} but expected {replay.capacity}"
      new_meta_params_zero_padded = optax.apply_updates(zero_padded_logits, meta_updates)
      assert len(new_meta_params_zero_padded) == replay.capacity, f"new_meta_params_zero_padded was length={len(new_meta_params_zero_padded)} but expected {replay.capacity}"
      new_meta_params = list(new_meta_params_zero_padded[:len(logits)])
      assert len(new_meta_params) == len(logits), f"new_meta_params was length={len(new_meta_params)} but expected {len(logits)}"
      return rng_key, new_meta_opt_state, new_meta_params

    self._update = update #jax.jit(update)
    self._meta_update = meta_update #jax.jit(meta_update)

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
        print(f"Calling META LEARNING on {self._frame_t}")
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
      print(f"Calling LEARNING on {self._frame_t}")
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    print("RESETTING the agent.")
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


# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
_ENVIRONMENT_NAME = flags.DEFINE_string('environment_name', 'pong', '')
_ENVIRONMENT_HEIGHT = flags.DEFINE_integer('environment_height', 84, '')
_ENVIRONMENT_WIDTH = flags.DEFINE_integer('environment_width', 84, '')
_REPLAY_CAPACITY = flags.DEFINE_integer('replay_capacity', int(10000), '')
_COMPRESS_STATE = flags.DEFINE_bool('compress_state', True, '')
_MIN_REPLAY_CAPACITY_FRACTION = flags.DEFINE_float(
    'min_replay_capacity_fraction', 0.05, ''
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 10, '')
_MAX_FRAMES_PER_EPISODE = flags.DEFINE_integer(
    'max_frames_per_episode', 108000, ''
)  # 30 mins.
_NUM_ACTION_REPEATS = flags.DEFINE_integer('num_action_repeats', 4, '')
_NUM_STACKED_FRAMES = flags.DEFINE_integer('num_stacked_frames', 4, '')
_EXPLORATION_EPSILON_BEGIN_VALUE = flags.DEFINE_float(
    'exploration_epsilon_begin_value', 1.0, ''
)
_EXPLORATION_EPSILON_END_VALUE = flags.DEFINE_float(
    'exploration_epsilon_end_value', 0.1, ''
)
_EXPLORATION_EPSILON_DECAY_FRAME_FRACTION = flags.DEFINE_float(
    'exploration_epsilon_decay_frame_fraction', 0.1, ''
)
_EVAL_EXPLORATION_EPSILON = flags.DEFINE_float(
    'eval_exploration_epsilon', 0.05, ''
)
_TARGET_NETWORK_UPDATE_PERIOD = flags.DEFINE_integer(
    'target_network_update_period', 50, ''
)
_GRAD_ERROR_BOUND = flags.DEFINE_float('grad_error_bound', 1.0 / 32, '')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.00025, '')
_OPTIMIZER_EPSILON = flags.DEFINE_float('optimizer_epsilon', 0.01 / 32**2, '')
_ADDITIONAL_DISCOUNT = flags.DEFINE_float('additional_discount', 0.99, '')
_MAX_ABS_REWARD = flags.DEFINE_float('max_abs_reward', 1.0, '')
_SEED = flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 5, '')
_NUM_TRAIN_FRAMES = flags.DEFINE_integer(
    'num_train_frames', 1000, ''
)  # Per iteration.
_NUM_EVAL_FRAMES = flags.DEFINE_integer(
    'num_eval_frames', 500, ''
)  # Per iteration.
_LEARN_PERIOD = flags.DEFINE_integer('learn_period', 16, '')
_RESULTS_CSV_PATH = flags.DEFINE_string(
    'results_csv_path', './tmp/results.csv', ''
)
_META_LEARNING_RATE = flags.DEFINE_float('meta_learning_rate', 0.00025, '')


def main(argv):
  """Trains DQN agent on Atari."""
  del argv
  logging.info('DQN with MGSC on Atari on %s.', jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(_SEED.value)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
  )

  if _RESULTS_CSV_PATH.value:
    writer = parts.CsvWriter(_RESULTS_CSV_PATH.value)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        _ENVIRONMENT_NAME.value, seed=random_state.randint(1, 2**32)
    )
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', _ENVIRONMENT_NAME.value)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())
  num_actions = env.action_spec().num_values
  network_fn = networks.dqn_atari_network(num_actions)
  network = hk.transform(network_fn)

  def preprocessor_builder():
    return processors.atari(
        additional_discount=_ADDITIONAL_DISCOUNT.value,
        max_abs_reward=_MAX_ABS_REWARD.value,
        resize_shape=(_ENVIRONMENT_HEIGHT.value, _ENVIRONMENT_WIDTH.value),
        num_action_repeats=_NUM_ACTION_REPEATS.value,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=_NUM_STACKED_FRAMES.value,
        grayscaling=True,
    )

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(
      dm_env.TimeStep, sample_processed_timestep
  )
  sample_network_input = sample_processed_timestep.observation
  # print(f"===========\n{num_actions=}\n{type(sample_processed_timestep)}\n{sample_processed_timestep=}\n===========")
  chex.assert_shape(
      sample_network_input,
      (
          _ENVIRONMENT_HEIGHT.value,
          _ENVIRONMENT_WIDTH.value,
          _NUM_STACKED_FRAMES.value,
      ),
  )

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(
          _MIN_REPLAY_CAPACITY_FRACTION.value
          * _REPLAY_CAPACITY.value
          * _NUM_ACTION_REPEATS.value
      ),
      decay_steps=int(
          _EXPLORATION_EPSILON_DECAY_FRAME_FRACTION.value
          * _NUM_ITERATIONS.value
          * _NUM_TRAIN_FRAMES.value
      ),
      begin_value=_EXPLORATION_EPSILON_BEGIN_VALUE.value,
      end_value=_EXPLORATION_EPSILON_END_VALUE.value,
  )

  if _COMPRESS_STATE.value:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t),
      )

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t),
      )

  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.MGSCFiFoTransitionReplay( # CHANGED
      _REPLAY_CAPACITY.value, replay_structure, random_state, encoder, decoder
  )

  optimizer = optax.rmsprop(
      learning_rate=_LEARNING_RATE.value,
      decay=0.95,
      eps=_OPTIMIZER_EPSILON.value,
      centered=True,
  )

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = TestMGSCDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      optimizer=optimizer,
      transition_accumulator=replay_lib.TransitionAccumulator(),
      replay=replay,
      batch_size=_BATCH_SIZE.value,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=_MIN_REPLAY_CAPACITY_FRACTION.value,
      learn_period=_LEARN_PERIOD.value,
      target_network_update_period=_TARGET_NETWORK_UPDATE_PERIOD.value,
      grad_error_bound=_GRAD_ERROR_BOUND.value,
      rng_key=train_rng_key,
      meta_optimizer=optax.adam(
        learning_rate=_META_LEARNING_RATE.value
      )
  )
  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=_EVAL_EXPLORATION_EPSILON.value,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= _NUM_ITERATIONS.value:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    num_train_frames = 0 if state.iteration == 0 else _NUM_TRAIN_FRAMES.value
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    train_trackers = parts.make_default_trackers(train_agent)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    eval_seq_truncated = itertools.islice(eval_seq, _NUM_EVAL_FRAMES.value)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    # Logging and checkpointing.
    human_normalized_score = atari_data.get_human_normalized_score(
        _ENVIRONMENT_NAME.value, eval_stats['episode_return']
    )
    capped_human_normalized_score = np.amin([1.0, human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * _NUM_TRAIN_FRAMES.value, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_state_value', train_stats['state_value'], '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1.0 - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)
