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


from time import time


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
from dqn_zoo.dqn_mgsc_batched import agent
from dqn_zoo.dqn_mgsc_batched import jaxpr_grapher

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
_ENVIRONMENT_NAME = flags.DEFINE_string('environment_name', 'pong', '')
_ENVIRONMENT_HEIGHT = flags.DEFINE_integer('environment_height', 84, '')
_ENVIRONMENT_WIDTH = flags.DEFINE_integer('environment_width', 84, '')
_REPLAY_CAPACITY = flags.DEFINE_integer('replay_capacity', int(1e6), '')
_COMPRESS_STATE = flags.DEFINE_bool('compress_state', True, '')
_MIN_REPLAY_CAPACITY_FRACTION = flags.DEFINE_float(
    'min_replay_capacity_fraction', 0.05, ''
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, '')
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
    'exploration_epsilon_decay_frame_fraction', 0.02, ''
)
_EVAL_EXPLORATION_EPSILON = flags.DEFINE_float(
    'eval_exploration_epsilon', 0.05, ''
)
_TARGET_NETWORK_UPDATE_PERIOD = flags.DEFINE_integer(
    'target_network_update_period', int(4e4), ''
)
_GRAD_ERROR_BOUND = flags.DEFINE_float('grad_error_bound', 1.0 / 32, '')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.00025, '')
_OPTIMIZER_EPSILON = flags.DEFINE_float('optimizer_epsilon', 0.01 / 32**2, '')
_ADDITIONAL_DISCOUNT = flags.DEFINE_float('additional_discount', 0.99, '')
_MAX_ABS_REWARD = flags.DEFINE_float('max_abs_reward', 1.0, '')
_SEED = flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 200, '')
_NUM_TRAIN_FRAMES = flags.DEFINE_integer(
    'num_train_frames', int(1e6), ''
)  # Per iteration.
_NUM_EVAL_FRAMES = flags.DEFINE_integer(
    'num_eval_frames', int(5e5), ''
)  # Per iteration.
_LEARN_PERIOD = flags.DEFINE_integer('learn_period', 16, '')
_RESULTS_CSV_PATH = flags.DEFINE_string(
    'results_csv_path', './tmp/results.csv', ''
)
_META_LEARNING_RATE = flags.DEFINE_float('meta_learning_rate', 0.00025, '')
_META_BATCH_SIZE = flags.DEFINE_integer('meta_batch_size', int(1e5), '')


def main(argv):
  """Trains DQN agent on Atari."""
  del argv
  logging.info('DQN with MGSC with batches of size=%d on Atari on %s.', _META_BATCH_SIZE.value, jax.lib.xla_bridge.get_backend().platform)
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

  train_agent = agent.MGSCDqn(
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
        learning_rate = _META_LEARNING_RATE.value
      ),
      meta_batch_size=_META_BATCH_SIZE.value
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
  writer.close()

  transitions = [
    replay_lib.Transition(
      s_tm1=np.full((84, 84, 4), 1, dtype=np.uint8),
      a_tm1=int(1),
      r_t=float(1),
      discount_t=float(0.99),
      s_t=np.full((84, 84, 4), 1, dtype=np.uint8)
    )
    for i in range(_META_BATCH_SIZE.value)
  ]
  # stack them
  transposed = zip(*transitions)
  stacked = [np.stack(xs, axis=0) for xs in transposed]
  transitions = replay_lib.Transition(*stacked)
  logits = [1.0 for i in range(_META_BATCH_SIZE.value)]
  online_transition = replay_lib.Transition(
    s_tm1=np.full((84, 84, 4), 1, dtype=np.uint8),
    a_tm1=int(1),
    r_t=float(1),
    discount_t=float(0.99),
    s_t=np.full((84, 84, 4), 1, dtype=np.uint8)
  )

  print('building env...')
  env = environment_builder()
  print('graphing...')
  # graph = jaxpr_grapher.jaxpr_graph(
  #   train_agent._unjitted_meta_update, # the function
  #   train_agent._rng_key,
  #   train_agent._opt_state,
  #   train_agent._meta_opt_state,
  #   train_agent._online_params,
  #   train_agent._target_params,
  #   transitions,
  #   logits,
  #   online_transition
  # )
  graph = jaxpr_grapher.jaxpr_graph(
    train_agent._unjitted_update, # the function
    train_agent._rng_key,
    train_agent._opt_state,
    train_agent._online_params,
    train_agent._target_params,
    transitions,
  )
  print('rendering...')
  graph.render(directory='.').replace('\\', '/')
  print('done.')


if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)
