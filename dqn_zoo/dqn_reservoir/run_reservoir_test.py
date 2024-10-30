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
from dqn_zoo.dqn_mgsc import agent


NUM_EPOCHS = 5000
CAPACITY = 50
NUM_TRANSITIONS = 200
EPSILON = 0.025 # 2.5% error


def test():
    replay_structure = replay_lib.Transition(None, None, None, None, None)
    transitions = [replay_lib.Transition(i, i, i, i, i) for i in range(NUM_TRANSITIONS)]
    counts = collections.defaultdict(lambda: 0)
    for epoch in range(NUM_EPOCHS):
        rng = np.random.RandomState(epoch)
        replay = replay_lib.ReservoirTransitionReplay(CAPACITY, replay_structure, rng, None, None)
        for trans in transitions:
            replay.add(trans)
        for trans in replay.get(replay.ids()):
            counts[trans] += 1
    expected_percent = CAPACITY / NUM_TRANSITIONS
    for k in counts.keys():
        counts[k] = counts[k] / NUM_EPOCHS
        assert expected_percent - EPSILON <= counts[k] and counts[k] <= expected_percent + EPSILON, f"{counts[k]=} was not within the expected value of {expected_percent=}±{EPSILON}"
    print("Operates as expected!")


if __name__ == '__main__':
    test()