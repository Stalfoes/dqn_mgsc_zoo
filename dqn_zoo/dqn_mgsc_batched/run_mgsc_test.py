import jax
import jax.numpy as jnp
import rlax
import chex
import optax
import haiku as hk
import numpy as np

from absl import app
from absl import flags
from absl import logging

from dqn_zoo import parts
from dqn_zoo import networks
from dqn_zoo import replay as replay_lib


SEED = 0

_ENVIRONMENT_HEIGHT = 84
_ENVIRONMENT_WIDTH = 84
_NUM_STACKED_FRAMES = 4
_REPLAY_CAPACITY = 10

_batch_q_learning = jax.vmap(rlax.q_learning)

num_actions = 4
network_fn = networks.dqn_atari_network(num_actions)
network = hk.transform(network_fn)
sample_network_input = jnp.full((_ENVIRONMENT_HEIGHT,_ENVIRONMENT_WIDTH,_NUM_STACKED_FRAMES), 0)
grad_error_bound = 1.0 / 32
random_state = np.random.RandomState(SEED)
replay_structure = replay_lib.Transition(
    s_tm1=None,
    a_tm1=None,
    r_t=None,
    discount_t=None,
    s_t=None,
)
replay = replay_lib.ReservoirTransitionReplay(_REPLAY_CAPACITY, replay_structure, random_state, None, None)
optimizer = None
meta_optimizer = None
opt_state = None
meta_opt_state = None


def norm_of_pytree(params, target_params):
    """Subtracts pytrees and flattens and takes the L2 norm of the flattened vector"""
    l2_norms = jax.tree_util.tree_map(lambda t, e: jnp.linalg.norm(t - e, ord=None) ** 2, target_params, params)
    l2_norms_list, _ = jax.tree_util.tree_flatten(l2_norms)
    reduced = jnp.sum(jnp.array(l2_norms_list))
    return reduced

def loss_fn(online_params, target_params, transitions, rng_key): # implemented by default in DQNZoo
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
    ) # should be an array of floats
    td_errors = rlax.clip_gradient(
        td_errors, -grad_error_bound, grad_error_bound
    ) # should be an array of floats
    losses = rlax.l2_loss(td_errors) # should be an array of floats
    chex.assert_shape(losses, (len(transitions.s_tm1),)) # chex.assert_shape(losses, (self._batch_size,))
    loss = jnp.mean(losses) # should be just a single float
    return loss

def meta_loss_fn(logits, transitions, online_params, target_params, online_transition, opt_state, rng_key):
    e_to_logits = jnp.power(jnp.e, jnp.array(logits))
    probabilities = e_to_logits / jnp.sum(e_to_logits)
    rng_key, exp_loss_key, target_loss_key = jax.random.split(rng_key, 3)

    online_transition = type(online_transition)(*[jnp.expand_dims(jnp.array(online_transition[i], dtype=online_transition[i].dtype), 0) for i in range(len(online_transition))])
    unbatched_transitions = [
        type(online_transition)(*[
            jnp.expand_dims(jnp.array(transitions[i][t], dtype=transitions[i].dtype), 0) for i in range(len(transitions))
        ])
        for t in range(len(transitions.s_tm1))
    ]
    
    grad_loss_fn = jax.grad(loss_fn)
    grads_list = [grad_loss_fn(online_params, target_params, trans, exp_loss_key) for trans in unbatched_transitions]
    weighted_grads = [jax.tree_util.tree_map(lambda v: v * prob, grad) for grad,prob in zip(grads_list, probabilities)]
    summed_weighted_grads = jax.tree_util.tree_map(lambda *v: sum(v), *weighted_grads)
    updates, new_opt_state = optimizer.update(summed_weighted_grads, opt_state) # RMS Prop
    expected_online_params = optax.apply_updates(online_params, updates) # this is just a summation of params + updates

    d_loss_d_expected_params = jax.grad(loss_fn)(
        expected_online_params, online_params, online_transition, target_loss_key
    )
    target_updates, new_opt_state = optimizer.update(d_loss_d_expected_params, new_opt_state) # RMS Prop
    target_online_params = optax.apply_updates(expected_online_params, target_updates) # summation
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
    d_loss_d_meta_params = jax.grad(meta_loss_fn)(
        logits, transitions, online_params, target_params, online_transition, opt_state, rng_key
    )
    meta_updates, new_meta_opt_state = meta_optimizer.update(d_loss_d_meta_params, meta_opt_state)
    new_meta_params = optax.apply_updates(jnp.array(logits), meta_updates)
    return rng_key, new_meta_opt_state, new_meta_params


def shape_of_pytree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    return jax.tree_util.tree_unflatten(treedef, [l.shape for l in leaves])

def make_params(network_rng_key):
    return network.init(network_rng_key, sample_network_input[None, ...])

def make_params_of_value(network_rng_key, value):
    params = make_params(network_rng_key)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    new_leaves = [jnp.full_like(leaf, value) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

def make_transition(value):
    transition=replay_lib.Transition(
            s_tm1=jnp.full((84,84,4), value, dtype=jnp.uint8),
            a_tm1=jnp.full((), value, dtype=jnp.uint32),
            r_t=jnp.full((), value, dtype=jnp.float32),
            discount_t=jnp.full((), value, dtype=jnp.float32),
            s_t=jnp.full((84,84,4), value, dtype=jnp.uint8)
        )
    return transition

def make_batched_transitions(length):
    samples = [make_transition(n) for n in range(length)]
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    batched = replay_lib.Transition(*stacked)
    chex.assert_shape(batched[0], (10,84,84,4))
    chex.assert_shape(batched[1], (10,))
    return batched

def test_stack_transition():
    online_transition = make_transition(0)
    online_transition = type(online_transition)(*[jnp.expand_dims(jnp.array(online_transition[i], dtype=online_transition[i].dtype), 0) for i in range(len(online_transition))])
    chex.assert_shape(online_transition[0], (1,84,84,4))
    chex.assert_shape(online_transition[1], (1,))
    print('stack_transition is all good')

def test_unbatch_transition():
    transitions = make_batched_transitions(_REPLAY_CAPACITY)
    unbatched_transitions = [
        type(transitions)(*[
            jnp.expand_dims(jnp.array(transitions[i][t], dtype=transitions[i].dtype), 0) for i in range(len(transitions))
        ])
        for t in range(len(transitions.s_tm1))
    ]
    assert len(unbatched_transitions) == _REPLAY_CAPACITY, f'Length of unbatched transitions must be {_REPLAY_CAPACITY} but {len(unbatched_transitions)} was received'
    chex.assert_shape(unbatched_transitions[0][0], (1,84,84,4))
    chex.assert_shape(unbatched_transitions[0][1], (1,))
    print('unbatch_transition is all good')

def test_norm_of_pytree():
    key = jax.random.PRNGKey(SEED)
    key, pkey1, pkey2 = jax.random.split(key, 3)
    params = make_params(pkey1)
    target_params = make_params(pkey2)
    a = norm_of_pytree(params, target_params)
    chex.assert_shape(a, ())
    print(f"norm_of_pytree worked with the norm being {a}")

def test_meta_update_shapes():
    rng_key = jax.random.PRNGKey(SEED)
    rng_key, online_key, target_key = jax.random.split(rng_key, 3)
    logits = [0] * _REPLAY_CAPACITY
    transitions = make_batched_transitions(_REPLAY_CAPACITY)
    online_transition = make_transition(0)
    online_params = make_params(online_key)
    target_params = make_params(target_key)
    _, _, new_logits = meta_update(rng_key, opt_state, meta_opt_state, online_params, target_params, transitions, logits, online_transition)
    assert isinstance(new_logits, list), f"new_logits must be of type list, instead {type(list)} was returned"
    assert len(new_logits) == _REPLAY_CAPACITY, f"new_logits must be length {_REPLAY_CAPACITY} but instead it was {len(new_logits)}"

def test_weighted_sum():
    rng_key = jax.random.PRNGKey(SEED)
    rng_keys = jax.random.split(rng_key, _REPLAY_CAPACITY)
    logits = list(range(_REPLAY_CAPACITY))
    grads_list = [make_params_of_value(rng_keys[i], 1) for i in range(_REPLAY_CAPACITY)]
    e_to_logits = jnp.power(jnp.e, jnp.array(logits))
    probabilities = e_to_logits / jnp.sum(e_to_logits)
    weighted_grads = [jax.tree_util.tree_map(lambda v: v * prob, grad) for grad,prob in zip(grads_list, probabilities)]
    summed_weighted_grads = jax.tree_util.tree_map(lambda *v: sum(v), *weighted_grads)
    """{
    'sequential/sequential/conv2_d': {'b': (32,), 'w': (8, 8, 4, 32)},
    'sequential/sequential/conv2_d_1': {'b': (64,), 'w': (4, 4, 32, 64)},
    'sequential/sequential/conv2_d_2': {'b': (64,), 'w': (3, 3, 64, 64)},
    'sequential/sequential_1/linear': {'b': (512,), 'w': (3136, 512)},
    'sequential/sequential_1/linear_1': {'b': (4,), 'w': (512, 4)}}
    """
    assert summed_weighted_grads['sequential/sequential/conv2_d']['b'][20] == 1, f"Summed ['sequential/sequential/conv2_d']['b'] must be 1, but it was {summed_weighted_grads['sequential/sequential/conv2_d']['b']}"
    assert summed_weighted_grads['sequential/sequential/conv2_d']['w'][2,2,20,20] == 1, f"Summed ['sequential/sequential/conv2_d']['w'] must be 1, but it was {summed_weighted_grads['sequential/sequential/conv2_d']['w']}"
    assert summed_weighted_grads['sequential/sequential_1/linear']['b'][300] == 1, f"Summed ['sequential/sequential_1/linear']['b'] must be 1, but it was {summed_weighted_grads['sequential/sequential_1/linear']['b']}"
    assert summed_weighted_grads['sequential/sequential_1/linear']['w'][3000,300] == 1, f"Summed ['sequential/sequential_1/linear']['w'] must be 1, but it was {summed_weighted_grads['sequential/sequential_1/linear']['w']}"
    print("weighted_sum is good")

def main(argv):
    test_norm_of_pytree() # OK
    test_stack_transition() # OK
    test_unbatch_transition() # OK
    # TODO -- test output shape of updates and grads to the logits
    #         check to see where the gradients travel when taking the gradient
    test_weighted_sum()


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    jax.config.config_with_absl()
    app.run(main)