import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

LayersDict = Dict[str, tf.layers.Layer]
import collections
import numpy as np


def dense(input, input_size, output_size, activation, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse, initializer=tf.random_normal_initializer(stddev=0.15)):
        weights = tf.get_variable('weights', [input_size, output_size])
        biases = tf.get_variable('biases', [output_size])
        output = tf.matmul(input, weights) + biases
        if activation:
            output = activation(output)
        return output


def gru(input, hidden, input_size, hidden_size, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse, initializer=tf.random_normal_initializer(stddev=0.15)):
        Wxr = tf.get_variable('weights_xr', [input_size, hidden_size])
        Wxz = tf.get_variable('weights_xz', [input_size, hidden_size])
        Wxh = tf.get_variable('weights_xh', [input_size, hidden_size])
        Whr = tf.get_variable('weights_hr', [hidden_size, hidden_size])
        Whz = tf.get_variable('weights_hz', [hidden_size, hidden_size])
        Whh = tf.get_variable('weights_hh', [hidden_size, hidden_size])
        br = tf.get_variable('biases_r', [1, hidden_size])
        bz = tf.get_variable('biases_z', [1, hidden_size])
        bh = tf.get_variable('biases_h', [1, hidden_size])

        x, h_ = input, hidden
        r = tf.sigmoid(tf.matmul(x, Wxr) + tf.matmul(h_, Whr) + br)
        z = tf.sigmoid(tf.matmul(x, Wxz) + tf.matmul(h_, Whz) + bz)

        h_hat = tf.tanh(tf.matmul(x, Wxh) + tf.matmul(tf.multiply(r, h_), Whh) + bh)

        output = tf.multiply((1 - z), h_hat) + tf.multiply(z, h_)

        return output


def networks_sequential(inputs: tf.Tensor, layers: LayersDict) -> tf.Tensor:
    """Applies a sequence of layers to an input."""
    output = inputs
    for layer in layers.values():
        output = layer(output)
    output = tf.squeeze(output, axis=1)
    return output


def networks_build_mlp(
        hid_sizes: Iterable[int],
        name: Optional[str] = None,
        activation: Optional[Callable] = tf.nn.relu,
        initializer: Optional[Callable] = None,
) -> LayersDict:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    # Hidden layers
    for i, size in enumerate(hid_sizes):
        key = f"{name}_dense{i}"
        layer = tf.layers.Dense(
            size, activation=activation, kernel_initializer=initializer, name=key
        )  # type: tf.layers.Layer
        layers[key] = layer

    # Final layer
    layer = tf.layers.Dense(
        1, kernel_initializer=initializer, name=f"{name}_dense_final"
    )  # type: tf.layers.Layer
    layers[f"{name}_dense_final"] = layer

    return layers


# Copyright 2019 The Dreamer Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def networks_build_encoder(
        hid_sizes: Iterable[int],
        name: Optional[str] = None,
        activation: Optional[Callable] = tf.nn.relu,
        initializer: Optional[Callable] = None,
) -> LayersDict:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    # Hidden layers
    for i, size in enumerate(hid_sizes):
        key = f"{name}_dense{i}"
        layer = tf.layers.Dense(
            size, activation=activation, kernel_initializer=initializer, name=key
        )  # type: tf.layers.Layer
        layers[key] = layer

    # Final layer
    layer = tf.layers.Dense(
        1, kernel_initializer=initializer, name=f"{name}_dense_final"
    )  # type: tf.layers.Layer
    layers[f"{name}_dense_final"] = layer

    return layers


def encoder(obs, reuse=True):
    with tf.variable_scope('encoder', reuse=reuse):
        kwargs = dict(strides=2, activation=tf.nn.relu)
        # hidden = tf.reshape(obs, [-1] + obs.shape[2:].as_list())
        hidden = obs
        hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
        hidden = tf.layers.conv2d(hidden, 64, 4, **kwargs)
        hidden = tf.layers.conv2d(hidden, 128, 4, **kwargs)
        hidden = tf.layers.conv2d(hidden, 256, 4, **kwargs)
        hidden = tf.layers.flatten(hidden)
        assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
        hidden = tf.reshape(hidden, shape(obs)[:1] + [
            np.prod(hidden.shape[1:].as_list())])
    return hidden


def decoder(features, data_shape, std=1.0, reuse=True, deterministic=True):
    with tf.variable_scope('decoder', reuse=reuse):
        kwargs = dict(strides=2, activation=tf.nn.relu)
        hidden = tf.layers.dense(features, 1024, None)
        hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
        hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
        hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
        mean = tf.layers.conv2d_transpose(hidden, data_shape[-1], 6, strides=2)
        assert mean.shape[1:].as_list() == list(data_shape), (mean.shape[1:].as_list(), list(data_shape))
        mean = tf.reshape(mean, [-1] + list(data_shape))
    if deterministic:
        return mean
    else:
        return tfd.Independent(tfd.Normal(mean, std), len(data_shape))


def preprocess(observ, bits, deterministic=True):
    bins = 2 ** bits
    image = tf.cast(observ, tf.float32)
    if bits < 8:
        image = tf.floor(image / 2 ** (8 - bits))
    image = image / bins
    if not deterministic:
        image = image + tf.random_uniform(tf.shape(image), 0, 1.0 / bins)
    image = image - 0.5
    return image


def postprocess(image, bits, dtype=tf.float32):
    bins = 2 ** bits
    if dtype == tf.float32:
        image = tf.floor(bins * (image + 0.5)) / bins
    elif dtype == tf.uint8:
        image = image + 0.5
        image = tf.floor(bins * image)
        image = image * (256.0 / bins)
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    else:
        raise NotImplementedError(dtype)
    return image


_builtin_zip = zip


def map_(function, *structures, **kwargs):
    # Named keyword arguments are not allowed after *args in Python 2.
    flatten = kwargs.pop('flatten', False)
    assert not kwargs, 'map() got unexpected keyword arguments.'

    def impl(function, *structures):
        if len(structures) == 0:
            return structures
        if all(isinstance(s, (tuple, list)) for s in structures):
            if len(set(len(x) for x in structures)) > 1:
                raise ValueError('Cannot merge tuples or lists of different length.')
            args = tuple((impl(function, *x) for x in _builtin_zip(*structures)))
            if hasattr(structures[0], '_fields'):  # namedtuple
                return type(structures[0])(*args)
            else:  # tuple, list
                return type(structures[0])(args)
        if all(isinstance(s, dict) for s in structures):
            if len(set(frozenset(x.keys()) for x in structures)) > 1:
                raise ValueError('Cannot merge dicts with different keys.')
            merged = {
                k: impl(function, *(s[k] for s in structures))
                for k in structures[0]}
            return type(structures[0])(merged)
        return function(*structures)

    result = impl(function, *structures)
    if flatten:
        result = flatten_(result)
    return result


def flatten_(structure):
    if isinstance(structure, dict):
        result = ()
        for key in sorted(list(structure.keys())):
            result += flatten_(structure[key])
        return result
    if isinstance(structure, (tuple, list)):
        result = ()
        for element in structure:
            result += flatten_(element)
        return result
    return (structure,)


def tools_mask(tensor, mask=None, length=None, value=0, debug=False):
    if len([x for x in (mask, length) if x is not None]) != 1:
        raise KeyError('Exactly one of mask and length must be provided.')
    with tf.name_scope('mask'):
        if mask is None:
            range_ = tf.range(tensor.shape[1].value)
            mask = range_[None, :] < length[:, None]
        batch_dims = mask.shape.ndims
        while tensor.shape.ndims > mask.shape.ndims:
            mask = mask[..., None]
        multiples = [1] * batch_dims + tensor.shape[batch_dims:].as_list()
        mask = tf.tile(mask, multiples)
        masked = tf.where(mask, tensor, value * tf.ones_like(tensor))
        if debug:
            masked = tf.check_numerics(masked, 'masked')
        return masked


map = map_
flatten = flatten_


class Base(tf.nn.rnn_cell.RNNCell):

    def __init__(self, transition_tpl, posterior_tpl, reuse=None):
        super(Base, self).__init__(_reuse=reuse)
        self._posterior_tpl = posterior_tpl
        self._transition_tpl = transition_tpl
        self._debug = False

    @property
    def state_size(self):
        raise NotImplementedError

    @property
    def updates(self):
        return []

    @property
    def losses(self):
        return []

    @property
    def output_size(self):
        return (self.state_size, self.state_size)

    def zero_state(self, batch_size, dtype):
        return map_(
            lambda size: tf.zeros([batch_size, size], dtype),
            self.state_size)

    def features_from_state(self, state):
        raise NotImplementedError

    def dist_from_state(self, state, mask=None):
        raise NotImplementedError

    def divergence_from_states(self, lhs, rhs, mask=None):
        lhs = self.dist_from_state(lhs, mask)
        rhs = self.dist_from_state(rhs, mask)
        divergence = tfd.kl_divergence(lhs, rhs)
        if mask is not None:
            divergence = tools_mask(divergence, mask)
        return divergence

    def call(self, inputs, prev_state):
        obs, prev_action, use_obs = inputs
        if self._debug:
            with tf.control_dependencies([tf.assert_equal(use_obs, use_obs[0, 0])]):
                use_obs = tf.identity(use_obs)
        use_obs = use_obs[0, 0]
        zero_obs = map_(tf.zeros_like, obs)
        prior = self._transition_tpl(prev_state, prev_action, zero_obs)
        posterior = tf.cond(
            use_obs,
            lambda: self._posterior_tpl(prev_state, prev_action, obs),
            lambda: prior)
        return (prior, posterior), posterior


class RSSM(Base):

    def __init__(
            self, state_size, belief_size, embed_size,
            future_rnn=True, mean_only=False, min_stddev=0.1, activation=tf.nn.elu,
            num_layers=1):
        self._state_size = state_size[0]
        self._belief_size = belief_size
        self._embed_size = embed_size
        self._future_rnn = future_rnn
        self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
        self._kwargs = dict(units=self._embed_size, activation=activation)
        self._mean_only = mean_only
        self._min_stddev = min_stddev
        self._num_layers = num_layers
        super(RSSM, self).__init__(
            tf.make_template('transition', self._transition),
            tf.make_template('posterior', self._posterior))

    @property
    def state_size(self):
        return {
            'mean': self._state_size,
            'stddev': self._state_size,
            'sample': self._state_size,
            'belief': self._belief_size,
            'rnn_state': self._belief_size,
        }

    @property
    def feature_size(self):
        return self._belief_size + self._state_size

    def dist_from_state(self, state, mask=None):
        if mask is not None:
            stddev = tools_mask(state['stddev'], mask, value=1)
        else:
            stddev = state['stddev']
        dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
        return dist

    def features_from_state(self, state):
        return tf.concat([state['sample'], state['belief']], -1)

    def divergence_from_states(self, lhs, rhs, mask=None):
        lhs = self.dist_from_state(lhs, mask)
        rhs = self.dist_from_state(rhs, mask)
        divergence = tfd.kl_divergence(lhs, rhs)
        if mask is not None:
            divergence = tools_mask(divergence, mask)
        return divergence

    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def mmd_from_states(self, lhs, rhs, mask=None):
        lhs = tf.squeeze(lhs['sample'], axis=1)
        rhs = tf.squeeze(rhs['sample'], axis=1)
        x_kernel = self.compute_kernel(lhs, lhs)
        y_kernel = self.compute_kernel(rhs, rhs)
        xy_kernel = self.compute_kernel(lhs, rhs)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        if mask is not None:
            mmd = tools_mask(mmd, mask)
        return mmd

    def _transition(self, prev_state, prev_action, zero_obs):
        print('prev_state[sample], prev_action: ', prev_state['sample'], prev_action)
        hidden = tf.concat([prev_state['sample'], prev_action], -1)
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
        if self._future_rnn:
            hidden = belief
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        mean = tf.layers.dense(hidden, self._state_size, None)
        stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
        stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': belief,
            'rnn_state': rnn_state,
        }

    def _posterior(self, prev_state, prev_action, obs):
        prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
        hidden = tf.concat([prior['belief'], obs], -1)
        for _ in range(self._num_layers):
            hidden = tf.layers.dense(hidden, **self._kwargs)
        mean = tf.layers.dense(hidden, self._state_size, None)
        stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
        stddev += self._min_stddev
        if self._mean_only:
            sample = mean
        else:
            sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': prior['belief'],
            'rnn_state': prior['rnn_state'],
        }


def closed_loop(cell, embedded, prev_action, debug=False):
    use_obs = tf.ones(tf.shape(embedded[:, :, :1])[:3], tf.bool)
    (prior, posterior), _ = tf.nn.dynamic_rnn(
        cell, (embedded, prev_action, use_obs), dtype=tf.float32)
    return prior, posterior


def shape(tensor):
    static = tensor.get_shape().as_list()
    dynamic = tf.unstack(tf.shape(tensor))
    assert len(static) == len(dynamic)
    combined = [d if s is None else s for s, d in zip(static, dynamic)]
    return combined
