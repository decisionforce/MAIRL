from collections import OrderedDict
import ops
import tensorflow as tf


class ForwardModel(object):
    def __init__(self, state_size, action_size, encoding_size, lr, forward_model_type='gru', obs_mode='state',
                 use_scale_dot_product=True, use_skip_connection=True, use_dropout=False):
        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size
        self.obs_mode = obs_mode

        self.lr = lr
        self.forward_model_type = forward_model_type
        if self.forward_model_type in ['kl-rssm', 'mmd-rssm']:
            self._cell = tf.contrib.rnn.GRUBlockCell(encoding_size)
        elif self.forward_model_type in ['transformer']:
            self.num_heads = 4
            assert encoding_size % self.num_heads == 0, 'encoding_size % self.num_heads == {}'.format(
                encoding_size % self.num_heads)
            self.depth = encoding_size // self.num_heads
            self.hidden_dropout_prob = 0.1
            self.use_scale_dot_product = use_scale_dot_product
            self.use_skip_connection = use_skip_connection
            self.use_dropout = use_dropout

    def forward(self, input, reuse=False):
        with tf.variable_scope('forward_model'):
            state = tf.cast(input[0], tf.float32)
            action = tf.cast(input[1], tf.float32)
            gru_state = tf.cast(input[2], tf.float32)

            if self.forward_model_type in ['kl-rssm', 'mmd-rssm']:
                hidden = tf.concat([action], -1)
                for i in range(2):
                    hidden = tf.layers.dense(hidden, **dict(units=self.encoding_size, activation=tf.nn.elu),
                                             name='prior_enc_{}'.format(i), reuse=tf.AUTO_REUSE)
                belief, rnn_state = self._cell(hidden, tf.zeros_like(hidden))
                prior = {
                    'belief': belief,
                }
                hidden = tf.concat([prior['belief'], state], -1)
                for i in range(2):
                    hidden = tf.layers.dense(hidden, **dict(units=self.encoding_size, activation=tf.nn.elu),
                                             name='post_dec_{}'.format(i), reuse=tf.AUTO_REUSE)
                mean = tf.layers.dense(hidden, self.state_size, None, name='post_mean', reuse=tf.AUTO_REUSE)

                sample = mean

                gru_state = belief
                next_state = sample
                divergence_loss = 0.
            elif self.forward_model_type in ['transformer']:
                # State embedding
                state_embedder1 = ops.dense(state, self.state_size, self.encoding_size, tf.nn.relu, "encoder1_state",
                                            reuse)
                divergence_loss = 0.
                state_embedder2 = ops.dense(state_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid,
                                            "encoder2_state", reuse)

                # Action embedding
                action_embedder1 = ops.dense(action, self.action_size, self.encoding_size, tf.nn.relu,
                                             "encoder1_action", reuse)
                action_embedder2 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid,
                                             "encoder2_action", reuse)

                # Multi-head
                if self.use_scale_dot_product:
                    action_embedder3 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid,
                                                 "value", reuse)
                    batch_size = tf.shape(state)[0]
                    state_embedder2_query = self.split_heads(state_embedder2, batch_size)  # query
                    action_embedder2 = self.split_heads(action_embedder2, batch_size)  # key
                    action_embedder3 = self.split_heads(action_embedder3, batch_size)  # value
                    scaled_attention = self.scaled_dot_product_attention(state_embedder2_query, action_embedder2,
                                                                         action_embedder3,
                                                                         mask=None)  # scaled_attention =
                    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
                    joint_embedding = tf.reshape(scaled_attention, (batch_size, self.encoding_size))
                    # Skip Connection
                    if self.use_skip_connection:
                        joint_embedding = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, None,
                                                    "cross_att_dense", reuse)
                        if self.use_dropout:
                            joint_embedding = tf.nn.dropout(joint_embedding, keep_prob=1 - self.hidden_dropout_prob, )
                        joint_embedding = joint_embedding + state_embedder2
                else:
                    # Joint embedding
                    joint_embedding = tf.multiply(state_embedder2, action_embedder2)

                # Next state prediction
                hidden1 = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder3",
                                    reuse)
                hidden2 = ops.dense(hidden1, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder4", reuse)
                hidden3 = ops.dense(hidden2, self.encoding_size, self.encoding_size, tf.nn.relu, "decoder1", reuse)
                next_state = ops.dense(hidden3, self.encoding_size, self.state_size, None, "decoder2", reuse)

                gru_state = tf.cast(gru_state, tf.float64)
            else:
                # State embedding
                state_embedder1 = ops.dense(state, self.state_size, self.encoding_size, tf.nn.relu, "encoder1_state",
                                            reuse)
                gru_state = ops.gru(state_embedder1, gru_state, self.encoding_size, self.encoding_size, 'gru1', reuse)
                divergence_loss = 0.
                state_embedder2 = ops.dense(gru_state, self.encoding_size, self.encoding_size, tf.sigmoid,
                                            "encoder2_state", reuse)

                # Action embedding
                action_embedder1 = ops.dense(action, self.action_size, self.encoding_size, tf.nn.relu,
                                             "encoder1_action", reuse)
                action_embedder2 = ops.dense(action_embedder1, self.encoding_size, self.encoding_size, tf.sigmoid,
                                             "encoder2_action", reuse)

                # Joint embedding
                joint_embedding = tf.multiply(state_embedder2, action_embedder2)

                # Next state prediction
                hidden1 = ops.dense(joint_embedding, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder3",
                                    reuse)
                hidden2 = ops.dense(hidden1, self.encoding_size, self.encoding_size, tf.nn.relu, "encoder4", reuse)
                hidden3 = ops.dense(hidden2, self.encoding_size, self.encoding_size, tf.nn.relu, "decoder1", reuse)
                next_state = ops.dense(hidden3, self.encoding_size, self.state_size, None, "decoder2", reuse)

                gru_state = tf.cast(gru_state, tf.float64)

            return next_state, gru_state, divergence_loss

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # add the mask zero out padding tokens.
        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)

        return tf.matmul(attention_weights, value)

    def backward(self, loss):
        if self.obs_mode == 'state':
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_model')
        elif self.obs_mode == 'pixel':
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_model') \
                           + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') \
                           + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        else:
            raise NotImplementedError
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self, objective):
        self.loss = objective
        self.minimize = self.backward(self.loss)
