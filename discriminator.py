import tensorflow as tf
import ops


class Discriminator(object):
    def __init__(self, in_dim, out_dim, size, lr, do_keep_prob, weight_decay, use_airl=True, airl_entropy_weight=1.0,
                 airl_discount_factor=1.0, phi_hidden_size=None, state_only=False):
        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.solver_params = {
            'lr': lr,
            'weight_decay': weight_decay
        }

        self.use_airl = use_airl
        # if self.use_airl:
        self.airl_entropy_weight = airl_entropy_weight
        self.airl_discount_factor = airl_discount_factor
        self.phi_hidden_size = phi_hidden_size
        self.state_only = state_only
        self.reward = None
        self.shaped_reward_output = None

    def forward(self, state, action, prev_state=None, done_inp=None, log_policy_act_prob=None, reuse=False):

        with tf.variable_scope('discriminator'):
            if self.state_only:
                concat = tf.concat(axis=1, values=[state])
            else:
                concat = tf.concat(axis=1, values=[state, action])
            h0 = ops.dense(concat, self.arch_params['in_dim'], self.arch_params['n_hidden_0'], tf.nn.relu, 'dense0', reuse)
            h1 = ops.dense(h0, self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1'], tf.nn.relu, 'dense1', reuse)
            relu1_do = tf.nn.dropout(h1, self.arch_params['do_keep_prob'])
            d = ops.dense(relu1_do, self.arch_params['n_hidden_1'], self.arch_params['out_dim'], None, 'dense2', reuse)


            if self.use_airl:
                self.reward = d
                with tf.variable_scope("phi_network"):
                    if self.phi_hidden_size is None:
                        hid_sizes = (32, 32)
                    else:
                        hid_sizes = self.phi_hidden_size

                    with tf.variable_scope("phi", reuse=tf.AUTO_REUSE):
                        old_o = prev_state
                        new_o = state

                        # Weight share, just with different inputs old_o and new_o
                        phi_mlp = ops.networks_build_mlp(hid_sizes=hid_sizes, name="shaping")
                        old_shaping_output = ops.networks_sequential(old_o, phi_mlp)
                        new_shaping_output = ops.networks_sequential(new_o, phi_mlp)

                    # end_potential is the potential when the episode terminates.
                    if self.airl_discount_factor == 1.0:
                        # If undiscounted, terminal state must have potential 0.
                        end_potential = tf.constant(0.0)
                    else:
                        # Otherwise, it can be arbitrary, so make a trainable variable.
                        end_potential = tf.Variable(
                            name="end_phi", shape=(), dtype=tf.float32, initial_value=0.0
                        )

                with tf.variable_scope("f_network"):
                    new_shaping = (
                            done_inp * end_potential
                            + (1 - done_inp) * new_shaping_output
                    )
                    self.shaped_reward_output = (
                            self.reward
                            + self.airl_discount_factor * new_shaping
                            - old_shaping_output
                    )

                d = (
                    self.shaped_reward_output - tf.stop_gradient(log_policy_act_prob)
                )

        return d, self.shaped_reward_output, self.reward

    def backward(self, loss):
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # weight decay
        loss += self.solver_params['weight_decay'] * tf.add_n([tf.nn.l2_loss(w) for w in self.weights if 'weights' in w.name])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=self.weights)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self, objective):
        self.loss = objective
        self.minimize = self.backward(self.loss)
