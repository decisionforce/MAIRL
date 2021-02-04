import numpy as np
import tensorflow as tf

import os
import common
from ER import ER
from forward_model import ForwardModel
from discriminator import Discriminator
from policy import Policy
import ops

class MGAIL(object):
    def __init__(self, environment):

        self.env = environment

        # Create placeholders for all the inputs
        self.states_ = tf.placeholder("float", shape=(None,) + self.env.state_size, name='states_')  # Batch x State, previous state
        self.states = tf.placeholder("float", shape=(None,) + self.env.state_size, name='states')  # Batch x State, current_state
        self.actions = tf.placeholder("float", shape=(None, self.env.action_size), name='action')  # Batch x Action
        self.label = tf.placeholder("float", shape=(None, 1), name='label')
        self.gamma = tf.placeholder("float", shape=(), name='gamma')
        self.temp = tf.placeholder("float", shape=(), name='temperature')
        self.noise = tf.placeholder("float", shape=(), name='noise_flag')
        self.do_keep_prob = tf.placeholder("float", shape=(), name='do_keep_prob')
        if self.env.use_airl:
            self.done_ph = tf.placeholder(name="dones", shape=(None,), dtype=tf.float32)

        # Create MGAIL blocks
        self.forward_model = ForwardModel(state_size=self.env.state_size[0] if self.env.obs_mode == 'state' else self.env.encoder_feat_size,
                                          action_size=self.env.action_size,
                                          encoding_size=self.env.fm_size,
                                          lr=self.env.fm_lr,
                                          forward_model_type=self.env.forward_model_type,
                                          obs_mode=self.env.obs_mode,
                                          use_scale_dot_product=self.env.use_scale_dot_product,
                                          use_skip_connection=self.env.use_skip_connection,
                                          use_dropout=self.env.use_dropout)

        if self.env.obs_mode == 'pixel':
            if self.env.state_only:
                feat_in_dim = 1024  # self.env.encoder_feat_size[0]
                policy_input_feat = 1024
            else:
                feat_in_dim = 1024 + self.env.action_size  # self.env.encoder_feat_size[0]
                policy_input_feat = 1024
        else:
            if self.env.state_only:
                feat_in_dim = self.env.state_size[0]
                policy_input_feat = self.env.state_size[0]
            else:
                feat_in_dim = self.env.state_size[0] + self.env.action_size
                policy_input_feat = self.env.state_size[0]

        self.discriminator = Discriminator(
                                           in_dim=feat_in_dim,
                                           out_dim=self.env.disc_out_dim,
                                           size=self.env.d_size,
                                           lr=self.env.d_lr,
                                           do_keep_prob=self.do_keep_prob,
                                           weight_decay=self.env.weight_decay,
                                           use_airl=self.env.use_airl,
                                           phi_hidden_size=self.env.phi_size,
                                           state_only=self.env.state_only,
                                           )

        self.policy = Policy(in_dim=policy_input_feat,
                              out_dim=self.env.action_size,
                              size=self.env.p_size,
                              lr=self.env.p_lr,
                              do_keep_prob=self.do_keep_prob,
                              n_accum_steps=self.env.policy_accum_steps,
                              weight_decay=self.env.weight_decay)

        # Create experience buffers
        self.er_agent = ER(memory_size=self.env.er_agent_size,
                           state_dim=self.env.state_size,
                           action_dim=self.env.action_size,
                           reward_dim=1,  # stub connection
                           qpos_dim=self.env.qpos_size,
                           qvel_dim=self.env.qvel_size,
                           batch_size=self.env.batch_size,
                           history_length=1)

        self.er_expert = common.load_er(fname=os.path.join(self.env.run_dir, self.env.expert_data),
                                        batch_size=self.env.batch_size,
                                        history_length=1,
                                        traj_length=2)

        self.env.sigma = self.er_expert.actions_std / self.env.noise_intensity

        if self.env.obs_mode == 'pixel':
            current_states = ops.preprocess(self.states, bits=8)
            current_states_feat = ops.encoder(current_states, reuse=tf.AUTO_REUSE)
            prev_states = ops.preprocess(self.states_, bits=8)
            prev_states_feat = ops.encoder(prev_states, reuse=tf.AUTO_REUSE)
        else:
            # Normalize the inputs
            prev_states = common.normalize(self.states_, self.er_expert.states_mean, self.er_expert.states_std)
            current_states = common.normalize(self.states, self.er_expert.states_mean, self.er_expert.states_std)
            prev_states_feat = prev_states
            current_states_feat = current_states

        if self.env.continuous_actions:
            actions = common.normalize(self.actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        else:
            actions = self.actions

        # 1. Forward Model
        initial_gru_state = np.ones((1, self.forward_model.encoding_size))
        forward_model_prediction, _, divergence_loss = self.forward_model.forward([prev_states_feat, actions, initial_gru_state])
        if self.env.obs_mode == 'pixel':
            forward_model_prediction = ops.decoder(forward_model_prediction, data_shape=self.env.state_size, reuse=tf.AUTO_REUSE)
            self.forward_model_prediction = ops.postprocess(forward_model_prediction, bits=8, dtype=tf.uint8)
        else:
            self.forward_model_prediction = forward_model_prediction
        forward_model_loss = tf.reduce_mean(tf.square(current_states-forward_model_prediction)) + self.env.forward_model_lambda * tf.reduce_mean(divergence_loss)
        self.forward_model.train(objective=forward_model_loss)

        if self.env.use_airl:
            # 1.1 action log prob
            logits = self.policy.forward(current_states_feat)
            if self.env.continuous_actions:
                mean, logstd = logits, tf.log(tf.ones_like(logits))
                std = tf.exp(logstd)

                n_elts = tf.cast(tf.reduce_prod(mean.shape[1:]), tf.float32)  # first dimension is batch size
                log_normalizer = n_elts / 2. * (np.log(2 * np.pi).astype(np.float32)) + 1 / 2 * tf.reduce_sum(logstd,
                                                                                                              axis=1)
                # Diagonal Gaussian action probability, for every action
                action_logprob = -tf.reduce_sum(tf.square(actions - mean) / (2 * std), axis=1) - log_normalizer
            else:
                # Override since the implementation of tfp.RelaxedOneHotCategorical
                # yields positive values.
                if actions.shape[1:] != logits.shape[1:]:
                    actions = tf.cast(actions, tf.int8)
                    values = tf.one_hot(
                        actions, logits.shape.as_list()[-1], dtype=tf.float32)
                    assert values.shape == logits.shape, (values.shape, logits.shape)
                else:
                    values = actions

                # [0]'s implementation (see line below) seems to be an approximation
                # to the actual Gumbel Softmax density.
                # TODO: to confirm 'action' or 'value'
                action_logprob = -tf.reduce_sum(
                    -values * tf.nn.log_softmax(logits, axis=-1), axis=-1)
                # prob = logit[np.arange(self.action_test.shape[0]), self.action_test]
                # action_logprob = tf.log(prob)
            # 2. Discriminator
            self.discriminator.airl_entropy_weight = self.env.airl_entropy_weight
            # labels = tf.concat([1 - self.label, self.label], 1)
            # labels = 1 - self.label  # 0 for expert, 1 for policy
            labels = self.label  # 1 for expert, 0 for policy
            d, self.disc_shaped_reward_output, self.disc_reward = self.discriminator.forward(state=current_states_feat, action=actions, prev_state=prev_states_feat, done_inp=self.done_ph, log_policy_act_prob=action_logprob,)

            # 2.1 0-1 accuracy
            correct_predictions = tf.equal(tf.argmax(d, 1), tf.argmax(labels, 1))
            self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            # 2.2 prediction
            d_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=d,
                name="disc_loss",
            )
            # Construct generator reward:
            # \[\hat{r}(s,a) = \log(D_{\theta}(s,a)) - \log(1 - D_{\theta}(s,a)).\]
            # This simplifies to:
            # \[\hat{r}(s,a) = f_{\theta}(s,a) - \log \pi(a \mid s).\]
            # This is just an entropy-regularized objective
            # ent_bonus = -self.env.airl_entropy_weight * self.discriminator.log_policy_act_prob_ph
            # policy_train_reward = self.discriminator.reward_net.reward_output_train + ent_bonus
        else:
            # 2. Discriminator
            labels = tf.concat([1 - self.label, self.label], 1)
            d, _, _ = self.discriminator.forward(state=current_states_feat, action=actions)

            # 2.1 0-1 accuracy
            correct_predictions = tf.equal(tf.argmax(d, 1), tf.argmax(labels, 1))
            self.discriminator.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            # 2.2 prediction
            d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
        # cost sensitive weighting (weight true=expert, predict=agent mistakes)
        d_loss_weighted = self.env.cost_sensitive_weight * tf.multiply(tf.to_float(tf.equal(tf.squeeze(self.label), 1.)), d_cross_entropy) +\
                                                           tf.multiply(tf.to_float(tf.equal(tf.squeeze(self.label), 0.)), d_cross_entropy)
        discriminator_loss = tf.reduce_mean(d_loss_weighted)
        self.discriminator.train(objective=discriminator_loss)

        # 3. Collect experience
        mu = self.policy.forward(current_states_feat)
        if self.env.continuous_actions:
            a = common.denormalize(mu, self.er_expert.actions_mean, self.er_expert.actions_std)
            eta = tf.random_normal(shape=tf.shape(a), stddev=self.env.sigma)
            self.action_test = tf.squeeze(a + self.noise * eta)
        else:
            a = common.gumbel_softmax(logits=mu, temperature=self.temp)
            self.action_test = tf.argmax(a, dimension=1)

        # 4.3 AL
        def policy_loop(current_state_policy_update, t, total_cost, total_trans_err, env_term_sig, prev_state):
            if self.env.obs_mode == 'pixel':
                current_state_feat_policy_update = ops.encoder(current_state_policy_update, reuse=True)
                prev_state_feat_policy_update = ops.encoder(prev_state, reuse=True)
            else:
                current_state_feat_policy_update = current_state_policy_update
                prev_state_feat_policy_update = prev_state
            mu = self.policy.forward(current_state_feat_policy_update, reuse=True)

            if self.env.continuous_actions:
                eta = self.env.sigma * tf.random_normal(shape=tf.shape(mu))
                action = mu + eta

                if self.env.use_airl:
                    mean, logstd = mu, tf.log(tf.ones_like(mu) * self.env.sigma)
                    std = tf.exp(logstd)

                    n_elts = tf.cast(tf.reduce_prod(mean.shape[1:]), tf.float32)  # first dimension is batch size
                    log_normalizer = n_elts / 2. * (np.log(2 * np.pi).astype(np.float32)) + 1 / 2 * tf.reduce_sum(logstd,
                                                                                                                  axis=1)
                    # Diagonal Gaussian action probability, for every action
                    action_logprob = -tf.reduce_sum(tf.square(action - mean) / (2 * std), axis=1) - log_normalizer
            else:
                action = common.gumbel_softmax_sample(logits=mu, temperature=self.temp)

                if self.env.use_airl:
                    # Override since the implementation of tfp.RelaxedOneHotCategorical
                    # yields positive values.
                    if action.shape[1:] != logits.shape[1:]:
                        actions = tf.cast(action, tf.int8)
                        values = tf.one_hot(
                            actions, logits.shape.as_list()[-1], dtype=tf.float32)
                        assert values.shape == logits.shape, (values.shape, logits.shape)
                    else:
                        values = action

                    # [0]'s implementation (see line below) seems to be an approximation
                    # to the actual Gumbel Softmax density.
                    # TODO: to confirm 'action' or 'value'
                    action_logprob = -tf.reduce_sum(
                        -values * tf.nn.log_softmax(logits, axis=-1), axis=-1)

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            if self.env.use_airl:
                d, shaped_reward_output, reward = self.discriminator.forward(state=current_state_feat_policy_update, action=action, prev_state=prev_state_feat_policy_update, done_inp=tf.cast(env_term_sig, tf.float32), log_policy_act_prob=action_logprob, reuse=True)
                if self.env.alg in ['mairlTransfer', 'mairlImit4Transfer']:
                    reward_for_updating_policy = reward
                else:  # 'mairlImit'
                    reward_for_updating_policy = shaped_reward_output
                if self.env.train_mode and not self.env.alg in ['mairlTransfer', 'mairlImit4Transfer']:
                    ent_bonus = - self.env.airl_entropy_weight * tf.stop_gradient(action_logprob)
                    policy_reward = reward_for_updating_policy + ent_bonus
                else:
                    policy_reward = reward_for_updating_policy
                cost = tf.reduce_mean(-policy_reward) * self.env.policy_al_w
            else:
                d, _, _ = self.discriminator.forward(state=current_state_feat_policy_update, action=action, reuse=True)
                cost = self.al_loss(d)

            # add step cost
            total_cost += tf.multiply(tf.pow(self.gamma, t), cost)

            # get action
            if self.env.continuous_actions:
                a_sim = common.denormalize(action, self.er_expert.actions_mean, self.er_expert.actions_std)
            else:
                a_sim = tf.argmax(action, dimension=1)

            # get next state
            state_env, _, env_term_sig, = self.env.step(a_sim, mode='tensorflow')[:3]
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std)
            state_e = tf.stop_gradient(state_e)

            state_a, _, divergence_loss_a = self.forward_model.forward([current_state_feat_policy_update, action, initial_gru_state], reuse=True)
            if self.env.obs_mode == 'pixel':
                state_a = ops.decoder(state_a, data_shape=self.env.state_size, reuse=True)
            if True:  # self.env.alg in ['mgail']:
                state, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
            else:
                _, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
                state = state_a

            total_trans_err += tf.reduce_mean(abs(nu))
            t += 1

            if self.env.obs_mode == 'pixel':
                state = tf.slice(state, [0, 0, 0, 0], [1, -1, -1, -1])
            return state, t, total_cost, total_trans_err, env_term_sig, current_state_policy_update

        def policy_stop_condition(current_state_policy_update, t, cost, trans_err, env_term_sig, prev_state):
            cond = tf.logical_not(env_term_sig)  # not done: env_term_sig = False
            cond = tf.logical_and(cond, t < self.env.n_steps_train)
            cond = tf.logical_and(cond, trans_err < self.env.total_trans_err_allowed)
            return cond
        if self.env.obs_mode == 'pixel':
            state_0 = tf.slice(current_states, [0, 0, 0, 0], [1, -1, -1, -1])
        else:
            state_0 = tf.slice(current_states, [0, 0], [1, -1])
        # prev_state_0 = tf.slice(states_, [0, 0], [1, -1])
        loop_outputs = tf.while_loop(policy_stop_condition, policy_loop, [state_0, 0., 0., 0., False, state_0])
        self.policy.train(objective=loop_outputs[2])

    def al_loss(self, d):
        logit_agent, logit_expert = tf.split(axis=1, num_or_size_splits=2, value=d)

        # Cross entropy loss
        labels = tf.concat([tf.zeros_like(logit_agent), tf.ones_like(logit_expert)], 1)
        d_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=d, labels=labels)
        loss = tf.reduce_mean(d_cross_entropy)

        return loss*self.env.policy_al_w
