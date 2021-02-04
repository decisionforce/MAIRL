import sys, os, cv2
import time
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import logging

tf.get_logger().setLevel(logging.ERROR)
import common
from mgail import MGAIL
from tensorboardX import SummaryWriter
import moviepy.editor as mpy


class Driver(object):
    def __init__(self, environment):
        self.env = environment
        self.algorithm = MGAIL(environment=self.env)
        self.init_graph = tf.global_variables_initializer()
        if self.env.alg == 'mairlTransfer':
            variables_to_restore = [var for var in tf.global_variables()
                                    if var.name.startswith('discriminator')]
            # print('variables_to_restore: ', variables_to_restore)
            self.restore_disc = tf.train.Saver(variables_to_restore)
        self.saver = tf.train.Saver(max_to_keep=None)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        # Prevent tensorflow from taking all the gpu memory
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        if self.env.trained_model:
            if self.env.train_mode and self.env.alg == 'mairlTransfer':
                # initialize other parameters
                self.sess.run(self.init_graph)
                self.restore_disc.restore(self.sess, self.env.trained_model)
                print('(mairlTransfer) Restore {} successfully.'.format(self.env.trained_model))
            else:
                self.saver.restore(self.sess, self.env.trained_model)
                print('(Eval) Restore {} successfully.'.format(self.env.trained_model))
        else:
            self.sess.run(self.init_graph)
        self.sess.graph.finalize()
        self.run_dir = self.env.run_dir
        self.loss = 999. * np.ones(3)
        self.reward_mean = 0
        self.reward_std = 0
        self.run_avg = 0.001
        self.discriminator_policy_switch = 0
        self.policy_loop_time = 0
        self.disc_acc = 0
        self.er_count = 0
        self.itr = 0
        self.best_reward = 0
        self.mode = 'Prep'
        self.writer = SummaryWriter(log_dir=self.env.config_dir)
        np.set_printoptions(precision=2)
        np.set_printoptions(linewidth=220)

        self.video_index = 0

    def update_stats(self, module, attr, value):
        v = {'forward_model': 0, 'discriminator': 1, 'policy': 2}
        module_ind = v[module]
        if attr == 'loss':
            self.loss[module_ind] = self.run_avg * self.loss[module_ind] + (1 - self.run_avg) * np.asarray(value)
        elif attr == 'accuracy':
            self.disc_acc = self.run_avg * self.disc_acc + (1 - self.run_avg) * np.asarray(value)

    def train_forward_model(self):
        alg = self.algorithm
        states_, actions, _, states = self.algorithm.er_agent.sample()[:4]
        fetches = [alg.forward_model.minimize, alg.forward_model.loss]
        feed_dict = {alg.states_: states_, alg.states: states, alg.actions: actions,
                     alg.do_keep_prob: self.env.do_keep_prob}
        run_vals = self.sess.run(fetches, feed_dict)
        self.update_stats('forward_model', 'loss', run_vals[1])
        if self.itr % self.env.discr_policy_itrvl == 0:
            self.writer.add_scalar('train/forward_model/loss', run_vals[1], self.itr)

    def train_discriminator(self):
        alg = self.algorithm
        # get states and actions
        state_a, action_a, rewards_a, state_a_, terminals_a = self.algorithm.er_agent.sample()[:5]
        state_e, action_e, rewards_e, state_e_, terminals_e = self.algorithm.er_expert.sample()[:5]
        states = np.concatenate([state_a, state_e])
        dones = np.concatenate([terminals_a, terminals_e])
        if not self.env.continuous_actions:
            action_e = common.one_hot(action_e, num_classes=self.env.action_size)
        actions = np.concatenate([action_a, action_e])
        # labels (policy/expert) : 0/1, and in 1-hot form: policy-[1,0], expert-[0,1]
        labels_a = np.zeros(shape=(state_a.shape[0],))
        labels_e = np.ones(shape=(state_e.shape[0],))
        labels = np.expand_dims(np.concatenate([labels_a, labels_e]), axis=1)
        fetches = [alg.discriminator.minimize, alg.discriminator.loss, alg.discriminator.acc]
        if self.env.use_airl:
            states_ = np.concatenate([state_a_, state_e_])
            feed_dict = {alg.states: states, alg.states_: states_, alg.done_ph: dones, alg.actions: actions,
                         alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
        else:
            feed_dict = {alg.states: states, alg.actions: actions,
                         alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
        run_vals = self.sess.run(fetches, feed_dict)
        self.update_stats('discriminator', 'loss', run_vals[1])
        self.update_stats('discriminator', 'accuracy', run_vals[2])
        if self.itr % self.env.discr_policy_itrvl == 0:
            self.writer.add_scalar('train/discriminator/loss', run_vals[1], self.itr)
            self.writer.add_scalar('train/discriminator/accuracy', run_vals[2], self.itr)

    def get_recovered_reward(self, print_info=False):
        alg = self.algorithm
        # labels (policy/expert) : 0/1, and in 1-hot form: policy-[1,0], expert-[0,1]
        if self.env.name in ['GridWorldGym-v0']:
            # states = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            #                   [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
            #                   [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
            #                   [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            #                   [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
            #                   ])
            # x-axis: Horizontal, y-axis: Vertical
            obs_batch = []
            num_y = 0
            for pos_y in range(5):
                num_y += 1
                num_x = 0
                for pos_x in range(5):
                    num_x += 1
                    obs_batch.append([pos_x, pos_y])
            states = np.array(obs_batch)
            labels_a = np.zeros(shape=(states.shape[0],))
            labels = np.expand_dims(np.concatenate([labels_a]), axis=1)

            fetches = [alg.disc_reward]
            if self.env.use_airl:
                feed_dict = {alg.states: states,
                             alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
            else:
                feed_dict = {alg.states: states,
                             alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
            run_vals = self.sess.run(fetches, feed_dict)
            recovered_reward = run_vals[0].reshape((num_x, num_y))
            text = False   # True
        elif self.env.name in ['PointMazeLeft-v0', 'PointMazeRight-v0']:
            boundary_low = -0.1
            boundary_high = 0.6
            grid_size = 0.005

            obs_batch = []
            num_y = 0
            for pos_y in np.arange(boundary_low, boundary_high, grid_size):
                num_y += 1
                num_x = 0
                for pos_x in np.arange(boundary_low, boundary_high, grid_size):
                    num_x += 1
                    obs_batch.append([pos_x, pos_y, 0.])
            states = np.array(obs_batch)
            labels_a = np.zeros(shape=(states.shape[0],))
            labels = np.expand_dims(np.concatenate([labels_a]), axis=1)

            fetches = [alg.disc_reward]
            if self.env.use_airl:
                feed_dict = {alg.states: states,
                             alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
            else:
                feed_dict = {alg.states: states,
                             alg.label: labels, alg.do_keep_prob: self.env.do_keep_prob}
            run_vals = self.sess.run(fetches, feed_dict)
            recovered_reward = run_vals[0].reshape([num_x, num_y])
            text = False
        else:
            raise NotImplementedError('Env {} is not implemented for recovered_reward()'.format(self.env.name))

        if print_info:
            print('Recovered reward: {}'.format(recovered_reward))
        common.heatmap2d(hm_mat=recovered_reward, block=False, text=text,
                         save_path='{}'.format(
                             os.path.join(self.env.config_dir, 'recovered_reward_{}.png'.format(self.itr))), env_name=self.env.name)
        hm_mat_val_normed = (recovered_reward - np.min(recovered_reward)) / (np.max(recovered_reward) - np.min(recovered_reward))
        common.heatmap2d(hm_mat=hm_mat_val_normed, block=False, text=text,
                         save_path='{}'.format(os.path.join(self.env.config_dir, 'recovered_reward_normed_{}.png'.format(self.itr))), env_name=self.env.name)
        return run_vals[0]

    def visulize_forward_model(self):
        alg = self.algorithm
        qposs, qvels = alg.er_expert.sample(indexes=[1])[5:]
        env_observation = self.env.reset(qpos=qposs[0], qvel=qvels[0])
        noise_flag = False
        do_keep_prob = 1.
        env_next_observation_list = [env_observation]
        pred_next_observation_list = [env_observation]
        for i in range(19):
            policy_actions = self.sess.run(fetches=[alg.action_test],
                              feed_dict={alg.states: np.reshape(env_observation, (1,) + self.env.state_size),
                                         alg.do_keep_prob: do_keep_prob,
                                         alg.noise: noise_flag,
                                         alg.temp: self.env.temp})

            feed_dict = {alg.states_: np.array([env_observation]), alg.actions: policy_actions,
                         alg.do_keep_prob: self.env.do_keep_prob}
            pred_next_observation = self.sess.run(alg.forward_model_prediction, feed_dict)

            env_observation, reward, done, info, qpos, qvel = self.env.step(policy_actions, mode='python')

            env_next_observation_list.append(env_observation)
            pred_next_observation_list.append(np.squeeze(pred_next_observation, axis=0).copy())
        env_next_observation_list_np = np.hstack(env_next_observation_list)
        pred_next_observation_list_np = np.hstack(pred_next_observation_list)
        error = ((pred_next_observation_list_np - env_next_observation_list_np) + 255) / 2
        observation_img = np.vstack([env_next_observation_list_np, pred_next_observation_list_np, error])

        save_path = '{}'.format(
            os.path.join(self.env.config_dir, 'visualize_fm_{}.png'.format(self.itr)))
        cv2.imwrite(save_path, cv2.cvtColor(observation_img, cv2.COLOR_RGB2BGR))

    def train_policy(self):
        alg = self.algorithm

        # reset the policy gradient
        self.sess.run([alg.policy.reset_grad_op], {})

        # Adversarial Learning
        if self.env.get_status():
            state = self.env.reset()
        else:
            state = self.env.get_state()

        # Accumulate the (noisy) adversarial gradient
        for i in range(self.env.policy_accum_steps):
            # accumulate AL gradient
            fetches = [alg.policy.accum_grads_al, alg.policy.loss_al]
            feed_dict = {alg.states: np.array([state]), alg.gamma: self.env.gamma,
                         alg.do_keep_prob: self.env.do_keep_prob, alg.noise: 1., alg.temp: self.env.temp}
            run_vals = self.sess.run(fetches, feed_dict)
            self.update_stats('policy', 'loss', run_vals[1])
            if i == self.env.policy_accum_steps - 1 and self.itr % self.env.discr_policy_itrvl == 0:
                self.writer.add_scalar('train/policy/loss', run_vals[1], self.itr)
        # apply AL gradient
        self.sess.run([alg.policy.apply_grads_al], {})

    def collect_experience(self, record=1, vis=0, n_steps=None, noise_flag=True, start_at_zero=True):
        alg = self.algorithm

        # environment initialization point
        if start_at_zero:
            observation = self.env.reset()
        else:
            qposs, qvels = alg.er_expert.sample()[5:]
            observation = self.env.reset(qpos=qposs[0], qvel=qvels[0])

        do_keep_prob = self.env.do_keep_prob
        t = 0
        R = 0
        done = 0
        frames = list()
        if n_steps is None:
            n_steps = self.env.n_steps_test

        while not done:
            if vis:
                self.env.render()
            if not self.env.train_mode and self.env.save_video:
                img = self.env.render(mode='rgb_array')
                assert img is not None, img
                frames.append(img)
            if not noise_flag:
                do_keep_prob = 1.

            a = self.sess.run(fetches=[alg.action_test], feed_dict={alg.states: np.reshape(observation, (1,) + self.env.state_size),
                                                                    alg.do_keep_prob: do_keep_prob,
                                                                    alg.noise: noise_flag,
                                                                    alg.temp: self.env.temp})

            observation, reward, done, info, qpos, qvel = self.env.step(a, mode='python')

            done = done or t > n_steps
            t += 1
            R += reward

            if record:
                if self.env.continuous_actions:
                    action = a
                else:
                    action = np.zeros((1, self.env.action_size))
                    action[0, a[0]] = 1
                alg.er_agent.add(actions=action, rewards=[reward], next_states=[observation], terminals=[done],
                                 qposs=[qpos], qvels=[qvel])
        if len(frames) > 0:
            if not os.path.isdir(self.env.config_dir):
                os.mkdir(self.env.config_dir)
            fps = 2 if self.env.name.startswith('GridWorld') else 64
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(os.path.join(self.env.config_dir, 'eval_{}_{:.2f}.mp4'.format(self.video_index, R)),
                                 fps=fps)
            self.video_index += 1
        return R

    def train_step(self):
        # phase_1 - Adversarial training
        # forward_model: learning from agent data
        # discriminator: learning in an interleaved mode with policy
        # policy: learning in adversarial mode

        # Fill Experience Buffer
        if self.itr == 0:
            while self.algorithm.er_agent.current == self.algorithm.er_agent.count:
                self.collect_experience()
                buf = 'Collecting examples...%d/%d' % \
                      (self.algorithm.er_agent.current, self.algorithm.er_agent.states.shape[0])
                sys.stdout.write('\r' + buf)

        # Adversarial Learning
        else:
            self.train_forward_model()

            self.mode = 'Prep'
            if self.itr < self.env.prep_time and self.env.alg != 'mairlTransfer':
                self.train_discriminator()
            else:
                self.mode = 'AL'

                if self.discriminator_policy_switch and self.env.alg != 'mairlTransfer': #  and (self.itr % self.env.discr_policy_itrvl < self.env.discr_policy_itrvl / 10):
                    self.train_discriminator()
                else:
                    self.train_policy()

                if self.itr % self.env.collect_experience_interval == 0:
                    # if self.env.name in ['Walker2d-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'HalfCheetah-v2', 'Pendulum-v0', 'Swimmer-v2']:
                    #     R = self.collect_experience(start_at_zero=False, n_steps=self.env.n_steps_train)
                    # else:
                    #     R = self.collect_experience(start_at_zero=True, n_steps=self.env.n_steps_train)
                    R = self.collect_experience(start_at_zero=False, n_steps=self.env.n_steps_train)
                    self.writer.add_scalar('train/reward_mean', R, self.itr)
                # switch discriminator-policy
                if self.itr % self.env.discr_policy_itrvl == 0:
                    self.discriminator_policy_switch = not self.discriminator_policy_switch

        # print progress
        if self.itr % 100 == 0:
            self.print_info_line('slim')

    def print_info_line(self, mode):
        if mode == 'full':
            buf = '%s Training(%s): iter %d, loss: %s R: %.1f, R_std: %.2f\n' % \
                  (time.strftime("%H:%M:%S"), self.mode, self.itr, self.loss, self.reward_mean, self.reward_std)
        else:
            buf = "processing iter: %d, loss(forward_model,discriminator,policy): %s" % (self.itr, self.loss)
        sys.stdout.write('\r' + buf)

    def save_model(self, dir_name=None, info=None):
        import os
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, 'snapshots')
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        fname = os.path.join(dir_name, time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % self.itr))
        common.save_params(fname=fname, saver=self.saver, session=self.sess)
        if info:
            with open(os.path.join(dir_name, 'log.txt'), 'a') as f:
                f.write("{0}: {1}\n".format(fname, info))
