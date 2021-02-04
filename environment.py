import tensorflow as tf
import numpy as np
import gym, cv2
import time, os
os.environ['LANG']='en_US'  # For Pendulum-v0 UTF-8 error
# from envs.examples import airl_envs
try:
    from senseact.envs.ur.reacher_env import ReacherEnv
    from senseact.utils import tf_set_seeds, NormalizedEnv
except:
    print('No senseact package install')
import random
from gym import spaces

class Environment(object):
    def __init__(self, run_dir, env_name, alg='mairlImit', train_mode=False, obs_mode='pixel'):
        """

        :param run_dir:
        :param env_name:
        :param alg: 'mairlImit', 'mairlImit4Transfer', 'mairlTransfer', 'mgail'
        :param obs_mode: 'pixel', 'state'
        """
        self.run_dir = run_dir
        self.name = env_name
        self.alg = alg
        self.obs_mode = obs_mode
        assert self.alg in ['mairlImit', 'mairlImit4Transfer', 'mairlTransfer', 'mgail'], '{} is not Implemented!'.format(self.alg)
        self.train_mode = train_mode
        if env_name in ['UR5_Reacher']:
            rand_state = np.random.RandomState(1).get_state()
            env = ReacherEnv(
                setup="UR5_6dof",
                host="192.168.1.102",
                dof=6,
                control_type="velocity",
                target_type="position",
                reset_type="zero",
                reward_type="precision",
                derivative_type="none",
                deriv_action_max=5,
                first_deriv_max=2,
                accel_max=1.4,
                speed_max=0.3,
                speedj_a=1.4,
                episode_length_time=4.0,
                episode_length_step=None,
                actuation_sync_period=1,
                dt=0.04,
                run_mode="multiprocess",
                rllab_box=False,
                movej_t=2.0,
                delay=0.0,
                random_state=rand_state
            )
            self.gym = NormalizedEnv(env)
            self.gym.start()
        else:
            self.gym = gym.make(self.name)
        self.random_initialization = True
        self._connect()
        self._train_params()
        self.set_seed()

    def _step(self, action):
        action = np.squeeze(action)
        if action.shape == ():
            action = np.expand_dims(action, axis=0)
            # or use:  action = 【action]
        self.t += 1
        if isinstance(self.gym.action_space, spaces.Discrete):
            action = int(action)
        result = self.gym.step(action)
        self.state, self.reward, self.done, self.info = result[:4]
        if self.obs_mode == 'pixel':
            self.state = cv2.resize(self.gym.render('rgb_array'), dsize=(64, 64), interpolation=cv2.INTER_AREA)
        if self.random_initialization:
            if hasattr(self.gym, 'env') and hasattr(self.gym.env, 'data'):
                self.qpos, self.qvel = self.gym.env.data.qpos.flatten(), self.gym.env.data.qvel.flatten()
            else:
                self.qpos, self.qvel = [], []
            return np.float32(self.state), np.float32(self.reward), self.done, np.float32(self.qpos), np.float32(self.qvel)
        else:
            return np.float32(self.state), np.float32(self.reward), self.done

    def step(self, action, mode):
        qvel, qpos = [], []
        if mode == 'tensorflow':
            if self.random_initialization:
                state, reward, done, qval, qpos = tf.py_func(self._step, inp=[action], Tout=[tf.float32, tf.float32, tf.bool, tf.float32, tf.float32], name='env_step_func')
            else:
                state, reward, done = tf.py_func(self._step, inp=[action],
                                                 Tout=[tf.float32, tf.float32, tf.bool],
                                                 name='env_step_func')

            state = tf.reshape(state, shape=self.state_size)
            done.set_shape(())
        else:
            if self.random_initialization:
                state, reward, done, qvel, qpos = self._step(action)
            else:
                state, reward, done = self._step(action)

        return state, reward, done, 0., qvel, qpos

    def reset(self, qpos=None, qvel=None):
        self.t = 0
        self.state = self.gym.reset()
        if self.obs_mode == 'pixel':
            self.state = cv2.resize(self.gym.render('rgb_array'), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        if self.random_initialization and qpos is not None and qvel is not None and hasattr(self.gym, 'env') and hasattr(self.gym.env, 'set_state'):
            self.gym.env.set_state(qpos, qvel)
        return np.float32(self.state)

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def render(self, mode='human'):
        img = self.gym.render(mode=mode)
        return img

    def _connect(self):
        if self.obs_mode == 'pixel':
            self.state_size = (64, 64, 3)
        else:
            if isinstance(self.gym.observation_space, spaces.Box):
                self.state_size = self.gym.observation_space.shape
            else:
                self.state_size = (self.gym.observation_space.n,)
        if isinstance(self.gym.action_space, spaces.Box):
            self.action_size = self.gym.action_space.shape[0]
        else:
            self.action_size = self.gym.action_space.n
        self.action_space = np.asarray([None]*self.action_size)
        if hasattr(self.gym, 'env') and hasattr(self.gym.env, 'data'):
            self.qpos_size = self.gym.env.data.qpos.shape[0]
            self.qvel_size = self.gym.env.data.qvel.shape[0]
        else:
            self.qpos_size = 0
            self.qvel_size = 0

    def set_seed(self):
        tf.set_random_seed(self.seed)
        random.seed(self.seed)
        self.gym.seed(self.seed)
        np.random.seed(self.seed)

    def _train_params(self):
        self.seed = 0
        if self.name == 'Hopper-v2':
            self.expert_data = 'expert_trajectories/hopper_er.bin'
        elif self.name in ['Ant-v2', 'CartPole-v0', 'GridWorldGym-v0', 'HalfCheetah-v2', 'Swimmer-v2', 'Pendulum-v0']:
            self.expert_data = 'expert_data/{}_expert_{}.bin'.format(self.obs_mode, self.name)
        elif self.name == 'PointMazeRight-v0':
            self.expert_data = 'expert_data/{}_expert_{}.bin'.format(self.obs_mode, 'PointMazeLeft-v0')
        elif self.name == 'DisabledAnt-v0':
            self.expert_data = 'expert_data/{}_expert_{}.bin'.format(self.obs_mode, 'CustomAnt-v0')
        elif self.name in ['PointMazeLeft-v0', 'CustomAnt-v0']:
            self.expert_data = 'packages/gail_expert/{}_expert_{}.bin'.format(self.obs_mode, self.name)
        elif self.name in ['UR5_Reacher']:
            self.expert_data = 'packages/gail_expert/{}_expert_{}.bin'.format(self.obs_mode, self.name)
        else:
            raise NotImplementedError('Env {} is not implemented.'.format(self.name))

        if not self.train_mode:
            self.trained_model = 'snapshots/20200705225434_Ant-v2_train_mairlImit_s_100/2020-07-06-07-20-175000.sn'
            # Test episode number: self.n_train_iters / self.test_interval * self.n_episodes_test
            self.n_train_iters = 1
            self.test_interval = 1
            self.n_episodes_test = 10
        else:
            if self.alg == 'mairlTransfer':
                self.trained_model = 'snapshots/20200804190406_PointMazeLeft-v0_train_mairlImit4Transfer_s_10_False_False_False/2020-08-05-11-01-720000.sn'
            else:
                self.trained_model = None
            self.n_train_iters = 1000000
            self.test_interval = 1000
            self.n_episodes_test = 1

        if self.name in ['GridWorldGym-v0']:
            self.n_steps_test = self.gym.spec.max_episode_steps  # 20
        else:
            self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = True
        if self.name in ['GridWorldGym-v0', 'MountainCar-v0', 'CartPole-v0']:
            self.continuous_actions = False
        else:
            self.continuous_actions = True
        self.airl_entropy_weight = 1.0
        if self.alg in ['mairlImit4Transfer', 'mairlTransfer']:
            self.use_airl = True
            self.disc_out_dim = 1
            self.phi_size = None  # [200, 100]
            self.forward_model_type = 'gru'
            self.state_only = True  # False
        elif self.alg in ['mairlImit']:
            self.use_airl = True
            self.disc_out_dim = 1
            self.phi_size = None  # [200, 100]
            self.forward_model_type = 'transformer'  # 'transformer'  # 'gru'
            self.state_only = False
        else:
            self.use_airl = False
            self.disc_out_dim = 2
            self.phi_size = None  # [200, 100]
            self.forward_model_type = 'gru'
            self.state_only = False

        # Main parameters to play with:
        self.er_agent_size = 50000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        if self.state_only:
            if self.name in ['PointMazeLeft-v0', 'CustomAnt-v0']:
                self.discr_policy_itrvl = 10
            else:
                self.discr_policy_itrvl = 100
            self.prep_time = 0
            self.save_best_ckpt = False
        else:
            self.discr_policy_itrvl = 100
            self.prep_time = 1000
            self.save_best_ckpt = True
        if self.forward_model_type == 'transformer':
            self.use_scale_dot_product = True
            self.use_skip_connection = True
            self.use_dropout = False
        else:
            self.use_scale_dot_product = False
            self.use_skip_connection = False
            self.use_dropout = False
        self.gamma = 0.99
        self.batch_size = 512  # 70
        self.weight_decay = 1e-7
        self.policy_al_w = 1e-2
        self.policy_tr_w = 1e-4
        self.policy_accum_steps = 7
        self.total_trans_err_allowed = 1000
        self.temp = 1.
        self.cost_sensitive_weight = 0.8
        self.noise_intensity = 6.
        self.do_keep_prob = 0.75
        self.forward_model_lambda = 0.  # 0.1

        # Hidden layers size
        self.fm_size = 100
        self.d_size = [200, 100]
        self.p_size = [100, 50]
        self.encoder_feat_size = 1024  # (30,)

        # Learning rates
        self.fm_lr = 1e-4
        self.d_lr = 1e-3
        self.p_lr = 1e-4

        # Log
        self.exp_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()), self.name,
                                                'train' if self.train_mode else 'eval', self.alg,
                                                's' if self.state_only else 'sa', self.discr_policy_itrvl,
                                                self.use_scale_dot_product, self.use_skip_connection, self.use_dropout)
        self.config_dir = os.path.join(self.run_dir, 'snapshots', self.exp_name)
        self.log_intervel = 100
        self.save_video = True

        if not os.path.isdir(self.config_dir):
            os.makedirs(self.config_dir)

        with open(os.path.join(self.config_dir, 'log.txt'), 'a') as f:
            f.write("{0}: {1}\n".format('seed', self.seed))
            f.write("{0}: {1}\n".format('name', self.name))
            f.write("{0}: {1}\n".format('expert_data', self.expert_data))
            f.write("{0}: {1}\n".format('train_mode', self.train_mode))
            f.write("{0}: {1}\n".format('trained_model', self.trained_model))
            f.write("{0}: {1}\n".format('n_train_iters', self.n_train_iters))
            f.write("{0}: {1}\n".format('test_interval', self.test_interval))
            f.write("{0}: {1}\n".format('n_episodes_test', self.n_episodes_test))
            f.write("{0}: {1}\n".format('alg', self.alg))
            f.write("{0}: {1}\n".format('n_steps_test', self.n_steps_test))
            f.write("{0}: {1}\n".format('vis_flag', self.vis_flag))
            f.write("{0}: {1}\n".format('save_models', self.save_models))

            f.write("{0}: {1}\n".format('continuous_actions', self.continuous_actions))
            f.write("{0}: {1}\n".format('airl_entropy_weight', self.airl_entropy_weight))
            f.write("{0}: {1}\n".format('use_airl', self.use_airl))
            f.write("{0}: {1}\n".format('disc_out_dim', self.disc_out_dim))
            f.write("{0}: {1}\n".format('phi_size', self.phi_size))
            f.write("{0}: {1}\n".format('forward_model_type', self.forward_model_type))
            f.write("{0}: {1}\n".format('state_only', self.state_only))
            f.write("{0}: {1}\n".format('er_agent_size', self.er_agent_size))
            f.write("{0}: {1}\n".format('collect_experience_interval', self.collect_experience_interval))
            f.write("{0}: {1}\n".format('n_steps_train', self.n_steps_train))
            f.write("{0}: {1}\n".format('discr_policy_itrvl', self.discr_policy_itrvl))
            f.write("{0}: {1}\n".format('prep_time', self.prep_time))

            f.write("{0}: {1}\n".format('gamma', self.gamma))
            f.write("{0}: {1}\n".format('batch_size', self.batch_size))
            f.write("{0}: {1}\n".format('weight_decay', self.weight_decay))
            f.write("{0}: {1}\n".format('policy_al_w', self.policy_al_w))
            f.write("{0}: {1}\n".format('policy_tr_w', self.policy_tr_w))
            f.write("{0}: {1}\n".format('policy_accum_steps', self.policy_accum_steps))
            f.write("{0}: {1}\n".format('total_trans_err_allowed', self.total_trans_err_allowed))
            f.write("{0}: {1}\n".format('temp', self.temp))
            f.write("{0}: {1}\n".format('cost_sensitive_weight', self.cost_sensitive_weight))
            f.write("{0}: {1}\n".format('noise_intensity', self.noise_intensity))
            f.write("{0}: {1}\n".format('do_keep_prob', self.do_keep_prob))
            f.write("{0}: {1}\n".format('forward_model_lambda', self.forward_model_lambda))

            f.write("{0}: {1}\n".format('fm_size', self.fm_size))
            f.write("{0}: {1}\n".format('d_size', self.d_size))
            f.write("{0}: {1}\n".format('p_size', self.p_size))
            f.write("{0}: {1}\n".format('fm_lr', self.fm_lr))
            f.write("{0}: {1}\n".format('d_lr', self.d_lr))
            f.write("{0}: {1}\n".format('p_lr', self.p_lr))
            f.write("{0}: {1}\n".format('exp_name', self.exp_name))
            f.write("{0}: {1}\n".format('config_dir', self.config_dir))
            f.write("{0}: {1}\n".format('log_intervel', self.log_intervel))
            f.write("{0}: {1}\n".format('save_video', self.save_video))
            f.write("{0}: {1}\n".format('save_best_ckpt', self.save_best_ckpt))
            f.write("{0}: {1}\n".format('obs_mode', self.obs_mode))
            f.write("{0}: {1}\n".format('use_scale_dot_product', self.use_scale_dot_product))
            f.write("{0}: {1}\n".format('use_skip_connection', self.use_skip_connection))
            f.write("{0}: {1}\n".format('use_dropout', self.use_dropout))



