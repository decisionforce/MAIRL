import numpy as np
import os, sys
from environment import Environment
from driver import Driver
import argparse

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}


def str2bool(value, raise_exc=False):
    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def dispatcher(env):

    driver = Driver(env)
    best_rew = -np.inf

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:
            # Visualize forward model
            if env.obs_mode in ['pixel']:
                driver.visulize_forward_model()
            # Visualize tabular reward
            if env.name in ['GridWorldGym-v0', 'PointMazeRight-v0', 'PointMazeLeft-v0'] and env.state_only:
                driver.get_recovered_reward(print_info=not env.train_mode)
            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)

            # print info line
            driver.print_info_line('full')
            driver.writer.add_scalar('eval/reward_mean', driver.reward_mean, driver.itr)
            driver.writer.add_scalar('eval/reward_std', driver.reward_std, driver.itr)

            # save snapshot
            if env.train_mode and env.save_models and (driver.reward_mean >= best_rew or not env.save_best_ckpt):
                best_rew = driver.reward_mean
                driver.save_model(dir_name=env.config_dir, info=driver.reward_mean)

        driver.itr += 1

    driver.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAIRL.')
    parser.add_argument('--env_name', metavar='E', type=str, default='Hopper-v2',
                        help='Environment Name')
    parser.add_argument('--alg', metavar='A', type=str, default='mairlImit',
                        help='algorithm (mairlImit / mairlImit4Transfer / mairlTransfer / mgail)')
    parser.add_argument('--obs_mode', metavar='O', type=str, default='state',
                        help='(state / pixel)')
    parser.add_argument('--train_mode', metavar='M', type=str2bool, default=True,
                        help='whether in the train mode (true) or eval mode (false)')
    args = parser.parse_args()

    # load environment
    env = Environment(run_dir=os.path.curdir, env_name=args.env_name, train_mode=args.train_mode,
                      alg=args.alg, obs_mode=args.obs_mode)

    # start training
    dispatcher(env=env)
