import pickle as cPickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def save_params(fname, saver, session):
    saver.save(session, fname)


def load_er(fname, batch_size, history_length, traj_length):
    f = open(fname, 'rb')
    er = cPickle.load(f, encoding='latin1')
    er.batch_size = batch_size
    er = set_er_stats(er, history_length, traj_length)
    return er


def set_er_stats(er, history_length, traj_length):
    state_dim = er.states.shape[1:]
    action_dim = er.actions.shape[-1]
    er.prestates = np.empty((er.batch_size, history_length,) + state_dim, dtype=np.float32)
    er.poststates = np.empty((er.batch_size, history_length,) + state_dim, dtype=np.float32)
    er.traj_states = np.empty((er.batch_size, traj_length,) + state_dim, dtype=np.float32)
    er.traj_actions = np.empty((er.batch_size, traj_length-1, action_dim), dtype=np.float32)
    er.states_min = np.min(er.states[:er.count], axis=0)
    er.states_max = np.max(er.states[:er.count], axis=0)
    er.actions_min = np.min(er.actions[:er.count], axis=0)
    er.actions_max = np.max(er.actions[:er.count], axis=0)
    er.states_mean = np.mean(er.states[:er.count], axis=0)
    er.actions_mean = np.mean(er.actions[:er.count], axis=0)
    er.states_std = np.std(er.states[:er.count], axis=0)
    er.states_std[er.states_std == 0] = 1
    er.actions_std = np.std(er.actions[:er.count], axis=0)
    return er


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu


def normalize(x, mean, std):
    return (x - mean)/std


def denormalize(x, mean, std):
    return x * std + mean


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def one_hot(a, num_classes):
    """
     if you have a vector with shape of (10000,) this function transforms it to (10000,C)
     Note that a is zero-indexed, i.e. one_hot(np.array([0, 1]), 2) will give [[1, 0], [0, 1]]
    :param a:
    :param num_classes: number of classes you have
    :return:
    """
    a = a.astype(np.int64)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def heatmap2d(hm_mat, title='', new_fig=True, block=True, fig_num=1, text=False, save_path=None, env_name=None):
    """
    Display heatmap
    input:
      hm_mat:   mxn 2d np array
    """
    print('map shape: {}, data type: {}'.format(hm_mat.shape, hm_mat.dtype))

    if block or new_fig:
        plt.figure(fig_num)
        plt.clf()

    # plt.imshow(hm_mat, cmap='hot', interpolation='nearest')
    plt.imshow(hm_mat, interpolation='nearest')
    plt.title(title)
    plt.colorbar()

    if text:
        for y in range(hm_mat.shape[0]):
            for x in range(hm_mat.shape[1]):
                plt.text(x, y, '%.1f' % hm_mat[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
    if env_name in ['PointMazeRight-v0']:
        grid_size = 0.005
        rescale = 1. / grid_size
        boundary_low = -0.1
        barrier_range = [0.25, 0.57]
        barrier_y = 0.25

        plt.scatter((0.25 - boundary_low) * rescale, (0.5 - boundary_low) * rescale,
                   marker='*', s=150, c='r', edgecolors='k', linewidths=0.5)
        # plt.scatter((0.25 - boundary_low + np.random.uniform(low=-0.05, high=0.05)) * rescale,
        #            (0. - boundary_low + np.random.uniform(low=-0.05, high=0.05)) * rescale, marker='o', s=120,
        #            c='white', linewidths=0.5, edgecolors='k')
        plt.scatter((0.25 - boundary_low) * rescale,
                    (0. - boundary_low) * rescale, marker='o', s=120,
                    c='white', linewidths=0.5, edgecolors='k')
        plt.plot([(barrier_range[0] - boundary_low) * rescale, (barrier_range[1] - boundary_low) * rescale],
                [(barrier_y - boundary_low) * rescale, (barrier_y - boundary_low) * rescale],
                color='k', linewidth=10)

    if block:
        plt.ion()
        print('press enter to continue')
        plt.show()
        input()
    else:
        plt.savefig(save_path)