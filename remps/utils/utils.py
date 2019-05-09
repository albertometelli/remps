import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name, action="store_true", default=default, dest=dest, help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def get_tf_optimizer(optimizer_name, lrate=0.000001):

    if optimizer_name == "gd":
        return tf.train.GradientDescentOptimizer(lrate)
    elif optimizer_name == "adam":
        return tf.train.AdamOptimizer(learning_rate=0.99, beta1=0.9, beta2=0.99)
    elif optimizer_name == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate=0.0001, rho=0.2, epsilon=1e-12)
    elif optimizer_name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.7)
    return tf.train.GradientDescentOptimizer(0.0001)


# plot a gaussian process
def plot(X, mean, var, title="plot"):
    plt.figure(figsize=(12, 6))
    # plt.plot(mean[:,0], 'ro')
    plt.errorbar(X, mean, var, linestyle="None", marker="^", ecolor="g", color="b")
    plt.title(title)
    plt.show()


def make_session(num_cpu=None, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))

    print("Creating session with num_cpu:", num_cpu)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu, intra_op_parallelism_threads=num_cpu
    )
    tf_config.gpu_options.allocator_type = "BFC"
    return tf.Session(config=tf_config, graph=graph)


def get_default_tf_dtype():
    return tf.float64


def flat_and_pad(lol):
    """
    :param lol: list of lists with variable lengths
    :return: a vector containing elements from lol padded with same lengths
    """
    maxLength = np.max([len(l) for l in lol])
    if np.isscalar(lol[0][0]):
        dim = 1
    else:
        dim = lol[0][0].shape[0]

    # 1st dimension: # episodes
    # 2nd dimension: # timestep
    # 3rd dimension: characteristic dimension
    m = np.zeros((len(lol), maxLength, dim))

    for (i, l) in enumerate(lol):
        m[i, 0 : len(l)] = np.reshape(l, (-1, dim))
    return m.reshape((-1, dim))
