import os.path

import baselines.common.tf_util as U
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from tensorflow.contrib.distributions import Normal

from remps.model_approx.model_approximator import ModelApproximator
from remps.utils.utils import get_default_tf_dtype


def reduce_std(x, axis, dtype=tf.float32):
    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    var = tf.reduce_sum(tf.square(x - mean), axis=axis, keepdims=True) / (
        tf.cast(tf.shape(x)[0], dtype) - 1
    )
    return tf.sqrt(var)


class TorcsModel(ModelApproximator):
    def get_probability(self):
        return self.prob

    def get_omega(self):
        return self.omega

    def set_omega(self):
        pass

    def __init__(self, state_dim, action_dim, name="NN", training_set_size=4000):
        """
        Fit a NN for predicting the next state distribution
        Output of NN are the parameters of a parametric distribution (Gaussian)
        """
        self.sess = None
        self.log_prob = None
        self.prob = None
        self.dtype = get_default_tf_dtype()
        self.name = name
        self.x_range = 4.8
        self.theta_range = 180
        self.XData = None
        self.YData = None
        self.state_dim = state_dim
        self.x_dim = state_dim + 2 + action_dim
        self.action_dim = action_dim
        self.param_dim = 2
        self.gp_list = []
        self.training_set_size = training_set_size
        self.global_step = 0
        self.folder = "../" + self.name + "NNData" + "/"
        self.min_omega = 0
        self.max_omega = 1

    def __call__(
        self,
        states,
        actions,
        next_states,
        initial_omega=8,
        training_set_size=4000,
        actions_one_hot=None,
        sess=None,
        summary_writer=None,
    ):
        """
        :param states: Nxm matrix
        :param actions: Vector of all possible actions: Nx n_actions
        :param next_states: Nxm matrix containing the next states
        :param initial_omega: value of the initial omega
        :return:
        """
        self.sess = sess
        self.training_set_size = training_set_size
        self.summary_writer = summary_writer
        train_or_test = U.get_placeholder("train_or_test", tf.bool, ())
        # statistics
        self.Xmean_ph = U.get_placeholder(
            name="Xmean", dtype=self.dtype, shape=(1, self.x_dim)
        )
        self.Ymean_ph = U.get_placeholder(
            name="Ymean", dtype=self.dtype, shape=(1, self.state_dim)
        )
        self.Xstd_ph = U.get_placeholder(
            name="Xstd", dtype=self.dtype, shape=(1, self.x_dim)
        )
        self.Ystd_ph = U.get_placeholder(
            name="Ystd", dtype=self.dtype, shape=(1, self.state_dim)
        )
        self.X_train = U.get_placeholder(
            name="X_train", dtype=self.dtype, shape=(None, self.x_dim)
        )
        self.Y = U.get_placeholder(
            name="Y", dtype=self.dtype, shape=(None, self.state_dim)
        )
        vars_list = []
        with tf.variable_scope(self.name):
            # build the action vector
            self.omega = tf.get_variable(
                dtype=self.dtype,
                name="omega",
                shape=(1, self.param_dim),
                initializer=tf.initializers.constant(initial_omega),
            )
            Y = self.Y  # - YMean_) / Ystd_
            # build the action vector
            params = tf.tile(self.omega, (tf.shape(states)[0], 1))
            x_full = tf.concat([states, actions, params], axis=1)
            x_full = (x_full - self.Xmean_ph) / self.Xstd_ph
            next_states_full = next_states
            next_states_full = (next_states_full - self.Ymean_ph) / self.Ystd_ph

            x_input = U.switch(train_or_test, self.X_train, x_full)

            # build the network
            hidden_layer_size = 100
            h = tf.layers.dense(
                inputs=x_input, units=hidden_layer_size, activation=tf.nn.relu
            )

            # build the network
            hidden_layer_size_2 = 50
            h2 = tf.layers.dense(
                inputs=h, units=hidden_layer_size_2, activation=tf.nn.relu
            )

            means = tf.layers.dense(inputs=h2, units=self.state_dim)

            # variance part
            hidden_var = 50
            h = tf.layers.dense(inputs=x_input, units=hidden_var, activation=tf.nn.relu)
            h2 = tf.layers.dense(inputs=h, units=self.state_dim)
            var = tf.exp(h2)

            std = tf.sqrt(var)

            pdf = Normal(means, std)

            y_output = U.switch(train_or_test, Y, next_states_full)

            log_prob = tf.reduce_sum(pdf.log_prob(y_output), axis=1, keepdims=True)
            prob = tf.reduce_prod(pdf.prob(y_output), axis=1, keepdims=True)

            # loss is the negative loss likelihood
            self.loss = -tf.reduce_mean(log_prob)
            self.valid_loss = -tf.reduce_mean(log_prob)

            self.fitting_vars = [
                x
                for x in tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name
                )
                if "omega" not in x.name
            ]
            print(self.fitting_vars)
            # create fitting collection
            for v in self.fitting_vars:
                tf.add_to_collection("fitting", v)

            opt = tf.train.AdamOptimizer()
            self.minimize_op = opt.minimize(self.loss, var_list=self.fitting_vars)

            self.log_prob = log_prob
            self.prob = prob
            means_list = []
            variances_list = []
            for i in range(self.state_dim):
                means_ = tf.reshape(means[:, i], (-1, 1))
                means_list.append(means_)
                # same for variance
                var_ = tf.reshape(var[:, i], (-1, 1))
                variances_list.append(var_)

            self.means = tf.concat(means_list, axis=1)
            self.variances = tf.concat(variances_list, axis=1)
            self.train_or_test = train_or_test
            self.loss_summary = tf.summary.scalar("Loss", self.loss)
            self.valid_loss_summary = tf.summary.scalar("ValidLoss", self.valid_loss)
        return self.log_prob, self.prob

    def store_data(self, X, Y, normalize_data=False, update_statistics=True):
        """
        Store training data inside training set
        """
        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        np.save(self.folder + f"X{self.param_dim}{os.getpid()}.npy", X)
        np.save(self.folder + f"Y{self.param_dim}{os.getpid()}.npy", Y)
        np.save(
            self.folder + f"Xmean{self.param_dim}{os.getpid()}.npy",
            np.hstack(
                (
                    np.reshape(np.mean(X[:, : -self.param_dim], axis=0), (1, -1)),
                    np.zeros((1, self.param_dim)),
                )
            ),
        )
        np.save(
            self.folder + f"Xstd{self.param_dim}{os.getpid()}.npy",
            np.maximum(
                np.hstack(
                    (
                        np.reshape(np.std(X[:, : -self.param_dim], axis=0), (1, -1)),
                        np.ones((1, self.param_dim)),
                    )
                ),
                1e-10,
            ),
        )
        np.save(
            self.folder + f"Ymean{self.param_dim}{os.getpid()}.npy",
            np.reshape(np.mean(Y, axis=0), (1, -1)),
        )
        np.save(
            self.folder + f"Ystd{self.param_dim}{os.getpid()}.npy",
            np.maximum(np.reshape(np.std(Y, axis=0), (1, -1)), 1e-10),
        )

    def fit(
        self,
        load_from_file=False,
        save_to_file=True,
        action_ph=None,
        states_ph=None,
        next_states_ph=None,
        load_weights=True,
        add_onpolicy=False,
        training_step=5000,
        restart_fitting=False,
    ):

        training_set_size = 100000
        # X = np.load(self.folder + f"X{self.param_dim}{os.getpid()}.npy")
        # Y = np.load(self.folder + f"Y{self.param_dim}{os.getpid()}.npy")
        # self.Xmean = np.load(self.folder + f"Xmean{self.param_dim}{os.getpid()}.npy")
        # self.Xstd = np.load(self.folder + f"Xstd{self.param_dim}{os.getpid()}.npy")
        # self.Ymean = np.load(self.folder + f"Ymean{self.param_dim}{os.getpid()}.npy")
        # self.Ystd = np.load(self.folder + f"Ystd{self.param_dim}{os.getpid()}.npy")
        X = np.load(self.folder + f"X{self.param_dim}{os.getpid()}.npy")
        Y = np.load(self.folder + f"Y{self.param_dim}{os.getpid()}.npy")
        self.Xmean = np.load(self.folder + f"Xmean{self.param_dim}{os.getpid()}.npy")
        self.Xstd = np.load(self.folder + f"Xstd{self.param_dim}{os.getpid()}.npy")
        self.Ymean = np.load(self.folder + f"Ymean{self.param_dim}{os.getpid()}.npy")
        self.Ystd = np.load(self.folder + f"Ystd{self.param_dim}{os.getpid()}.npy")
        plot = False

        if add_onpolicy:
            X_onpolicy = np.load(
                self.folder + f"on_policyX{self.param_dim}{os.getpid()}.npy"
            )
            Y_onpolicy = np.load(
                self.folder + f"on_policyY{self.param_dim}{os.getpid()}.npy"
            )
            X = np.vstack((X, X_onpolicy))
            Y = np.vstack((Y, Y_onpolicy))

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # if not add_onpolicy:
        #     self.Xmean = np.hstack((np.reshape(np.mean(X_train[:,:-self.param_dim], axis=0), (1, -1)),np.zeros((1, self.param_dim))))
        #     self.Xstd = np.maximum(np.hstack((np.reshape(np.std(X_train[:,:-self.param_dim], axis=0), (1, -1)), np.ones((1, self.param_dim)))), 1e-10)
        #     self.Ymean = np.reshape(np.mean(y_train, axis=0), (1, -1))
        #     self.Ystd = np.maximum(np.reshape(np.std(y_test, axis=0), (1, -1)),1e-10)

        print(self.Xstd, self.Xmean)
        if not load_weights:
            X_train = (X_train - self.Xmean) / self.Xstd
            X_test = (X_test - self.Xmean) / self.Xstd
            y_train = (y_train - self.Ymean) / self.Ystd
            y_test = (y_test - self.Ymean) / self.Ystd
            batch_size = 1000
            for n in range(training_step):
                # sample a batch
                ind = np.arange(0, np.shape(X_train)[0])
                selected_ind = np.random.choice(ind, size=batch_size, replace=True)
                inputs = X_train[selected_ind, :]
                targets = y_train[selected_ind, :]
                feed_dict = {
                    self.train_or_test: True,
                    self.X_train: inputs,
                    self.Y: targets,
                    self.Xmean_ph: self.Xmean,
                    self.Ymean_ph: self.Ymean,
                    self.Xstd_ph: self.Xstd,
                    self.Ystd_ph: self.Ystd,
                    # dummy things
                    action_ph: [[0, 1]],
                    states_ph: np.ones((1, X.shape[1] - 2 - self.param_dim)),
                    next_states_ph: np.ones((1, Y.shape[1])),
                }
                # train
                loss, _, summary_str = self.sess.run(
                    [self.loss, self.minimize_op, self.loss_summary],
                    feed_dict=feed_dict,
                )
                self.summary_writer.add_summary(summary_str, self.global_step)

                # validation
                feed_dict = {
                    self.train_or_test: True,
                    self.X_train: X_test,
                    self.Y: y_test,
                    self.Xmean_ph: self.Xmean,
                    self.Ymean_ph: self.Ymean,
                    self.Xstd_ph: self.Xstd,
                    self.Ystd_ph: self.Ystd,
                    # dummy things
                    action_ph: [[0, 1]],
                    states_ph: np.ones((1, X.shape[1] - 2 - self.param_dim)),
                    next_states_ph: np.ones((1, Y.shape[1])),
                }
                valid_loss, loss_summary, probs = self.sess.run(
                    [self.valid_loss, self.valid_loss_summary, self.prob],
                    feed_dict=feed_dict,
                )
                self.summary_writer.add_summary(loss_summary, self.global_step)

                if n % 100 == 0:
                    print("step: ", n)
                    print("Loss: ", loss)
                    print("Validation Loss: ", valid_loss)

                # if n % 1000 == 0:
                #     # check Fitting
                #     means, variances = self.sess.run([self.means, self.variances], feed_dict=feed_dict)
                #     # subsampling
                #     ind = np.arange(0, np.shape(y_test)[0])
                #     selected_ind = np.random.choice(ind, size=100, replace=False)
                #     next_states_to_plot = y_test[selected_ind, :]
                #     means_to_plot = means[selected_ind, :]
                #     variances_to_plot = variances[selected_ind, :]
                #     xs = np.arange(0, np.shape(next_states_to_plot)[0])

                # for i in range(self.state_dim):
                #     plt.figure()
                #     plt.title("Prediction (blue) vs ground truth (red circles) " + str(i))
                #     plt.plot(xs, next_states_to_plot[:, i], 'ro')
                #     plt.errorbar(xs, means_to_plot[:, i], np.sqrt(variances_to_plot[:, i]), linestyle='None',
                #                  marker='^', ecolor='g',
                #                  color='b')
                #     plt.ylim(-1,1)
                #     plt.savefig(self.folder+"prediction_"+str(i)+"_step_"+str(n)+".png")
                #     print("saved")
                #     plt.close()

                self.global_step += 1

            print("FITTED!!")
            if not add_onpolicy:
                weights = U.GetFlat(self.fitting_vars)()
                np.save(
                    self.folder + f"weights{self.param_dim}{os.getpid()}.npy", weights
                )
        else:
            weights = np.load(self.folder + f"weights{self.param_dim}.npy")
            U.SetFromFlat(self.fitting_vars, dtype=self.dtype)(weights)

        if plot:
            # validation
            feed_dict = {
                self.train_or_test: True,
                self.X_train: X_test,
                self.Y: y_test,
                self.Xmean_ph: self.Xmean,
                self.Ymean_ph: self.Ymean,
                self.Xstd_ph: self.Xstd,
                self.Ystd_ph: self.Ystd,
                # dummy things
                action_ph: [[0, 1]],
                states_ph: np.ones((1, X.shape[1] - 2 - self.param_dim)),
                next_states_ph: np.ones((1, Y.shape[1])),
            }
            # check Fitting
            means, variances = self.sess.run(
                [self.means, self.variances], feed_dict=feed_dict
            )
            # subsampling
            ind = np.arange(0, np.shape(y_test)[0])
            selected_ind = np.random.choice(ind, size=100, replace=False)
            next_states_to_plot = y_test[selected_ind, :]
            means_to_plot = means[selected_ind, :]
            variances_to_plot = variances[selected_ind, :]
            xs = np.arange(0, np.shape(next_states_to_plot)[0])

            for i in range(self.state_dim):
                plt.figure()
                plt.title("Prediction (blue) vs ground truth (red circles) " + str(i))
                plt.plot(xs, next_states_to_plot[:, i], "ro")
                plt.errorbar(
                    xs,
                    means_to_plot[:, i],
                    np.sqrt(variances_to_plot[:, i]),
                    linestyle="None",
                    marker="^",
                    ecolor="g",
                    color="b",
                )
                plt.ylim(-1, 1)
                plt.savefig(self.folder + "final_prediction_" + str(i) + ".png")
                plt.close()

    def get_feed_dict(self):
        return {
            self.X_train: np.ones((1, self.Xmean.shape[1])),
            self.Y: np.ones((1, self.Ymean.shape[1])),
            self.Xmean_ph: self.Xmean,
            self.Xstd_ph: self.Xstd,
            self.Ymean_ph: self.Ymean,
            self.Ystd_ph: self.Ystd,
            self.train_or_test: False,
        }

    def get_variables_to_bound(self):
        return {self.omega: (self.min_omega, self.max_omega)}

    def get_variable_summaries(self):
        return [
            tf.summary.scalar("Rear Wing angle", self.omega[0, 0]),
            tf.summary.scalar("Front Wing angle", self.omega[0, 1]),
        ]

    @property
    def trainable_vars(self):
        return [self.omega]
