import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_X_y
from tensorflow.contrib.distributions import Normal

import baselines.common.tf_util as U
from remps.model_approx.model_approximator import ModelApproximator
from remps.utils.utils import get_default_tf_dtype


def reduce_std(x, axis, dtype=tf.float32):
    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    var = tf.reduce_sum(tf.square(x - mean), axis=axis, keepdims=True) / (
        tf.cast(tf.shape(x)[0], dtype) - 1
    )
    return tf.sqrt(var)


class NNModel(ModelApproximator):
    def get_probability(self):
        return self.prob

    def __init__(self, state_dim, param_dim, name="NN", training_set_size=4000):
        """
        Fit a NN for predicting the next state distribution
        Output of NN are the parameters of a parametric distribution (Gaussian)
        """
        self.sess = None
        self.log_prob = None
        self.prob = None
        self.dtype = get_default_tf_dtype()
        # must be initialized
        self.name = name
        self.x_range = 4.8
        self.theta_range = 180
        self.XData = None
        self.YData = None
        self.state_dim = state_dim
        self.x_dim = state_dim + param_dim
        self.gp_list = []
        self.training_set_size = training_set_size
        self.global_step = 0
        self.folder = self.name + "NNData" + "/"
        self.min_omega = 0.1
        self.max_omega = 30

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
        self.X = U.get_placeholder(name="X", dtype=self.dtype, shape=(None, self.x_dim))
        self.Y = U.get_placeholder(
            name="Y", dtype=self.dtype, shape=(None, self.state_dim)
        )
        with tf.variable_scope(self.name):
            # build the action vector
            self.omega = tf.get_variable(
                dtype=self.dtype,
                name="omega",
                shape=(),
                initializer=tf.initializers.constant(initial_omega),
            )
            X = self.X  # - Xmean_) / Xstd_
            Y = self.Y  # - YMean_) / Ystd_
            # build the action vector
            forces = self.omega * actions
            forces_full = tf.concat(
                [tf.reshape(forces[:, 0], (-1, 1)), tf.reshape(forces[:, 1], (-1, 1))],
                axis=0,
            )
            batch_size = tf.shape(states)[0]
            x_full = tf.concat([states, states], axis=0)
            x_full = tf.concat([x_full, forces_full], axis=1)
            x_full = (x_full - self.Xmean_ph) / self.Xstd_ph
            next_states_full = tf.concat([next_states, next_states], axis=0)
            next_states_full = (next_states_full - self.Ymean_ph) / self.Ystd_ph

            # build the network
            hidden_layer_size = 10
            biases = tf.get_variable(
                "b",
                [hidden_layer_size],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )
            W = tf.get_variable(
                "W",
                [self.x_dim, hidden_layer_size],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )

            x_input = U.switch(train_or_test, X, x_full)
            h = tf.matmul(x_input, W)
            h = tf.tanh(h + biases)

            # now we need state_dim output neurons, one for each state dimension to predict
            biases_out = tf.get_variable(
                "b_out",
                [self.state_dim],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )
            W_out = tf.get_variable(
                "W_out",
                [hidden_layer_size, self.state_dim],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )
            means = tf.matmul(h, W_out) + biases_out

            # x_input_first = x_input[:, 0:self.x_dim - 1]
            # forces = tf.reshape(x_input[:, self.x_dim - 1], (-1, 1))
            # x_input = tf.concat([x_input_first, tf.abs(forces)], axis=1)

            hidden_var = 10
            biases_var = tf.get_variable(
                "b_var",
                [hidden_var],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )
            W_var = tf.get_variable(
                "W_var",
                [self.x_dim, hidden_var],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )

            h = tf.nn.sigmoid(tf.matmul(x_input, W_var) + biases_var)

            W_out_var = tf.get_variable(
                "W_out_var",
                [hidden_var, self.state_dim],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )

            biases_out_var = tf.get_variable(
                "b_out_var",
                [self.state_dim],
                initializer=tf.random_normal_initializer(0, 0.001, dtype=self.dtype),
                dtype=self.dtype,
            )

            var = tf.exp(tf.matmul(h, W_out_var) + biases_out_var)

            std = tf.sqrt(var)

            pdf = Normal(means, std)

            y_output = U.switch(train_or_test, Y, next_states_full)

            log_prob = tf.reduce_sum(pdf.log_prob(y_output), axis=1, keepdims=True)
            prob = tf.reduce_prod(pdf.prob(y_output), axis=1, keepdims=True)

            # loss is the negative loss likelihood
            self.loss = -tf.reduce_mean(log_prob)
            self.valid_loss = -tf.reduce_mean(log_prob)

            self.fitting_vars = [
                biases,
                W,
                biases_out,
                W_out,
                biases_var,
                W_var,
                W_out_var,
                biases_out_var,
            ]
            # create fitting collection
            for v in self.fitting_vars:
                tf.add_to_collection("fitting", v)

            opt = tf.train.AdamOptimizer()
            self.minimize_op = opt.minimize(self.loss, var_list=self.fitting_vars)

            log_prob_a0 = log_prob[0:batch_size, :]
            log_prob_a1 = log_prob[batch_size:, :]
            prob_a0 = prob[0:batch_size, :]
            prob_a1 = prob[batch_size:, :]
            self.log_prob = tf.concat([log_prob_a0, log_prob_a1], axis=1)
            self.prob = tf.concat([prob_a0, prob_a1], axis=1)
            means_list = []
            var_list = []
            for i in range(self.state_dim):
                means_a0 = tf.reshape(means[0:batch_size, i], (-1, 1))
                means_a1 = tf.reshape(means[batch_size : 2 * batch_size, i], (-1, 1))
                means_actions = tf.concat([means_a0, means_a1], axis=1)
                means_ = tf.reduce_sum(
                    tf.multiply(means_actions, actions_one_hot), axis=1, keepdims=True
                )
                means_list.append(means_)
                # same for variance
                var_a0 = tf.reshape(var[0:batch_size, i], (-1, 1))
                var_a1 = tf.reshape(var[batch_size : 2 * batch_size, i], (-1, 1))
                var_actions = tf.concat([var_a0, var_a1], axis=1)
                var_ = tf.reduce_sum(
                    tf.multiply(var_actions, actions_one_hot), axis=1, keepdims=True
                )
                var_list.append(var_)

            self.means = tf.concat(means_list, axis=1)
            self.variances = tf.concat(var_list, axis=1)
            self.train_or_test = train_or_test
            self.loss_summary = tf.summary.scalar("Loss", self.loss)
            self.valid_loss_summary = tf.summary.scalar("ValidLoss", self.valid_loss)
        return self.log_prob, self.prob

    def store_data(self, X, Y, normalize_data=False, update_statistics=True):
        """
        Store training data inside training set
        """
        X, Y = check_X_y(X, Y, multi_output=True, y_numeric=True)
        np.save(self.folder + "X.npy", X)
        np.save(self.folder + "Y.npy", Y)
        # if self.XData is None:
        #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        #     self.XData = X_train
        #     self.YData = y_train
        #     self.XTest = X_test
        #     self.YTest = y_test
        # else:
        #     self.XData = np.vstack((self.XData, X))
        #     self.YData = np.vstack((self.YData, Y))
        #
        # if update_statistics:
        #     if normalize_data:
        #         self.Xmean = np.reshape(np.mean(self.XData, axis=0), (1,-1))
        #         self.Xstd = np.reshape(np.std(self.XData, axis=0), (1,-1))
        #         self.Ymean = np.reshape(np.mean(self.YData, axis=0), (1,-1))
        #         self.Ystd = np.reshape(np.std(self.YData, axis=0), (1,-1))
        #     else:
        #         self.Xmean = np.zeros((1, np.shape(self.XData)[1]), dtype=self.XData.dtype)
        #         self.Ymean = np.zeros((1, np.shape(self.YData)[1]), dtype=self.YData.dtype)
        #         self.Xstd = np.ones((1, np.shape(self.XData)[1]), dtype=self.XData.dtype)
        #         self.Ystd = np.ones((1, np.shape(self.YData)[1]), dtype=self.XData.dtype)
        # self.normalizedXData = (self.XData - self.Xmean) / self.Xstd
        # self.normalizedYData = (self.YData - self.Ymean) / self.Ystd
        # self.normalizedXTest = (self.XTest - self.Xmean) / self.Xstd
        # self.normalizedYTest = (self.YTest - self.Ymean) / self.Ystd

    # fit the gaussian process using XData and YData provided in store data
    def fit(
        self,
        load_from_file=False,
        save_to_file=True,
        action_ph=None,
        states_ph=None,
        next_states_ph=None,
        load_weights=True,
        add_onpolicy=False,
        training_step=40000,
    ):

        training_set_size = 100000
        X = np.load(self.folder + "X.npy")
        Y = np.load(self.folder + "Y.npy")

        if add_onpolicy:
            X_onpolicy = np.load(self.folder + "on_policyX.npy")
            Y_onpolicy = np.load(self.folder + "on_policyY.npy")
            X = np.vstack((X, X_onpolicy))
            Y = np.vstack((Y, Y_onpolicy))

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        if not add_onpolicy:
            self.Xmean = np.reshape(np.mean(X_train, axis=0), (1, -1))
            self.Xstd = np.reshape(np.std(X_train, axis=0), (1, -1))
            self.Ymean = np.reshape(np.mean(y_train, axis=0), (1, -1))
            self.Ystd = np.reshape(np.std(y_test, axis=0), (1, -1))

        if not load_weights:
            X_train = (X_train - self.Xmean) / self.Xstd
            X_test = (X_test - self.Xmean) / self.Xstd
            y_train = (y_train - self.Ymean) / self.Ystd
            y_test = (y_test - self.Ymean) / self.Ystd
            batch_size = 10000
            for _ in range(training_step):
                # sample a batch
                ind = np.arange(0, np.shape(X_train)[0])
                selected_ind = np.random.choice(ind, size=batch_size, replace=False)
                inputs = X_train[selected_ind, :]
                targets = y_train[selected_ind, :]
                feed_dict = {
                    self.train_or_test: True,
                    self.X: inputs,
                    self.Y: targets,
                    self.Xmean_ph: self.Xmean,
                    self.Ymean_ph: self.Ymean,
                    self.Xstd_ph: self.Xstd,
                    self.Ystd_ph: self.Ystd,
                    # dummy things
                    action_ph: [[0, 1]],
                    states_ph: np.ones((1, X.shape[1] - 1)),
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
                    self.X: X_test,
                    self.Y: y_test,
                    self.Xmean_ph: self.Xmean,
                    self.Ymean_ph: self.Ymean,
                    self.Xstd_ph: self.Xstd,
                    self.Ystd_ph: self.Ystd,
                    # dummy things
                    action_ph: [[0, 1]],
                    states_ph: np.ones((1, X.shape[1] - 1)),
                    next_states_ph: np.ones((1, Y.shape[1])),
                }
                valid_loss, loss_summary = self.sess.run(
                    [self.valid_loss, self.valid_loss_summary], feed_dict=feed_dict
                )
                self.summary_writer.add_summary(loss_summary, self.global_step)

                # print("Loss: ", loss)
                # print("Validation Loss: ", valid_loss)

                self.global_step += 1

            print("FITTED!!")
            if not add_onpolicy:
                weights = U.GetFlat(self.fitting_vars)()
                np.save(self.folder + "weights.npy", weights)
        else:
            weights = np.load(self.folder + "weights.npy")
            U.SetFromFlat(self.fitting_vars, dtype=self.dtype)(weights)

    def get_omega(self):
        return self.omega

    def set_omega(self):
        pass

    def get_feed_dict(self):
        return {
            self.X: np.ones((1, self.Xmean.shape[1])),
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
        return [tf.summary.scalar("Omega", self.omega)]

    @property
    def trainable_vars(self):
        return [self.omega]
