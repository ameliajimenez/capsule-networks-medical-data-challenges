from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf


def squash(s, axis=-1, epsilon=1e-7, name=None):
    """
    Non-linear function to ensure that vectors are in range [0,1]

    :param s: output of the capsule
    :param epsilon: to avoid division by zero
    :param name: name for TensorFlow scope
    """
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    """
    Computes the norm of a capsule

    :param s: output of the capsule
    :param epsilon: to avoid rounding errors
    :param keep_dims: do not reduce dimension of the tensor if True
    :param name: name for TensorFlow scope
    """
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def create_capsnet(x, y, n_caps1, d_caps1, n_caps2, d_caps2):
    """
    Creates a Capsule Network (CapsNet), composed of an encoder and a decoder.
    Encoder: two convolutional layers, followed by two capsule (primary and secondary) layers
    Decoder: two fully connected layers that reconstruct the input image

    :param x: input tensor (image), shape [?, nx, ny, channels]
    :param y: input tensor (label), shape [?]
    :param n_caps1: number of primary capsules
    :param d_caps1: dimensionality of each primary capsule
    :param n_caps2: number of secondary capsules, equal to the number of classes
    :param d_caps2: dimensionality of each secondary capsule
    """

    # Placeholder for the input image
    nx = 28
    ch = 1

    x_input = tf.reshape(x, tf.stack([-1, nx, nx, ch]))
    # this reshape is not necessary but it is useful to check the shape of the tensors at every layer

    # dictionary to monitor training/gradients during training per layer
    ouputs_per_layer = OrderedDict()

    ##### ENCODER #####
    # Conv1 : # conv + relu
    with tf.variable_scope('conv1'):
        conv1_params = {
            "filters": 256,
            "kernel_size": 9,
            "strides": 1,  # 1
            "padding": "valid",
            "activation": tf.nn.relu,
        }
        x = tf.layers.conv2d(x_input, name="conv1", **conv1_params)

        ouputs_per_layer['conv1'] = x

    # Conv2 : # conv + relu
    with tf.variable_scope('conv2'):
        conv2_params = {
            "filters": 256,  # 256 convolutional filters
            "kernel_size": 9,
            "strides": 2,
            "padding": "valid",
            "activation": tf.nn.relu
        }
        x = tf.layers.conv2d(x, name="conv2", **conv2_params)
        ouputs_per_layer['conv2'] = x

    # Capsule 1
    # Reshape & squash
    caps1_raw = tf.reshape(x, [-1, n_caps1, d_caps1], name="caps1_raw")

    caps1_output = squash(caps1_raw, name="caps1_output")

    # Transformation matrix
    init_sigma = 0.01

    W_init = tf.random_normal(
        shape=(1, n_caps1, n_caps2, d_caps2, d_caps1),
        stddev=init_sigma, dtype=tf.float32, name="W_init")
    W = tf.Variable(W_init, name="W")

    batch_size = tf.shape(x)[0]
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

    caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                           name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                       name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, n_caps2, 1, 1],
                                 name="caps1_output_tiled")

    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                name="caps2_predicted")

    ## Routing by agreement

    raw_weights = tf.zeros([batch_size, n_caps1, n_caps2, 1, 1],
                           dtype=np.float32, name="raw_weights")

    ### Round 1
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

    weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                       name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                 name="weighted_sum")

    caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                  name="caps2_output_round_1")

    caps2_output_round_1_tiled = tf.tile(
        caps2_output_round_1, [1, n_caps1, 1, 1, 1],
        name="caps2_output_round_1_tiled")

    agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                          transpose_a=True, name="agreement")

    ### Round 2
    raw_weights_round_2 = tf.add(raw_weights, agreement,
                                 name="raw_weights_round_2")

    routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                            dim=2,
                                            name="routing_weights_round_2")
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                               caps2_predicted,
                                               name="weighted_predictions_round_2")
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                         axis=1, keep_dims=True,
                                         name="weighted_sum_round_2")
    caps2_output_round_2 = squash(weighted_sum_round_2,
                                  axis=-2,
                                  name="caps2_output_round_2")

    caps2_output_round_2_tiled = tf.tile(
        caps2_output_round_2, [1, n_caps1, 1, 1, 1],
        name="caps2_output_round_2_tiled")

    agreement = tf.matmul(caps2_predicted, caps2_output_round_2_tiled,
                          transpose_a=True, name="agreement")

    ### Round 3
    raw_weights_round_3 = tf.add(raw_weights, agreement,
                                 name="raw_weights_round_3")

    routing_weights_round_3 = tf.nn.softmax(raw_weights_round_3,
                                            dim=2,
                                            name="routing_weights_round_3")

    weighted_predictions_round_3 = tf.multiply(routing_weights_round_3,
                                               caps2_predicted,
                                               name="weighted_predictions_round_3")

    weighted_sum_round_3 = tf.reduce_sum(weighted_predictions_round_3,
                                         axis=1, keep_dims=True,
                                         name="weighted_sum_round_3")

    caps2_output_round_3 = squash(weighted_sum_round_3,
                                  axis=-2,
                                  name="caps2_output_round_3")

    caps2_output = caps2_output_round_3  # <- output 1

    # Estimated Class Probabilities (Length)
    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

    y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

    y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

    ##### DECODER #####
    # Mask -> Reconstruction
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")

    # condition: use ground truth or predicted labels for reconstruction
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred,  # if False
                                     name="reconstruction_targets")

    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=n_caps2,
                                     name="reconstruction_mask")

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, n_caps2, 1, 1],
        name="reconstruction_mask_reshaped")

    caps2_output_masked = tf.multiply(
        caps2_output, reconstruction_mask_reshaped,
        name="caps2_output_masked")

    decoder_input = tf.reshape(caps2_output_masked,
                               [-1, n_caps2 * d_caps2],
                               name="decoder_input")

    # Fully connected layers
    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = nx*nx

    with tf.name_scope("decoder"):
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        decoder_output = tf.layers.dense(hidden2, n_output,   # <- output 2
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")

    return caps2_output, decoder_output


class CapsNet(object):
    """
    A Capsule Network

    :param n_class: number of classes for the classification problem
    :param channels: number of channels in the input image
    :param is_training: boolean tf.Variable, true indicates training phase
    :param cost_kwargs: (optional) kwargs passed to the cost function
    """

    def __init__(self, n_class, channels, is_training, cost_kwargs={}, **kwargs):

        tf.reset_default_graph()

        self.summaries = True

        self.x = tf.placeholder(shape=[None, None, None, channels], dtype=tf.float32, name='x_image')
        self.y = tf.placeholder(shape=[None], dtype=tf.int64, name='y_class')
        self.n_class = n_class
        self.n_caps1 = 1152  # 6x6x32
        self.d_caps1 = 8
        self.n_caps2 = n_class
        self.d_caps2 = 16

        # inputs: x -> image, y -> label, number and dimension of capsules
        capsule2, decoder = create_capsnet(self.x, self.y, self.n_caps1, self.d_caps1, self.n_caps2, self.d_caps2)

        # Weighted loss from margin and reconstruction losses
        self.cost_margin, self.cost_recons, self.cost = self._get_cost(capsule2, decoder, cost_kwargs)

        self.predicter = safe_norm(capsule2, axis=-2)
        self.predicter_embedding = capsule2
        self.decoder = decoder
        self.embedding = capsule2

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: data to predict on. shape [n, nx, nx, channels]
        :returns prediction: predicted classification labels
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            x_test = x_test.reshape([-1, 28, 28, 1])
            y_dummy = np.empty(x_test.shape[0])
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test,
                                                             self.y: y_dummy})

        return prediction

    def predict_embedding(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: data to predict on. shape [n, nx, nx, channels]
        :returns prediction: embedded features for the input
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            x_test = x_test.reshape([-1, 28, 28, 1])
            y_dummy = np.empty(x_test.shape[0])
            prediction = sess.run(self.predicter_embedding, feed_dict={self.x: x_test,
                                                                       self.y: y_dummy})

        return prediction

    def decode(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: data to predict on. shape [n, nx, nx, channels]
        :returns prediction: output of decoder, reconstructed image
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            y_dummy = np.empty(x_test.shape[0])
            x_test = x_test.reshape([-1, 28, 28, 1])
            prediction = sess.run(self.decoder, feed_dict={self.x: x_test,
                                                           self.y: y_dummy})

        return prediction

    def tweak_capsule_dimensions(self, model_path, caps2_output, y_test):
        """
       Feed modified output dimensions of the secondary capsule and obtain the tweaked reconstruction of the images

       Optional arguments are:
        :param model_path: path to the model checkpoint to restore
        :param caps2_output: modified output dimensions of the secondary capsule
        :param y_test: classification labels
        :returns prediction: output of decoder, reconstructed image
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            x_dummy = np.empty(shape=(y_test.shape[0], 28, 28, 1))
            prediction = sess.run(self.decoder, feed_dict={self.x: x_dummy,
                                                           self.y: y_test,
                                                           self.embedding: caps2_output
                                                           })
        return prediction

    def _get_cost(self, capsule2, decoder, cost_kwargs):
        """
        Weighted average of margin loss and reconstruction loss

        Optional arguments are:
        :param m_plus: (optional) value for the upper margin
        :param m_minus: (optional) value for the lower margin
        :param lambda: (optional) weights the contribution of each margin
        :param alpha: (optional) weight for the reconstruction loss
        """

        # Classification - Margin loss
        m_plus = cost_kwargs.pop("m_plus", 0.9)
        m_minus = cost_kwargs.pop("m_minus", 0.1)
        lambda_ = cost_kwargs.pop("lambda_", 0.5)
        alpha = cost_kwargs.pop("alphna", 0.0005)  # weight losses

        T = tf.one_hot(self.y, depth=self.n_caps2, name="T")

        caps2_output_norm = safe_norm(capsule2, axis=-2, keep_dims=True, name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, self.n_class), name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, self.n_class), name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")

        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="loss")

        # Reconstruction Loss
        n_output = 784  # 28*28

        X_flat = tf.reshape(self.x, [-1, n_output], name="X_flat")
        squared_difference = tf.square(X_flat - decoder, name="squared_difference")
        reconstruction_loss = tf.reduce_sum(squared_difference, name="reconstruction_loss")

        # Final Loss
        loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

        return margin_loss, reconstruction_loss, loss

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
    Trains a net instance

    :param net: the net instance to train
    :param batch_size: (optional) size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    verification_batch_size = 4

    def __init__(self, net, optimizer="adam", batch_size=64, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, global_step):

        learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)

        if self.optimizer == 'adam':
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.1)
            decay_steps = self.opt_kwargs.pop("decay_steps", 2000)  # corresponds to approx. 5 epochs

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)

        elif self.optimizer == 'adagrad':
            self.learning_rate_node = tf.Variable(learning_rate)
            opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate_node)

        elif self.optimizer == "momentum":
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.5)
            momentum = self.opt_kwargs.pop("momentum", 0.9)
            decay_steps = self.opt_kwargs.pop("decay_steps", 100)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum, **self.opt_kwargs)

        grads = opt.compute_gradients(self.net.cost)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op

        return train_op

    def _initialize(self, output_path):
        global_step = tf.Variable(0)
        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('loss_margin', self.net.cost_margin)
        tf.summary.scalar('loss_recons', self.net.cost_recons)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        output_path = os.path.abspath(output_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, n_epochs=100, restore=False):
        """
        Launches the training process, save model at minimum validation loss

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param n_epochs: number of epochs
        :param restore: Flag if previous model should be restored
        """
        train_batch_size = self.batch_size

        save_path = os.path.join(output_path, "model.cpkt")
        if n_epochs == 0:
            return save_path

        init = self._initialize(output_path)

        with tf.Session() as sess:
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            # summaries for training and validation
            summary_writer_train = tf.summary.FileWriter(output_path + '/logs/train', graph=sess.graph)
            summary_writer_val = tf.summary.FileWriter(output_path + '/logs/val', graph=sess.graph)

            n_iterations_validation = data_provider.validation.num_examples // train_batch_size
            n_iterations_per_epoch = data_provider.train.num_examples // train_batch_size

            logging.info("Start optimization")

            # loop over epochs
            for epoch in range(0, n_epochs):
                total_loss = 0

                # training
                for step in range(1, n_iterations_per_epoch + 1):
                    # training samples
                    batch_x, batch_y = data_provider.train.next_batch(train_batch_size)
                    batch_x = batch_x.reshape([-1, 28, 28, 1])
                    # Run optimization op (backprop)
                    _, loss, lr = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: batch_y})
                    total_loss += loss

                self.output_stats(sess, summary_writer_train, epoch, batch_x, batch_y, phase='Train')

                # validation
                loss_vals = []
                best_loss_val = np.infty
                for step in range(1, n_iterations_validation + 1):
                    # validation samples
                    val_x, val_y = data_provider.validation.next_batch(train_batch_size)
                    val_x = val_x.reshape([-1, 28, 28, 1])
                    loss_val = sess.run(self.net.cost,
                                        feed_dict={self.net.x: val_x,
                                                   self.net.y: val_y})

                    loss_vals.append(loss_val)

                loss_val = np.mean(loss_vals)

                self.output_stats(sess, summary_writer_val, epoch, val_x, val_y, phase='Val')

                if loss_val < best_loss_val:
                    save_path = self.net.save(sess, save_path)
                    best_loss_val = loss_val
                    print('Saved at epoch: {}, Validation loss: {:.4f}'.format(epoch, best_loss_val))

            logging.info("Optimization Finished!")

            return save_path

    def output_stats(self, sess, summary_writer, step, batch_x, batch_y, phase):
        # Calculate batch loss and accuracy
        if phase == 'Train':
            summary_str, loss, predictions = sess.run([self.summary_op, self.net.cost, self.net.predicter],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y})

            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))
        else:
            self.net.is_training = False
            summary_str, loss, predictions = sess.run([self.summary_op, self.net.cost, self.net.predicter],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y})
            self.net.is_training = True
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))
