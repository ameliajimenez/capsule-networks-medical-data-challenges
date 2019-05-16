from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def create_lenet(x, n_classes):
    """
    Creates the architecture of a LeNet

    :param x: input tensor (image), shape [?, nx, ny, channels]
    :param n_classes: number of classes in classification task
    """

    # Placeholder for the input image
    nx = 32
    ch = 1

    x_input = tf.reshape(x, tf.stack([-1, nx, nx, ch]))

    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: convolutional, activation (ReLu), pooling
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x_input, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: convolutional, activation (ReLu), pooling
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc1 = flatten(pool_2)  # 5*5*16 = 400

    # Layer 3: fully connected, activation (ReLu)
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # Layer 4: fully connected, activation (ReLu)
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    tf.summary.histogram("logits/activations", logits)

    return logits, fc2


class LeNet(object):
    """
    A LeNet

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

        # inputs: x -> image, n_class -> number of classes
        one_hot_y = tf.one_hot(self.y, depth=n_class, name="y_one_hot")
        logits, embedding = create_lenet(self.x, self.n_class)

        # Cost of loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))

        self.predicter = tf.nn.softmax(logits)
        self.predicter_embedding = embedding

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
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input image
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
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input image
            y_dummy = np.empty(x_test.shape[0])
            prediction = sess.run(self.predicter_embedding, feed_dict={self.x: x_test,
                                                                       self.y: y_dummy})

        return prediction

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
                    batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input image
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
                    val_x = np.pad(val_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input image
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
