from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np
import logging
import tensorflow as tf
from networks.config import Config
from tensorflow.python.ops import control_flow_ops

CONV_WEIGHT_DECAY = 0.0005
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.0005
FC_WEIGHT_STDDEV = 0.01
BASENET_VARIABLES = 'basenet_variables'


def create_basenet(x, n_classes, keep_prob, is_training):
    """
    Creates the architecture of a LeNet

    :param x: input tensor (image), shape [?, nx, ny, channels]
    :param n_classes: number of classes in classification task
    """

    # Placeholder for the input image
    nx = 28
    ch = 1

    x_input = tf.reshape(x, tf.stack([-1, nx, nx, ch]))

    """
        Creates a new Baseline for the given parametrization.

        :param x: input tensor, shape [?,nx,ny,channels]
        :param keep_prob: dropout probability tensor
        :param n_class: number of output labels
        :param summaries: Flag if summaries should be created
        :param is_training: boolean tf.Variable, true indicates training phase
        """

    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = True  # if you use batch normalization, set this to False

    # Conv1 : # conv + relu
    with tf.variable_scope('conv1'):
        c['conv_filters_out'] = 256
        c['ksize'] = 5
        c['stride'] = 1
        x = conv(x_input, c, 'VALID')
        x = tf.nn.relu(x)

    # Conv2 : # conv + relu
    with tf.variable_scope('conv2'):
        c['conv_filters_out'] = 256
        c['ksize'] = 5
        c['stride'] = 1
        x = conv(x, c, 'VALID')
        x = tf.nn.relu(x)

    # Conv3 : # conv + relu
    with tf.variable_scope('conv3'):
        c['conv_filters_out'] = 128
        c['ksize'] = 5
        c['stride'] = 1
        x = conv(x, c, 'VALID')
        x = tf.nn.relu(x)

    # FC1 : # conv + relu
    with tf.variable_scope('fc1'):
        c['conv_filters_out'] = 328
        c['ksize'] = 16
        c['stride'] = 1
        x = conv(x, c, 'VALID')
        x = tf.nn.relu(x)

    # FC2 : # conv + relu
    with tf.variable_scope('fc2'):
        c['conv_filters_out'] = 192
        c['ksize'] = 1
        c['stride'] = 1
        x = fc_conv(x, c, 'SAME', keep_prob)
        x = tf.nn.relu(x)
    x_embedding = x

    # FC3
    with tf.variable_scope('fc3'):
        c['conv_filters_out'] = n_classes
        c['ksize'] = 1
        c['stride'] = 1
        x = fc_conv(x, c, 'SAME')

    logits = x

    tf.summary.histogram("logits/activations", logits)

    return logits, x_embedding


class BaseNet(object):
    """
    A baseline network

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
        self.keep_prob = tf.placeholder(tf.float32)
        self.n_class = n_class
        self.is_training = is_training

        # inputs: x -> image, n_class -> number of classes
        one_hot_y = tf.one_hot(self.y, depth=n_class, name="y_one_hot")
        logits, embedding = create_basenet(self.x, self.n_class, keep_prob=self.keep_prob, is_training=self.is_training)

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
            y_dummy = np.empty(x_test.shape[0])
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test,
                                                             self.y: y_dummy,
                                                             self.keep_prob: 1.})

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
                                                                       self.y: y_dummy,
                                                                       self.keep_prob: 1.})

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

    def train(self, data_provider, output_path, n_epochs=100, dropout=0.5, restore=False):
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
                                   self.net.y: batch_y,
                                   self.net.keep_prob: dropout})
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
                                                   self.net.y: val_y,
                                                   self.net.keep_prob: 1.})

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
                                                                 self.net.y: batch_y,
                                                                 self.net.keep_prob: 1.})

            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))
        else:
            self.net.is_training = False
            summary_str, loss, predictions = sess.run([self.summary_op, self.net.cost, self.net.predicter],
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y,
                                                                 self.net.keep_prob: 1.})
            self.net.is_training = True
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))


# Convolutional Layer
def conv(x, c, padding):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)

    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)

    if c['use_bias']:
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        x = x+bias

    return x


# Fully connected as conv
def fc_conv(x, c, padding, keep_prob=1.):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=FC_WEIGHT_DECAY)

    biases = _get_variable('biases',
                           shape=[filters_out],
                           initializer=tf.zeros_initializer)

    x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding) + biases

    x = control_flow_ops.cond(
        c['is_training'], lambda: tf.nn.dropout(x, keep_prob),  # do dropout if training
        lambda: x)  # don't do dropout if val/test

    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "alexnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, BASENET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)
