# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rnn

import time
import os

import numpy as np
import tensorflow as tf

import poetry_reader

import inspect

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/home/bruce/dataset/nlp/Chinese_fiction", "data_path")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("checkpoint_path", "./", "checkpoint_path")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def lstm_cell(size):
    if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    else:
        return rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        logging.info("vocab_size in ptbmodel: {}".format(vocab_size))

        #self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_x")
        #self._targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_y")
        self._input_data = tf.placeholder(tf.int32, [batch_size, None], name="input_x")
        self._targets = tf.placeholder(tf.int32, [batch_size, None], name="input_y")

        lstm_cells = []
        for _ in range(config.num_layers):
            if is_training and config.keep_prob < 1:
                lstm = rnn.DropoutWrapper(lstm_cell(size), output_keep_prob=config.keep_prob)
            else:
                lstm = lstm_cell(size)
            lstm_cells.append(lstm)
        self.cell = rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
        
        self._initial_state = self.cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"), tf.variable_scope("RNN"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        """
        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(inputs, num_steps, 1)] """
        # outputs, state = rnn.static_rnn(self.cell, inputs, initial_state=self._initial_state, scope="RNN")
        # length of input data unfixed, use dynamic_rnn instead
        outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self._initial_state, scope="RNN")

        #output = tf.reshape(tf.concat(outputs, 1), [-1, size], name="output")
        output = tf.reshape(outputs, [-1, size], name="output")
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        """
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())]) """
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones_like([tf.reshape(self._targets, [-1])], dtype=data_type())])  # Creates a tensor(like before) with(but) all elements set to 1.
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 100000  # bigger params for larger training sets


class InferConfig(object):
    """Infer config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 100000  # bigger params for larger training sets


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 100000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 100000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 100000


def run_epoch(session, model, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    #epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    epoch_size = ((len(data) // model.batch_size) - 1) // 35  # 无意义
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y, length) in enumerate(poetry_reader.gen_batch(data, model.batch_size)):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += length

        # if verbose and step % (epoch_size // 10) == 10:
        if verbose and step % 100 == 0:
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = poetry_reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, vocab_size = raw_data

    config = get_config()
    config.vocab_size = vocab_size + 1
    logging.info("Set vocabulary size to: " + str(config.vocab_size))
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.vocab_size = vocab_size + 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        timestamp = str(int(time.time()))
        saver = tf.train.Saver(tf.all_variables())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            # print(train_data)
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            # print(valid_data)
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            path = FLAGS.checkpoint_path + "runs/" + timestamp
            print("Writing to {}\n".format(path + "/fiction_model"))
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(session, path + "/fiction_model", global_step=i)

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
