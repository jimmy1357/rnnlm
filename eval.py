# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import ptb_word_lm as lm
from ptb_word_lm import PTBModel as Model

tf.flags.DEFINE_string("checkpoint_dir", "./runs/1504700917", "checkpoint_path")
tf.flags.DEFINE_integer("fiction_words_count", 200, "Word count of the fiction written by RNN")

FLAGS = tf.flags.FLAGS

# evaluation
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)


def get_id2word():
    with open("id2word.dic", 'r')as fin:
        word_dict = {}
        for line in fin:
            line = line.strip()
            line = line.split("\t")
            if len(line) > 1:
                word_dict[int(line[0])] = line[1].decode("utf-8")
    return word_dict


def to_word(weights, word_dict):
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return word_dict[sample]
    """
    #print(weights.shape)
    return word_dict[np.argmax(weights)]


def get_word_dict():
    with open("word2id.dic", 'r')as fin:
        word_dict = {}
        for line in fin:
            line = line.strip()
            line = line.split("\t")
            if len(line) > 1:
                word_dict[line[0].decode("utf-8")] = int(line[1])
    return word_dict


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        """ 
        x_input = graph.get_operation_by_name("input_x").outputs[0]
        softmax_w = graph.get_operation_by_name("model/softmax_w").outputs[0]
        softmax_b = graph.get_operation_by_name("model/softmax_b").outputs[0]
        output = graph.get_operation_by_name("model/output").outputs[0]
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        """
        config = lm.InferConfig()
        # initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = Model(is_training=False, config=config)


        tf.initialize_all_variables().run()  # 这里是不是不能初始化，初始化了以后还会用训练好的模型吗?


        word_dict = get_word_dict()
        words = get_id2word()
        x = np.zeros((1, 1))
        x[0, 0] = word_dict[u'\u3002']  # 中文句号
        state_ = model.cell.zero_state(1, tf.float32)  # batch_size = 1
        state_ = sess.run(state_)  # convert tensor to ndarray, so that it can be feeded to the initial_state
        probs = tf.nn.softmax(model.logits)
        [probs_, state_] = sess.run([probs, model._final_state],
                                    feed_dict={model._input_data: x, model._initial_state: state_})
        word = to_word(probs_, words)  # convert probabilities to words and get the biggest one
        fiction = ""
        word_cnt = 0
        while word != u'\u3002' and word_cnt < FLAGS.fiction_words_count:
            fiction += word
            x = np.zeros((1, 1))
            x[0, 0] = word_dict[word]
            [probs_, state_] = sess.run([probs, model._final_state],
                                        feed_dict={model._input_data: x, model._initial_state: state_})
            word = to_word(probs_, words)
            word_cnt += 1
        print("FICTION:  " + fiction)
