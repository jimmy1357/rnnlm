# -*- coding: utf-8 -*-

import tensorflow as tf
import jieba

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "/root/jichen/data/changan.txt", "data path")
flags.DEFINE_string("train_data_path", "/root/jichen/data/ptb.train.txt", "data path")
flags.DEFINE_string("dev_data_path", "/root/jichen/data/ptb.valid.txt", "data path")
flags.DEFINE_string("test_data_path", "/root/jichen/data/ptb.test.txt", "data path")
flags.DEFINE_integer("train_length", 1600, "num. of lines in training set")
flags.DEFINE_integer("dev_length", 400, "num. of lines in dev set")
flags.DEFINE_integer("test_length", 400, "num. of lines in test set")

FLAGS = flags.FLAGS

def seg(sentence):
    words = jieba.lcut(sentence)
    line = ""
    for word in words:
        line += word + " "
    return line

def preprocess():
    cnt = 0
    with open(FLAGS.data_path, 'r')as fin, open(FLAGS.train_data_path, 'w')as train_out, open(FLAGS.dev_data_path, 'w')as dev_out, open(FLAGS.test_data_path, 'w')as test_out:
        for line in fin:
            line = line.strip()
            line = seg(line)
            cnt += 1
            if cnt < FLAGS.train_length:
                train_out.write(line + "\n")
            elif (cnt >= FLAGS.train_length) and (cnt < FLAGS.dev_length + FLAGS.train_length):
                dev_out.write(line + "\n")
            else:
                test_out.write(line + "\n")

if __name__ == "__main__":
    preprocess()
