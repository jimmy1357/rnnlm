# -*- coding: utf-8 -*-

import tensorflow as tf
import collections


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "/root/jichen/data/poetry.txt", "data path")
flags.DEFINE_string("train_data_path", "/root/jichen/data/ptb.train.txt", "data path")
flags.DEFINE_string("dev_data_path", "/root/jichen/data/ptb.valid.txt", "data path")
flags.DEFINE_string("test_data_path", "/root/jichen/data/ptb.test.txt", "data path")
flags.DEFINE_integer("train_length", 40000, "num. of lines in training set")
flags.DEFINE_integer("dev_length", 1500, "num. of lines in dev set")
flags.DEFINE_integer("test_length", 1500, "num. of lines in test set")
flags.DEFINE_integer("common_words_count", 3000, "num. of common words in Chinese")

FLAGS = flags.FLAGS

def seg_poetry(sentence, words):
    line = ""
    for word in sentence.decode("utf-8"):
        if word not in words:
            word = "#"
        line += word.encode("utf-8") + " "
    line = "[ " + line + "] "
    return line

def preprocess():
    cnt = 0
    words = common_words(FLAGS.common_words_count)
    """
    for w in words:
        print w.encode("utf-8")
    """
    with open(FLAGS.data_path, 'r')as fin, open(FLAGS.train_data_path, 'w')as train_out, open(FLAGS.dev_data_path, 'w')as dev_out, open(FLAGS.test_data_path, 'w')as test_out:
        for line in fin:
            line = line.strip()
            line = line.split(":")
            if len(line) <= 1:
                continue
            line = seg_poetry(line[1], words)
            cnt += 1
            if cnt < FLAGS.train_length:
                train_out.write(line + "\n")
            elif (cnt >= FLAGS.train_length) and (cnt < FLAGS.dev_length + FLAGS.train_length):
                dev_out.write(line + "\n")
            else:
                test_out.write(line + "\n")

def common_words(word_count):
    with open(FLAGS.data_path, 'r')as fin:
        all_words = []
        for line in fin:
            line = line.strip()
            line = line.split(":")
            if len(line) <= 1:
                continue
            all_words += [word for word in line[1].decode("utf-8")]
        counter = collections.Counter(all_words)
        wordslist = counter.most_common(word_count)  # wordslist = [('中', 10), ('国'， 9), ... ] ---> list[tuples]
        """
        words = []
        for k,v in wordslist:
            words.append(k)
        """
        words, _ = zip(*wordslist)  # unzip, separate word and its count, put words together, so as to counts. 
        return words

if __name__ == "__main__":
    preprocess()
