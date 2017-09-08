# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf
"""
按行分开读取诗歌，每首诗读取为一个list，不要按现在这种全部文章读成一个list的方法
"""
"""Utilities for parsing PTB text files."""

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "").split()


def _read_file(filename):
    poetry_list = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            line = line.strip().decode("utf-8").split()
            poetry_list.append(line)
    return poetry_list



def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    # save word 2 id dict
    with open("word2id.dic", 'w')as fout:
        for k, v in word_to_id.items():
            fout.write(k.encode("utf-8") + "\t" + str(v) + "\n")
    # save id 2 word dict
    with open("id2word.dic", "w")as fout:
        for k, v in word_to_id.items():
            fout.write(str(v) + "\t" + k.encode("utf-8") + "\n")

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_file(filename)
    poetry_id_list = []
    for poetry in data:
        poetry_id_list.append([word_to_id[word] for word in poetry if word in word_to_id])
    return poetry_id_list


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    print("len valid: " + str(len(valid_data)))
    return train_data, valid_data, test_data, vocabulary

def gen_batch(raw_data, batch_size):
    """Generate batch data of poetries for training.
    """
    poetry_cnt = len(raw_data)
    batch_len = poetry_cnt // batch_size
    for i in range(batch_len):
        batches = raw_data[batch_size * i : batch_size * (i + 1)]
        if i % 10 == 0 :
            print(batches)
        length = max(map(len, batches))  # choose the maximum length of batches, 选择一批诗中包含字数最多的诗，返回该诗的字数
        xdata = np.full((batch_size, length), 0, np.int32)  # 使用0初始化这个batch_size * length的矩阵( 这个矩阵中 保存的是word的id)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]  # 把batches 中的第row首诗赋值给xdata
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:,1:]  # :-1表示除了最后一个元素以外的其他元素，这一行的意思就是把xdata向前偏移一位，然后赋值给y，忽略x偏移后溢出的一位和补上的一位
        yield (xdata, ydata, length)

def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
