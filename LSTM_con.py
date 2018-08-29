# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 09:59:05 2018

@author: ConorGL

This is part 2 of attempting to build the Florida Man model. I will create an 
LSTM in order to predict words in a coherent sentence structure.
"""

"""Part 1 - Data preprocessing

This first section, we will be using the dataset from Penn Tree Bank and 
cleaning/preparing it for use in our LSTM

"""
import os
import tensorflow as tf
import sys
import collections

def read_words(filename):
    #This function joins up the filename into a list of words with the end of 
    #line token being replaced by <eos> (end of sentence)
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()



def build_vocab(filename):
    #This function gets the words from a file, creates a counter object from them,
    #sorts them by frequency, creates a new list of just the top 10000 words which
    #then finally returns a dictionary with each word assigned an id
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    #This function returns a list of integers of each word in each sentence of a 
    #file corresponding to the words position in the dictionary
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    """ 
    Now comes the important bit. We define how to load our data. As stated 
    previously we are using the PTB data and it is currently located in 
    "*simple-examples/data/" for me. Once we have assigned our data path, we use 
    create our vocabulary from the training data using the build_vocab function.

    The next step is to create our data for training/testing/validation. We use 
    the function file_to_word_ids to create lists of integers representing the 
    different words in each set. Finally we print some test values to ensure that
    the load process has correctly run.
    """
    
    data_path = 'simple-examples/data/'
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    #Test the outputs by printing some sample values
    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    tf_raw_data = tf.convert_to_tensor(raw_data, name="tf_raw_data", dtype=tf.int32)

    data_len = tf.size(tf_raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(tf_raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y