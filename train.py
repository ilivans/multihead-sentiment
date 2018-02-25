#!/usr/bin/python
"""
Train multihead-classifier with triplet loss
"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from utils import get_vocabulary_size, batch_generator
from words_encoder import WordsEncoder

# Embeddings params
NUM_WORDS = 10000
EMBEDDING_DIM = 100

# Model params
HIDDEN_SIZE = 150
HEAD_SIZE = 50
KEEP_PROB = 0.8
LEARNING_RATE = 1e-3

# Triplet loss params
MARGIN = 1.
TRIPLET_LOSS_COEF = 0.01

# Training params
BATCH_SIZE = 256
NUM_EPOCHS = 10
DELTA = 0.5
MODEL_PATH = './model'

# Load the data set
df = pd.read_csv("tweets.csv")
num_classes = df.shape[1] - 1
X_train, X_test, y_train, y_test = train_test_split(df.request.values,
                                                    df.iloc[:,1:].values,
                                                    test_size=0.1,
                                                    stratify=df.iloc[:,1:].values.argmax(axis=1),
                                                    random_state=42)

# Sequences pre-processing
words_encoder = WordsEncoder()
words_encoder.fit(X_train)
X_train = words_encoder.transform(X_train)
X_test = words_encoder.transform(X_test)
vocabulary_size = get_vocabulary_size(X_train)
sequence_length = words_encoder.max_len

# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, sequence_length], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None, num_classes], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    # tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
_, rnn_outputs = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                        inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
# tf.summary.histogram('RNN_outputs', rnn_outputs)
rnn_outputs = tf.concat(rnn_outputs, 1)

# Multi-head layer
heads = []
with tf.name_scope('Multihead_layer'):
    for _ in range(num_classes):
        heads.append(fully_connected(rnn_outputs, HEAD_SIZE))
    heads_concatenated = tf.concat(heads, axis=1)

# Triplet loss
with tf.name_scope("Triplet_loss"):
    triplet_loss = []
    for i, head in enumerate(heads):
        # positive_mask = tf.equal(tf.squeeze(tf.slice(target_ph, [0, i], [-1, i + 1])), 1.)
        positive_mask = tf.equal(target_ph[:, i], 1.)
        negative_mask = tf.logical_not(positive_mask)

        positive_mask.set_shape([None])  # Shape is required by tf.boolean_mask
        negative_mask.set_shape([None])
        anchor = tf.boolean_mask(head, positive_mask)
        negative = tf.boolean_mask(head, negative_mask)
        pos_indices = tf.random_uniform(
            tf.shape(anchor)[0: 1],
            minval=0,
            maxval=tf.shape(anchor)[0],
            dtype=tf.int32
        )
        neg_indices = tf.random_uniform(
            tf.shape(anchor)[0: 1],
            minval=0,
            maxval=tf.shape(negative)[0],
            dtype=tf.int32
        )
        positive = tf.gather(anchor, pos_indices)
        negative = tf.gather(negative, neg_indices)

        distance_positive = tf.norm(tf.subtract(anchor, positive), axis=1)
        distance_negative = tf.norm(tf.subtract(anchor, negative), axis=1)
        # triplet_loss += tf.reduce_mean(tf.maximum(0., MARGIN + distance_positive - distance_negative))
        triplet_loss.append(tf.reduce_mean(tf.maximum(0., MARGIN + distance_positive - distance_negative)))
    triplet_loss = tf.add_n(triplet_loss)

# Dropout
# drop = tf.nn.dropout(attention_output, keep_prob_ph)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    logits = fully_connected(heads_concatenated, num_classes)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_ph))
    tf.summary.scalar('loss', cross_entropy)

    loss = cross_entropy + triplet_loss * TRIPLET_LOSS_COEF

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1),
                                               tf.argmax(target_ph, 1)),
                                      tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(train_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               seq_len_ph: seq_len,
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(test_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
