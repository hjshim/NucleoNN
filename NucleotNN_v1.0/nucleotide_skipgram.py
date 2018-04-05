""" Vector Representations of alleles
NCE loss in TensorFlow and visualizing the embeddings on TensorBoard
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from process_data_echo import process_data

ALL_SIZE = 73            # since 0 is in the vector, length(alleles)+1
BATCH_SIZE = 128
EMBED_SIZE = 128           # Hyper-parameter: dimension of the embedding (distributional) vectors (trials: 300)
SKIP_WINDOW = 36            # Bio-parameter: context (nearby) window size
NUM_SAMPLED = 16           # Hyper-parameter: number of negative samples (trials: 64, 32)
LEARNING_RATE = 1e-2       # Hyper-parameter: learning rate (trials: 1e-9, 1e-6, 1e-3, 1e-1, 1)
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000

class SkipGramModel:
    # Step 1: build the graph for the skip-gram model
    def __init__(self, all_size, embed_size, batch_size, num_sampled, learning_rate):
        self.all_size = all_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        # Step 2: define the placeholders for input and output
        with tf.name_scope("data"):
            self.center_alleles = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_alleles')
            self.target_alleles = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_alleles')

    def _create_embedding(self):
        # Step 3: define weights -> assemble this part of the graph on the CPU
        with tf.device('/cpu:0'):
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.all_size, 
                                                                    self.embed_size], -1.0, 1.0), 
                                                                    name='embed_matrix')

    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # Step 4: define the model
                embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_alleles, name='embed')

                # Step 5: define loss function by constructing variables for NCE loss
                nce_weight = tf.Variable(tf.truncated_normal([self.all_size, self.embed_size],
                                                            stddev=1.0 / (self.embed_size ** 0.5)), 
                                                            name='nce_weight')
                nce_bias = tf.Variable(tf.zeros([ALL_SIZE]), name='nce_bias')

                # Step 6: approximate loss function by NCE
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                    biases=nce_bias, 
                                                    labels=self.target_alleles, 
                                                    inputs=embed, 
                                                    num_sampled=self.num_sampled, 
                                                    num_classes=self.all_size), name='loss')
    # Step 7: define optimizer SGD
    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, 
                                                              global_step=self.global_step)
    
    # Step 8: merge all summaries into one op for easier management
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    # Step 9: build the graph for our skip-gram model
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver() # save all variables (embed_matrix, nce_weight, nce_bias)

    initial_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if that checkpoint exists, restore from that checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0 # calculate average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_train_steps): # python 3
            centers, targets = next(batch_gen) # python 3
            feed_dict={model.center_alleles: centers, model.target_alleles: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], 
                                              feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skip-gram', index)

def main():
    model = SkipGramModel(ALL_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(ALL_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == '__main__':
    main()