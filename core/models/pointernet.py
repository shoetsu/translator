# coding:utf-8
import math, sys
import numpy as np
from pprint import pprint
sys.stdout = sys.stderr

import tensorflow as tf
from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID


def greedy_decode(predictions):
  pass


class PointerNetwork(ModelBase):
  def __init__(self, sess, conf, vocab):
    ModelBase.__init__(self, sess, conf)
    self.vocab = vocab
    self.e_cell = setup_cell(conf.cell_type, conf.hidden_size, conf.num_layers)
    self.d_cell = setup_cell(conf.cell_type, conf.hidden_size, conf.num_layers)
    with tf.name_scope('Placeholder'):
      batch_size = None
      input_max_len, output_max_len = None, conf.output_max_len
      self.is_training = tf.placeholder(tf.bool, [], name='is_training')
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * conf.dropout_rate

      # <Sample input to be fed>
      # e_inputs: [1, 40, 44, 0, 0], d_outputs: [2, 0, 0] (target=44)

      # The input word ids.
      self.e_inputs = tf.placeholder(
        tf.int32, [batch_size, input_max_len], name="EncoderInput")
      self.e_inputs_length = tf.count_nonzero(self.e_inputs, axis=1)
      #self.e_inputs_length = tf.placeholder(
      #  tf.int32, [batch_size], name='EncoderInputLength')

      # The pointed indexs of encoder's input.
      self.d_outputs = tf.placeholder(
        tf.int32, [batch_size, output_max_len], name="DecoderOutput")
      self.d_outputs_length = tf.count_nonzero(self.d_outputs, axis=1)
      #self.d_outputs_length = tf.placeholder(
      #  tf.int32, [batch_size], name='DecoderOutputLength') # only in training.

      self.batch_size = shape(self.e_inputs, 0)

      # A BOS will be appended to the decoder's outputs and prepended to the decoder's inputs.
      # e.g. (target: [3,2,5], weights: [1,1,1])
      # -> (d_input: [0, 3, 2, 5], d_output: [3, 2, 5, 0], d_weights:[1, 1, 1, 1])
      self.d_inputs = tf.concat([tf.zeros([self.batch_size, 1], dtype=tf.int32), self.d_outputs], axis=1)
      self.targets = tf.concat([self.d_outputs, tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
      self.d_outputs_weights = tf.sequence_mask(
        self.d_outputs_length+1, maxlen=shape(self.d_outputs, 1)+1, dtype=tf.float32)

    with tf.variable_scope('Embeddings') as scope:
      self.w_embeddings = self.initialize_embeddings(
        'Word', vocab.embeddings.shape, 
        initializer=tf.constant_initializer(vocab.embeddings))

    self.e_inputs_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.w_embeddings, self.e_inputs), self.keep_prob)

    with tf.variable_scope('Encoder') as scope:
      e_outputs, e_state = tf.nn.dynamic_rnn(
        self.e_cell, self.e_inputs_emb,
        sequence_length=self.e_inputs_length, scope=scope, dtype=tf.float32)
    attention_states = outputs = tf.nn.dropout(e_outputs, self.keep_prob)
    print 'attention_states', attention_states
    ################################################
    # Need a dummy to point on it. End of decoding.
    # self.e_inputs_emb = tf.nn.embedding_lookup(
    #   self.w_embeddings, 
    #   tf.concat([BOS_ID * tf.ones([self.batch_size, 1], dtype=tf.int32), 
    #              self.e_inputs], axis=1)
    # )
    #attention_states = tf.concat([tf.zeros([self.batch_size, 1, self.hidden_size]), e_outputs], axis=1)
    # self.e_inputs_emb = tf.nn.embedding_lookup(
    #   self.w_embeddings, 
    #   tf.concat([BOS_ID * tf.ones([self.batch_size, 1], dtype=tf.int32), 
    #              self.e_inputs], axis=1)
    # )
    #################################################
    self.e_state = e_state
    self.attention_states = attention_states

    with tf.variable_scope('Decoder') as scope:
      d_outputs, d_states  = pointer_decoder(
        self.e_inputs_emb, self.d_inputs, e_state,
        attention_states, self.d_cell, scope=scope)

    with tf.variable_scope('Decoder', reuse=True) as scope:
      predictions, _ = pointer_decoder(
        self.e_inputs_emb, self.d_inputs, e_state,
        attention_states, self.d_cell, scope=scope,
        feed_prev=True)

    self.greedy_predictions = predictions
    self.outputs = d_outputs

    self.loss = tf.contrib.seq2seq.sequence_loss(
      d_outputs, self.targets, self.d_outputs_weights)
    self.updates = self.get_updates(self.loss)

  def debug(self, data):
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, True)
      res = self.sess.run([self.e_inputs, self.e_inputs_length, self.d_outputs, self.d_outputs_length, self.d_inputs, self.d_outputs_weights, self.targets], feed_dict)
      titles = ['e_inputs', 'e_inputs_len', 'd_outputs','d_outputs_len', 'd_inputs', 'd_weights', 'targets']
      for t, r in zip(titles, res):
        print (t, r)
      exit(1)

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.e_inputs: batch.sources,
      self.d_outputs: batch.targets,
      self.is_training: is_training
    }
    return feed_dict
    
  def train(self, data):
    #self.debug(data)
    #exit(1)
    loss = 0.0
    num_steps = 0
    for i, batch in enumerate(data):
      
      feed_dict = self.get_input_feed(batch, True)
      step_loss, _ = self.sess.run([self.loss, self.updates], feed_dict)
      loss += math.exp(step_loss)
      num_steps += 1
    loss /= num_steps
    return loss

  def test(self, data):
    inputs = []
    outputs = []
    predictions = []
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      inputs.append(feed_dict[self.e_inputs])
      outputs.append(feed_dict[self.d_outputs])
      predictions_dist = self.sess.run(self.greedy_predictions, feed_dict)
      predictions.append(np.argmax(predictions_dist, axis=2))
    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return predictions
