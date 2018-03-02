# coding:utf-8
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.models.encoder import WordEncoder, SentenceEncoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID

def setup_decoder(e_inputs_emb, e_state, attention_states, d_cell, batch_size, 
                  output_max_len, scope=None):
  # The pointed indexs of encoder's input.
  d_outputs_ph = tf.placeholder(
    tf.int32, [None, output_max_len], name="DecoderOutput")

  # A BOS will be appended to the decoder's outputs and prepended to the decoder's inputs.
  # e.g. (target: [3,2,5], weights: [1,1,1])
  # -> (d_input: [0, 3, 2, 5], d_output: [3, 2, 5, 0], d_weights:[1, 1, 1, 1])

  d_inputs = tf.concat([tf.zeros([batch_size, 1], dtype=tf.int32), d_outputs_ph], axis=1)
  
  d_outputs, d_states  = pointer_decoder(
    e_inputs_emb, d_inputs, e_state,
    attention_states, d_cell, scope=scope,
    feed_prev=False)

  tf.get_variable_scope().reuse_variables()
  predictions, _ = pointer_decoder(
    e_inputs_emb, d_inputs, e_state,
    attention_states, d_cell, scope=scope,
    feed_prev=True)
  return d_outputs_ph, d_outputs, predictions

class PointerNetwork(ModelBase):
  def __init__(self, sess, conf, vocab):
    ModelBase.__init__(self, sess, conf)
    self.vocab = vocab
    input_max_len, output_max_len = None, conf.output_max_len
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    self.keep_prob = 1.0 - tf.to_float(self.is_training) * conf.dropout_rate

    # <Sample input to be fed>
    # e_inputs: [1, 40, 44, 0, 0], d_outputs: [2, 0, 0] (target=44)
    # The input word ids.
    self.e_inputs_ph = tf.placeholder(
      tf.int32, [None, input_max_len], name="EncoderInput")
    self.batch_size = shape(self.e_inputs_ph, 0)

    with tf.variable_scope('Embeddings') as scope:
      self.w_embeddings = self.initialize_embeddings(
        'Word', vocab.embeddings.shape, 
        initializer=tf.constant_initializer(vocab.embeddings),
        trainable=conf.train_embedding)

    with tf.variable_scope('WordEncoder') as scope:
      self.word_encoder = WordEncoder(conf, self.w_embeddings, self.keep_prob,
                                      shared_scope=scope)
      self.e_inputs_emb = self.word_encoder.encode([self.e_inputs_ph])

    with tf.variable_scope('SentEncoder') as scope:
      self.sent_encoder = SentenceEncoder(conf, self.keep_prob, 
                                          shared_scope=scope)
      e_inputs_length = tf.count_nonzero(self.e_inputs_ph, axis=1)
      e_outputs, e_state = self.sent_encoder.encode(
        self.e_inputs_emb, e_inputs_length)
      attention_states = e_outputs

    self.d_outputs_ph = []
    self.losses = []
    self.greedy_predictions = []

    for i in range(conf.num_columns):
      with tf.variable_scope('Decoder%d' % i) as scope:
        d_cell = setup_cell(conf.cell_type, conf.hidden_size, conf.num_layers)
        d_outputs_ph, d_outputs, predictions = setup_decoder(
          self.e_inputs_emb, e_state, attention_states, d_cell, 
          self.batch_size, output_max_len, scope=scope)
        d_outputs_length = tf.count_nonzero(d_outputs_ph, axis=1)
        targets = tf.concat([d_outputs_ph, tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
        # the length of outputs should be added by 1 because of EOS. 
        d_outputs_weights = tf.sequence_mask(
          d_outputs_length+1, maxlen=shape(d_outputs_ph, 1)+1, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
          d_outputs, targets, d_outputs_weights)
      self.d_outputs_ph.append(d_outputs_ph)
      self.losses.append(loss)
      self.greedy_predictions.append(predictions)
    self.loss = tf.reduce_mean(self.losses)
    self.updates = self.get_updates(self.loss)

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.e_inputs_ph: batch.sources,
      self.is_training: is_training
    }
    for d_outputs_ph, target in zip(self.d_outputs_ph, batch.targets):
      feed_dict[d_outputs_ph] = target
    return feed_dict

  def debug(self, data):
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, True)
      for k,v in feed_dict.items():
        print k, v
      break
    return

  def train(self, data):
    loss = 0.0
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, True)
      t = time.time()
      step_loss, _ = self.sess.run([self.losses, self.updates], feed_dict)
      step_loss = np.mean(step_loss)
      epoch_time += time.time() - t
      loss += math.exp(step_loss)
      num_steps += 1
    loss /= num_steps
    return loss, epoch_time

  def test(self, data):
    inputs = []
    outputs = []
    predictions = []
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      predictions_dist = self.sess.run(self.greedy_predictions, feed_dict)
      batch_predictions = [np.argmax(dist, axis=2) for dist in predictions_dist]
      predictions.append(batch_predictions)
    predictions = [np.concatenate(column_pred, axis=0) for column_pred in zip(*predictions)]
    predictions = list(zip(*predictions))
    return predictions
