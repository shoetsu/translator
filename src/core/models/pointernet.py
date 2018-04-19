# coding:utf-8
import math, sys, time, copy
import numpy as np
from pprint import pprint

import tensorflow as tf
from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.models import encoder as encoder_class
#from core.models.encoder import SentenceEncoder #WordEncoder, 
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID

def debug_print(variables, message=''):
  sys.stdout = sys.stderr
  print message, variables
  sys.stdout = sys.__stdout__


def setup_decoder(d_outputs_ph, e_inputs_emb, e_state, 
                  attention_states, d_cell, batch_size, 
                  output_max_len, scope=None, teacher_forcing=True):
  # The pointed indexs of encoder's input.

  # A BOS will be appended to the decoder's outputs and prepended to the decoder's inputs.
  # e.g. (target: [3,2,5], weights: [1,1,1])
  # -> (d_input: [0, 3, 2, 5], d_output: [3, 2, 5, 0], d_weights:[1, 1, 1, 1])

  with tf.name_scope('add_bos'):
    d_inputs = tf.concat([tf.zeros([batch_size, 1], dtype=tf.int32), d_outputs_ph], axis=1)
  
  # TODO: for some reason, setting teacher_forcing==True in training makes the performance worse...
  with tf.name_scope('train_decoder'):
    d_outputs, d_states, copied_inputs  = pointer_decoder(
      e_inputs_emb, d_inputs, e_state,
      attention_states, d_cell, scope=scope,
      feed_prev=not teacher_forcing)

  tf.get_variable_scope().reuse_variables()
  with tf.name_scope('test_decoder'):
    predictions, _, _ = pointer_decoder(
      e_inputs_emb, d_inputs, e_state,
      attention_states, d_cell, scope=scope,
      feed_prev=True)
  return d_outputs, predictions, copied_inputs


class PointerNetworkBase(ModelBase):
  def __init__(self, sess, config, vocab, is_training=None):
    ModelBase.__init__(self, sess, config)
    self.vocab = vocab
    self.use_pos = 'pos' in config.features
    self.use_wtype = 'wtype' in config.features
    self.target_columns = config.target_columns

    self.is_training = tf.placeholder(tf.bool, [], name='is_training') if is_training is None else is_training
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

class PointerNetwork(PointerNetworkBase):
  def __init__(self, sess, config, vocab, 
               encoder=None, is_training=None):
    PointerNetworkBase.__init__(self, sess, config, vocab, 
                                is_training=is_training)

    input_max_len, output_max_len = None, config.output_max_len

    # <Sample input>
    # e_inputs: [1, 40, 44, 0, 0], d_outputs: [2, 0, 0] (target=44)
    with tf.name_scope('EncoderInput'):
      self.e_inputs_ph = tf.placeholder(
        tf.int32, [None, input_max_len], name="EncoderInput")
      self.pos_inputs_ph = tf.placeholder(
        tf.int32, [None, input_max_len], name="EncoderInputPOS")
      self.wtype_inputs_ph = tf.placeholder(
        tf.int32, [None, input_max_len], name="EncoderInputWordType")

    with tf.name_scope('batch_size'):
      batch_size = shape(self.e_inputs_ph, 0)

    with tf.variable_scope('Embeddings') as scope:
      e_inputs_emb = []

      w_embeddings = self.initialize_embeddings(
        'Word', vocab.word.embeddings.shape, 
        initializer=tf.constant_initializer(vocab.word.embeddings),
        trainable=config.train_embedding)

      e_inputs_emb.append(tf.nn.embedding_lookup(w_embeddings, self.e_inputs_ph))

      if self.use_pos:
        pos_embeddings = self.initialize_embeddings(
          'POS', [vocab.pos.size, config.feature_size], 
          trainable=True)
        e_inputs_emb.append(tf.nn.embedding_lookup(pos_embeddings, self.pos_inputs_ph))
      if self.use_wtype:
        wtype_embeddings = self.initialize_embeddings(
          'Wtype', [vocab.wtype.size, config.feature_size], 
          trainable=True)
        e_inputs_emb.append(tf.nn.embedding_lookup(wtype_embeddings, self.wtype_inputs_ph))

      e_inputs_emb = tf.concat(e_inputs_emb, axis=-1)
      e_inputs_emb = tf.nn.dropout(e_inputs_emb, self.keep_prob)

    with tf.variable_scope('SentEncoder') as scope:
      # If an encoder is not given, prepare a new one.
      if encoder is None:
        encoder_type = getattr(encoder_class, config.encoder_type)
        sent_encoder = encoder_type(config, self.keep_prob, 
                                    shared_scope=scope)
      else:
        sent_encoder = encoder

      e_inputs_length = tf.count_nonzero(self.e_inputs_ph, axis=1)
      e_outputs, e_state = sent_encoder.encode(
        e_inputs_emb, e_inputs_length)
      attention_states = e_outputs

    self.d_outputs_ph = []
    self.losses = []
    self.greedy_predictions = []
    self.copied_inputs = []
    for i, col_name in enumerate(self.target_columns):
      with tf.name_scope('DecoderOutput%d' % i):
        d_outputs_ph = tf.placeholder(
          tf.int32, [None, output_max_len], name="DecoderOutput")

      ds_name = 'Decoder' if config.share_decoder else 'Decoder%d' % i 
      with tf.variable_scope(ds_name) as scope:
        d_cell = setup_cell(config.cell_type, config.rnn_size, config.num_layers,
                            keep_prob=self.keep_prob)
        teacher_forcing = config.teacher_forcing if 'teacher_forcing' in config else False
        d_outputs, predictions, copied_inputs = setup_decoder(
          d_outputs_ph, e_inputs_emb, e_state, attention_states, d_cell, 
          batch_size, output_max_len, scope=scope, 
          teacher_forcing=teacher_forcing)
        self.copied_inputs.append(copied_inputs)
        d_outputs_length = tf.count_nonzero(d_outputs_ph, axis=1, 
                                            name='outputs_length')
        with tf.name_scope('add_eos'):
          targets = tf.concat([d_outputs_ph, tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1)

        # the length of outputs should be also added by 1 because of EOS. 
        with tf.name_scope('output_weights'):
          d_outputs_weights = tf.sequence_mask(
            d_outputs_length+1, maxlen=shape(d_outputs_ph, 1)+1, dtype=tf.float32)
        with tf.name_scope('loss%d' % i):
          loss = tf.contrib.seq2seq.sequence_loss(
            d_outputs, targets, d_outputs_weights)
      self.d_outputs_ph.append(d_outputs_ph)
      self.losses.append(loss)
      self.greedy_predictions.append(predictions)
    with tf.name_scope('Loss'):
      self.loss = tf.reduce_mean(self.losses)
    self.updates = self.get_updates(self.loss)

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.e_inputs_ph: batch.sources,
      self.is_training: is_training
    }
    if self.use_pos:
      feed_dict[self.pos_inputs_ph] = batch.pos
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
      step_loss, _ = self.sess.run([self.loss, self.updates], feed_dict)
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


class IndependentPointerNetwork(PointerNetwork):
  def __init__(self, sess, config, vocab):
    PointerNetworkBase.__init__(self, sess, config, vocab)
    self.models = []
    for col in self.target_columns:
      with tf.variable_scope(col):
        config.target_columns = [col]
        model = PointerNetwork(sess, config, vocab, 
                               is_training=self.is_training)
        self.models.append(model)
    config.target_columns = self.target_columns
    self.accumulate_models(self.models)

  def accumulate_models(self, models):
    # Accumulated placeholders
    self.e_inputs_ph = [m.e_inputs_ph for m in models]
    self.pos_inputs_ph = [m.pos_inputs_ph for m in models]
    self.wtype_inputs_ph = [m.wtype_inputs_ph for m in models]
    self.d_outputs_ph = [m.d_outputs_ph[0] for m in models]

    self.copied_inputs = [m.copied_inputs[0] for m in models]
    self.greedy_predictions = [m.greedy_predictions[0] for m in models]
    self.loss = tf.reduce_mean([m.loss for m in models], name='loss')
    self.updates = self.get_updates(self.loss)

  def get_input_feed(self, batch, is_training):
    feed_dict = {}
    for k in self.e_inputs_ph:
      feed_dict[k] = batch.sources
    feed_dict[self.is_training] = is_training

    if self.use_pos:
      for k in self.pos_inputs_ph:
        feed_dict[k] = batch.pos
    if self.use_wtype:
      for k in self.wtype_inputs_ph:
        feed_dict[k] = batch.wtype

    for d_outputs_ph, target in zip(self.d_outputs_ph, batch.targets):
      feed_dict[d_outputs_ph] = target
    return feed_dict

class HybridPointerNetwork(IndependentPointerNetwork):
  def __init__(self, sess, config, vocab):
    PointerNetworkBase.__init__(self, sess, config, vocab)
    self.models = []
    with tf.variable_scope('SharedEncoder') as shared_scope:
      encoder_type = getattr(encoder_class, config.encoder_type)
      shared_encoder = encoder_type(config, self.keep_prob, 
                                    shared_scope=shared_scope)

    with tf.variable_scope('MultiEncoderWrapper') as multi_encoder_scope:
      pass
    
    for col in self.target_columns:
      with tf.variable_scope(col):
        config.target_columns = [col]
        with tf.variable_scope('SentEncoder') as scope:
          encoder_type = getattr(encoder_class, config.encoder_type)
          independent_encoder = encoder_type(config, self.keep_prob, 
                                             shared_scope=scope)
        hybrid_encoder = encoder_class.MultiEncoderWrapper(
          [shared_encoder, independent_encoder], config.rnn_size,
          shared_scope=multi_encoder_scope)

        model = PointerNetwork(sess, config, vocab, 
                               encoder=hybrid_encoder,
                               is_training=self.is_training)
        self.models.append(model)
    config.target_columns = self.target_columns
    self.accumulate_models(self.models)
