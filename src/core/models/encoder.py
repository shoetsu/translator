# coding: utf-8 
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.util import nest
from core.models import setup_cell
from utils.tf_utils import linear

# def merge_state(state):
#   if isinstance(state[0], LSTMStateTuple):
#     new_c = tf.concat([s.c for s in state], axis=1)
#     new_h = tf.concat([s.h for s in state], axis=1)
#     state = LSTMStateTuple(c=new_c, h=new_h)
#   else:
#     state = tf.concat(state, 1)
#   return state

def merge_state(state, rnn_size, activation=tf.nn.tanh):
  """
  This function assumes that the state is an output from 'tf.nn.bidirectional_dynamic_rnn' i.e. state = (fw_state, bw_state). the state can also be a nested tuple such as state = ((fw_state_0, fw_state_1, ...), (bw_state_0, bw_state_1)) if our RNN has multiple layers. 
  """

  if not type(state) == tuple:
    raise ValueError
  if isinstance(state[0], LSTMStateTuple):
    raise NotImplementedError

  # In the function linear(), two states from the forward and backward RNN (both states have the shape of [batch_size, rnn_state]) are combined and transformed into the tensor with the shape of [batch_size, rnn_state] to make the encoder's and the decoder's state size equal.

  if type(state[0]) == tuple: # num_layers >= 2
    new_state = []
    for fs, bs in zip(*state):
      ns = tf.concat([fs, bs], axis=-1)
      if rnn_size is not None:
        ns = linear(ns, rnn_size, activation=activation)
      new_state.append(ns)
    new_state = tuple(new_state)
  else:
    new_state = tf.concat(state, 1)
    new_state = linear(new_state, rnn_size, activation=activation)
  return new_state


class RNNEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.tanh, shared_scope=None):
    self.rnn_size = config.rnn_size
    self.keep_prob = keep_prob
    self.activation = activation
    self.shared_scope = shared_scope
    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.cell_type, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, input_embeddings, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
      outputs, state = tf.nn.dynamic_rnn(
        self.cell_fw, input_embeddings,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
    return outputs, state

class BidirectionalRNNEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.tanh, shared_scope=None):
    self.rnn_size = config.rnn_size
    self.keep_prob = keep_prob
    self.activation = activation
    self.shared_scope = shared_scope
    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(config.cell_type, config.rnn_size, 
                                num_layers=config.num_layers, 
                                keep_prob=self.keep_prob)
    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(
        config.cell_type, config.rnn_size, 
        num_layers=config.num_layers, keep_prob=self.keep_prob)

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, input_embeddings, sequence_length):
    with tf.variable_scope(self.shared_scope or "SentenceEncoder") as scope:
      outputs, state = tf.nn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, input_embeddings,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
      with tf.variable_scope("outputs"):
        outputs = tf.concat(outputs, 2)
        outputs = linear(outputs, self.rnn_size)

      with tf.variable_scope("state"):
        state = merge_state(state, self.rnn_size)
    return outputs, state


class MultiEncoderWrapper(object):
  def __init__(self, encoders, rnn_size, shared_scope=None):
    """
    Args 
      encoders: a list of SentenceEncoder. 
    """
    self.encoders = encoders
    self.rnn_size = rnn_size
    self.shared_scope = shared_scope

  def encode(self, input_embeddings, sequence_length):
    with tf.variable_scope(self.shared_scope or "MultiEncoderWrapper") as scope:
      outputs, state = zip(*[e.encode(input_embeddings, sequence_length) for e in self.encoders])
      with tf.variable_scope('outputs'):
        outputs = tf.concat(outputs, 2)
        outputs = linear(outputs, self.rnn_size)
      with tf.variable_scope('state'):
        state = merge_state(state, self.rnn_size)
    return outputs, state
