# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A pointer-network helper.
Based on attenton_decoder implementation from TensorFlow
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys 
from six.moves import xrange # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs

#try:
#  from tensorflow.python.ops.rnn_cell_impl import _linear
#except:
#  from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear

from utils.tf_utils import shape, _linear

def pointer_decoder(encoder_inputs_emb, decoder_inputs, initial_state, 
                    attention_states, cell,
                    feed_prev=True, dtype=dtypes.float32, scope=None):
  encoder_inputs = encoder_inputs_emb
  attn_length = shape(attention_states, 1)
  attn_size = shape(attention_states, 2)

  with tf.name_scope('attention_setup'):
    # Prepare the weights for attention calculation. We assume here the sizes of attention_states (encoder's outputs), encoder's state, decoder's output are same.
    attnw = tf.get_variable("AttnW1", [1, attn_size, attn_size])
    attnw2 = tf.get_variable("AttnW2", [attn_size, attn_size])
    attnv = tf.get_variable("AttnV", [attn_size])

    # Calculate W1 * attention_states in advance since each output and state of encoder is unchanged while decoding.
    attention_states = tf.nn.conv1d(attention_states, attnw, 1, 'SAME')
  sys.stdout = sys.stderr

  def attention_weight(output):
    """
    Calculate attention weights for every encoder's input by taking an inner product the weight bector (attnv) with the conbined and transformed the encoder's output and decoder's state.

    output_probabilities[i] = V・tanh(W1・attention_state[i] + W2・decoder's output[t])
     - i: the index of an input word
     - t: current time-step in decoding
     - v: a tensor with the shape [attention_size]
     - W1: a tensor with the shape [attention_size, encoder's rnn_size]
     - W2: a tensor with the shape [attention_size, decoder's rnn_size]
    """
    y = tf.matmul(output, attnw2)
    y = tf.reshape(y, [-1, 1, attn_size])

    attention_vectors = tf.nn.softmax(tf.reduce_sum(attnv * tf.tanh(attention_states + y), axis=2))
    return attention_vectors

  states = [initial_state]
  outputs = []
  pointed_idxs = []
  with tf.name_scope('Decode_Timestep'):
    for i, d in enumerate(tf.unstack(decoder_inputs, axis=1)):
      with tf.name_scope('Decode_%d' % i):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        # The first input to the decoder is something like _START (or just a _PAD) token we prepared to start decoding.
        pointed_idx = d

        # If feed_prev == True, inputs to decoder won't be used except the first one. The model makes decisions of which should be the next inputs by itself.
        if feed_prev and i > 0:
          # Take argmax to decide which indices of input should be most possible.
          pointed_idx = tf.argmax(output, axis=1, output_type=tf.int32)
        pointed_idxs.append(pointed_idx)
        with tf.name_scope('copy_from_encoder_inputs'):
          # Convert the pointed index into one-hot, and get the pointed encoder_inputs by multiplying and reduce_sum.
          pointed_idx = tf.reshape(tf.one_hot(pointed_idx, depth=attn_length), [-1, attn_length, 1]) 
          inp = tf.reduce_sum(encoder_inputs * pointed_idx, axis=1)

          # In their original paper, the gradients shouldn't be propagated to input embeddings through these copying. The embeddings should be updated only from the encoder.
          inp = tf.stop_gradient(inp)
        output, state = cell(inp, states[-1])

        # Calculate the output (and the next input) distribution 
        with tf.name_scope('attention_weight'):
          output = attention_weight(output)
        states.append(state)
        outputs.append(output)
  with tf.name_scope('outputs'):
    outputs = tf.stack(outputs, axis=1)
  with tf.name_scope('states'):
    states = tf.stack(states, axis=1)
  with tf.name_scope('pointed_idx'):
    pointed_idxs = tf.stack(pointed_idxs, axis=1)
  return outputs, states, pointed_idxs
