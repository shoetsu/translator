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
  print 'encoder_inputs',encoder_inputs_emb
  print 'decoder_inputs', decoder_inputs
  print 'attention_states', attention_states
  encoder_inputs = encoder_inputs_emb
  #attn_length = attention_states.get_shape()[1].value
  #attn_size = attention_states.get_shape()[2].value
  attn_length = shape(attention_states, 1)
  attn_size = shape(attention_states, 2)
  attnw = tf.get_variable("AttnW", [1, attn_size, attn_size])
  attention_states = tf.nn.conv1d(attention_states, attnw, 1, 'SAME')
  attnv = tf.get_variable("AttnV", [attn_size])

  def attention_weight(output):
    print 'attention_states', attention_states
    print 'output', output
    y = _linear(output, attn_size, True)
    y = tf.reshape(y, [-1, 1, attn_size])
    attention_vectors = tf.nn.softmax(tf.reduce_sum(attnv * tf.tanh(y + attention_states), axis=2))
    return attention_vectors

  states = [initial_state]
  outputs = []
  inputs = []

  for i, d in enumerate(tf.unstack(decoder_inputs, axis=1)):
  #for i in tf.range(0, shape(decoder_inputs, 1)):
    #d = decoder_inputs[i]
    print 'i=%d'%i
    if i > 0:
      tf.get_variable_scope().reuse_variables()
    pointed_idx = d
    # in testing, inputs to decoder won't be used except the first one.
    if feed_prev and i > 0:
      # take argmax, convert the pointed index into one-hot, and get the pointed encoder_inputs by multiplying and reduce_sum.
      pointed_idx =  tf.argmax(output, axis=1)
        
    pointed_idx = tf.reshape(tf.one_hot(pointed_idx, depth=attn_length), [-1, attn_length, 1]) 
    inp = tf.reduce_sum(encoder_inputs * pointed_idx, axis=1) 
    print 'pointed_idx',pointed_idx
    print 'next_inp', inp
    output, state = cell(inp, states[-1])
    output = attention_weight(output)
    print 'output', output
    inputs.append(inp)
    states.append(state)
    outputs.append(output)
  outputs = tf.stack(outputs, axis=1)
  states = tf.stack(states, axis=1)
  return outputs, states #, inputs
