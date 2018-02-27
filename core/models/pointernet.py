# coding:utf-8
import math, sys
sys.stdout = sys.stderr
import tensorflow as tf

from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID


class PointerNetwork(ModelBase):
  def __init__(self, sess, config, vocab):
    ModelBase.__init__(self, sess, config)

    self.cell_type = config.cell_type
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers

    self.e_cell = setup_cell(self.cell_type, self.hidden_size, self.num_layers)
    self.d_cell = setup_cell(self.cell_type, self.hidden_size, self.num_layers)
    with tf.name_scope('Placeholder'):
      batch_size = None
      input_max_len, output_max_len = None, config.output_max_len
      self.is_training = tf.placeholder(tf.bool, [], name='is_training')

      # The input word ids.
      self.e_inputs = tf.placeholder(
        tf.int32, [batch_size, input_max_len], name="EncoderInput")
      self.e_inputs_length = tf.placeholder(
        tf.int32, [batch_size], name='EncoderInputLength')

      # The pointed indexs of encoder's input.
      self.d_outputs = tf.placeholder(
        tf.int32, [batch_size, output_max_len], name="DecoderOutput")
      self.d_outputs_length = tf.placeholder(
        tf.int32, [batch_size], name='DecoderOutputLength') # only in training.


      
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

    self.e_inputs_emb = tf.nn.embedding_lookup(
      self.w_embeddings, self.e_inputs)

    with tf.variable_scope('Encoder') as scope:
      e_outputs, e_state = tf.nn.dynamic_rnn(
        self.e_cell, self.e_inputs_emb,
        sequence_length=self.e_inputs_length, scope=scope, dtype=tf.float32)
    attention_states = e_outputs
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

  def get_input_feed(self, batch):
    pass
    
  def train(self, data):
    pass

  def test(self, data):
    pass
