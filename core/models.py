# coding:utf-8
import tensorflow as tf

def setup_cell(cell_type, size, num_layers):
  def _get_single_cell():
    return getattr(tf.contrib.rnn, cell_type)(size)
  if num_layers > 1:
    cell = tf.contrib.rnn.MultiRNNCell([_get_single_cell() for _ in range(num_layers)]) 
  else:
    cell = _get_single_cell()
  return cell

class ModelBase(object):
  def __init__(self, sess, config):
    self.sess = sess
    self.max_gradient_norm = config.max_gradient_norm

    with tf.name_scope('global_variables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.learning_rate = tf.train.exponential_decay(
        config.learning_rate, self.global_step,
        config.decay_frequency, config.decay_rate, staircase=True)

  def get_updates(self, loss):
    with tf.name_scope("update"):
      params = tf.contrib.framework.get_trainable_variables()
      opt = tf.train.AdamOptimizer(self.learning_rate)
      gradients = [grad for grad, _ in opt.compute_gradients(loss)]
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
      grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
      updates = opt.apply_gradients(
        grad_and_vars, global_step=self.global_step)
    return updates

  def add_epoch(self):
    self.sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))


class PointerNetwork(ModelBase):
  #def __init__(self, max_len, input_size, size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor):
  def __init__(self, sess, config):
    super(PointerNetwork, self).__init__(sess, config)
    self.hidden_size = config.hidden_size
    self.batch_size = config.batch_size
    self.input_max_len = config.input_max_len
    self.output_max_len = config.output_max_len

    self.encoder_cell = setup_cell(config.cell_type, config.hidden_size, config.num_layers)
    self.decoder_cell = setup_cell(config.cell_type, config.hidden_size, config.num_layers)

    with tf.name_scope('Placeholder'):
      self.encoder_inputs = tf.placeholder(
        tf.float32, [None, config.input_max_len], name="EncoderInput")

      self.decoder_inputs = tf.placeholder(
        tf.float32, [None, config.output_max_len], name="DecoderInput")
