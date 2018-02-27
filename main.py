# coding:utf-8

import sys, os, random, copy, collections, time, re, argparse
import pyhocon
import numpy as np
from pprint import pprint
from logging import FileHandler
import tensorflow as tf

from utils import common
from core import models, datasets
from core.vocabularies import WordVocabularyWithEmbedding, _BOS, _PAD

#log_file = args.log_file if args.log_file else None

log_file = None
logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()


class Manager(object):
  def __init__(self, args, sess):
    self.sess = sess
    self.model_path = args.checkpoint_path
    self.summaries_path = self.model_path + '/summaries'
    self.checkpoints_path = self.model_path + '/checkpoints'
    self.tests_path = self.model_path + '/tests'
    self.mode = args.mode
    self.config_path = args.config_path
    self.config = config = self.get_config(args)
    if not args.interactive:
      self.vocab = WordVocabularyWithEmbedding(config.embeddings, vocab_size=config.vocab_size, lowercase=config.lowercase)
      self.model = self.create_model(self.config, self.vocab)
      dataset_type = getattr(datasets, config.dataset_type)
      self.dataset = common.dotDict({
        'train': dataset_type(config.dataset_path.train, self.vocab),
        'test': dataset_type(config.dataset_path.test, self.vocab),
      })
      

  def get_config(self, args):
    # Read and restore config
    config = pyhocon.ConfigFactory.parse_file(self.config_path)
    config_restored_path = os.path.join(self.model_path, 'config')
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)

    if args.cleanup or not os.path.exists(config_restored_path):
      with open(config_restored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    return common.recDotDict(config)

  @common.timewatch(logger)
  def train(self):
    config = self.config
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    for epoch in xrange(self.model.epoch.eval(), self.config.max_epoch):
      train_batches = self.dataset.train.get_batch(config.batch_size, input_max_len=config.input_max_len, output_max_len=config.output_max_len, shuffle=True)
      test_batches = self.dataset.test.get_batch(1, input_max_len=config.input_max_len, output_max_len=config.output_max_len, shuffle=False)

      sys.stdout.write('Epoch %d \n' % (epoch))
      loss = self.model.train(train_batches)
      sys.stdout.write('Train loss: %.3f \n' % (loss))
      #test_inputs, test_targets, predictions = self.model.test(test_batches)
      predictions = self.model.test(test_batches)
      sys.stdout = open(self.tests_path + '/test.%02d.txt' % epoch, 'w')
      for j, (s, t) in enumerate(self.dataset.test.raw_data):
        inp = ' '.join([x for x in s if x not in [_BOS, _PAD]])
        out = ' '.join([x for x in t if x not in [_BOS, _PAD]])
        pred =  [inp[k] for k in predictions[j] if k != 0]
        pred = ' '.join([x for x in pred if x not in [_BOS, _PAD]])
        sys.stdout.write('Test input      %d:\t%s\n' % (j, inp))
        sys.stdout.write('Test output     %d:\t%s\n' % (j, out))
        sys.stdout.write('Test prediction %d:\t%s\n' % (j, pred))
      sys.stdout = sys.__stdout__
      self.model.add_epoch()
      if epoch % 5 == 0:
        self.saver.save(self.sess, checkpoint_path, global_step=self.model.epoch)
    self.saver.save(self.sess, checkpoint_path, global_step=self.model.epoch)

  def create_model(self, config, vocab, checkpoint_path=None):
    m = getattr(models, config.model_type)(self.sess, config, vocab)

    if not checkpoint_path:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      logger.info("Reading model parameters from %s" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      logger.info("Created model with fresh parameters.")
      self.sess.run(tf.global_variables_initializer())

    variables_path = self.model_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      f.write('\n'.join(variable_names) + '\n')

    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                self.sess.graph)
    self.reuse = True
    return m


def main(args):
  random.seed(0)
  np.random.seed(0)
  tf_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True, # GPU上で実行できない演算を自動でCPUに
    gpu_options=tf.GPUOptions(
      allow_growth=True, # True->必要になったら確保, False->全部
    )
  )

  with tf.Graph().as_default(), tf.Session(config=tf_config).as_default() as sess:
    tf.set_random_seed(0)
    manager = Manager(args, sess)
    if args.mode == 'train':
      manager.train()
    else:
      pass
  return manager

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_path")
  parser.add_argument("mode")
  parser.add_argument("config_path")
  parser.add_argument("--cleanup", default=False, type=common.str2bool)
  parser.add_argument("--debug", default=False, type=common.str2bool)
  parser.add_argument("--log_file", default=None)
  parser.add_argument("--interactive", default=False)
  #parser.add_argument("-d", "--debug", default=False, type=common.str2bool)
  args  = parser.parse_args()
  main(args)

