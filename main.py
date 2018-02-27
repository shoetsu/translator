# coding:utf-8
import sys, os, random, copy, collections, time, re, argparse
import pyhocon
import numpy as np
from pprint import pprint
from logging import FileHandler
import tensorflow as tf

from utils import common
from core import models, datasets
from core.vocabularies import WordVocabularyWithEmbedding

class Manager(object):
  def __init__(self, args, sess):
    self.sess = sess
    log_file = args.log_file if args.log_file else None
    self.logger = common.logManager(handler=FileHandler(log_file)) if log_file else common.logManager()
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
      #dataset_type = getattr(datasets, config.dataset_type)
      #self.dataset = dataset_type(config.dataset_path, self.vocab)

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

  def train(self):
    pass

  def create_model(self, config, vocab, checkpoint_path=None):
    m = getattr(models, config.model_type)(self.sess, config, vocab)

    if not checkpoint_path:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      self.logger.info("Reading model parameters from %s" % checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
    else:
      self.logger.info("Created model with fresh parameters.")
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

