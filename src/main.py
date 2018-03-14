# coding:utf-8

import sys, os, random, copy, collections, time, re, argparse
import pyhocon
import numpy as np
from pprint import pprint
from logging import FileHandler
import tensorflow as tf

from utils import common, evaluation, tf_utils
from core import models, datasets
from core.vocabularies import WordVocabularyWithEmbedding, _BOS, _PAD

tf_config = tf.ConfigProto(
  log_device_placement=False,
  allow_soft_placement=True, 
  gpu_options=tf.GPUOptions(
    allow_growth=True, # If False, all memories of the GPU will be occupied.
  )
)
default_config = common.recDotDict({
  'share_encoder': True,
  'share_decoder': False,
})

class Manager(object):
  @common.timewatch()
  def __init__(self, args, sess, vocab=None):
    self.sess = sess
    self.config = self.get_config(args)
    self.mode = args.mode
    self.logger = common.logManager(handler=FileHandler(args.log_file)) if args.log_file else common.logManager()

    sys.stderr.write(str(self.config) + '\n')

    self.dataset_type = getattr(datasets, self.config.dataset_type)
    if not args.interactive:
      self.vocab = WordVocabularyWithEmbedding(
        self.config.embeddings, 
        vocab_size=self.config.vocab_size, 
        lowercase=self.config.lowercase) if vocab is None else vocab
      self.dataset = self.dataset_type(
        self.config.dataset_path, self.vocab,
        num_train_data=self.config.num_train_data, no_train=args.mode!='train')



  def get_config(self, args):
    self.model_path = args.checkpoint_path
    self.summaries_path = self.model_path + '/summaries'
    self.checkpoints_path = self.model_path + '/checkpoints'
    self.tests_path = self.model_path + '/tests'
    self.config_path = args.config_path if args.config_path else self.model_path + '/config'

    # Read and restore config
    sys.stderr.write('Reading a config from %s ...\n' % (self.config_path))
    config = pyhocon.ConfigFactory.parse_file(self.config_path)
    config_restored_path = os.path.join(self.model_path, 'config')
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)

    if args.cleanup or not os.path.exists(config_restored_path):
      sys.stderr.write('Restore the config to %s ...\n' % (config_restored_path))

      with open(config_restored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    config = common.recDotDict(config)

    default_config.update(config)
    config = default_config

    # Override configs by temporary args.
    if 'test_data_path' and args.test_data_path:
      config.dataset_path.test = args.test_data_path
    if 'batch_size' in args and args.batch_size:
      config.batch_size = args.batch_size
    if 'debug' in args:
      config.debug = args.debug
    return config

  def save_model(self, model, save_as_best=False):
    checkpoint_path = self.checkpoints_path + '/model.ckpt'
    self.saver.save(self.sess, checkpoint_path, global_step=model.epoch)
    if save_as_best:
      suffixes = ['data-00000-of-00001', 'index', 'meta']
      for s in suffixes:
        source_path = self.checkpoints_path + "/model.ckpt-%d.%s" % (model.epoch.eval(), s)
        target_path = self.checkpoints_path + "/model.ckpt.best.%s" % (s)
        cmd = "cp %s %s" % (source_path, target_path)
        os.system(cmd)

  @common.timewatch()
  def train(self, model=None):
    if model is None:
      model = self.create_model(
        self.sess, self.config, self.vocab)
    testing_results = []
    for epoch in xrange(model.epoch.eval(), self.config.max_epoch):
      train_batches = self.dataset.train.get_batch(
        self.config.batch_size, input_max_len=self.config.input_max_len, 
        output_max_len=self.config.output_max_len, shuffle=True)

      loss, epoch_time = model.train(train_batches)
      summary = tf_utils.make_summary({
        'loss': loss
      })
      self.summary_writer.add_summary(summary, model.epoch.eval())

      self.logger.info('(Epoch %d) Train loss: %.3f (%.1f sec)' % (epoch, loss, epoch_time))
      df = self.test(model=model, dataset=self.dataset.valid,
                     in_training=True)
      average_accuracy = np.mean(df.values.tolist()[0])
      if len(testing_results) == 0 or average_accuracy > max(testing_results):
        save_as_best = True
        best_epoch = epoch
        best_result = average_accuracy
        self.logger.info('(Epoch %d) Update highest accuracy: %.3f' % (best_epoch, best_result))
      else:
        save_as_best = False
      testing_results.append(average_accuracy)
      self.save_model(model, save_as_best=save_as_best)
      model.add_epoch()
    return

  def debug(self):
    pass

  def demo(self, model=None, inp=None):
    if model is None:
      model = self.create_model(
        self.sess, self.config, self.vocab, 
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')

    def decode(origin_inp):
      origin_inp = [origin_inp]
      demo_data = datasets.create_demo_batch(
        origin_inp, self.dataset_type, self.vocab)
      batch = demo_data.get_batch(1, output_max_len=self.config.output_max_len, 
                                  shuffle=False) 
      predictions = model.test(batch)
      origin_inp = demo_data.original_sources[0]
      normalized_inp, _ = demo_data.symbolized
      normalized_inp = demo_data.vocab.ids2tokens(normalized_inp[0])
      pred = datasets.find_token_from_sentence(predictions[0], origin_inp)
      inp = datasets.remove_special_tokens(origin_inp)
      sys.stdout.write("Source      :\t%s\n" % ' '.join(inp))
      sys.stdout.write("Source(unk) :\t%s\n" % ' '.join(normalized_inp))
      sys.stdout.write("Target      :\t%s\n" % ' | '.join(pred))

    if inp:
      decode(inp)
    else:
      while True:
        sys.stdout.write('Input: ')
        inp = sys.stdin.readline().replace('\n', '')
        decode(inp)

  def test(self, model=None, dataset=None, verbose=True, in_training=False):
    config = self.config
    if dataset is None:
      dataset = self.dataset.test

    _, test_filename = common.separate_path_and_filename(
      self.config.dataset_path.test)

    if model is None: 
      model = self.create_model(
        self.sess, self.config, self.vocab, 
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')
 
    test_filename = '%s.%02d' % (test_filename, model.epoch.eval()) if in_training else '%s.best' % (test_filename)
    #output_types = ['overall'] if in_training else None 
    output_types = None

    batches = dataset.get_batch(
      config.batch_size, input_max_len=None, 
      output_max_len=config.output_max_len, shuffle=False)
    predictions = model.test(batches)
    epoch = model.epoch.eval()
    index, sources, targets = dataset.raw_data

    test_output_path = os.path.join(self.tests_path, test_filename)
    df, summary = dataset.show_results(sources, targets, predictions, 
                                       verbose=verbose, 
                                       target_path_prefix=test_output_path,
                                       output_types=output_types)
    if in_training:
      self.summary_writer.add_summary(summary, model.epoch.eval())
    return df

  @common.timewatch()
  def create_model(self, sess, config, vocab, 
                   checkpoint_path=None, cleanup=False):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      m = getattr(models, config.model_type)(sess, config, vocab)

    if not checkpoint_path and not cleanup:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
                                #max_to_keep=config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stderr.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(sess, checkpoint_path)
    else:
      sys.stderr.write("Created model with fresh parameters.\n")
      sess.run(tf.global_variables_initializer())

    variables_path = self.model_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      f.write('\n'.join(variable_names) + '\n')

    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                sess.graph)
    return m

  def evaluate(self):
    dataset = self.dataset_type(
      self.config.dataset_path.test, self.vocab)
    index, sources, targets = dataset.raw_data
    predictions = []

    import pandas as pd
    for l in pd.read_csv(args.evaluate_data_path).values.tolist():
      idx, _, lb, ub, cur, rate = l
      if idx not in index:
        continue
      else:
        predictions.append([lb, ub, cur, rate])
    df, _ = dataset.show_results(sources, targets, predictions, 
                                 prediction_is_index=False)
    print df

def main(args):
  random.seed(0)
  np.random.seed(0)
  with tf.Graph().as_default(), tf.Session(config=tf_config).as_default() as sess:
    tf.set_random_seed(0)
    manager = Manager(args, sess)
    if args.mode == 'train':
      manager.train()
    elif args.mode == 'test':
      manager.test()
    elif args.mode == 'evaluate':
      manager.evaluate()
    elif args.mode == 'demo':
      manager.demo()
    elif args.mode == 'debug':
      manager.debug()
    else:
      raise ValueError('args.mode must be \'train\', \'test\', or \'demo\'.')

  if args.mode == 'train':
    vocab = manager.vocab
    with tf.Graph().as_default(), tf.Session(config=tf_config).as_default() as sess:
      tf.set_random_seed(0)
      manager = Manager(args, sess, vocab=vocab)
      manager.test()
  return manager

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_path")
  parser.add_argument("mode")
  parser.add_argument("config_path")
  
  parser.add_argument("--debug", default=False, type=common.str2bool)
  parser.add_argument("--cleanup", default=False, type=common.str2bool)
  parser.add_argument("--interactive", default=False, type=common.str2bool)
  parser.add_argument("--log_file", default=None, type=str)
  parser.add_argument("--test_data_path", default=None, type=str)
  parser.add_argument("--evaluate_data_path", default='dataset/baseline.complicated.csv', type=str)
  parser.add_argument("--batch_size", default=None, type=int)
  args  = parser.parse_args()
  main(args)

