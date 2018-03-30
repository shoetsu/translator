# coding:utf-8

import sys, os, random, copy, collections, time, re, argparse
import pyhocon
import numpy as np
from pprint import pprint
from logging import FileHandler
import tensorflow as tf

from utils import common, evaluation, tf_utils
from core import models, datasets
from core.vocabularies import WordVocabularyWithEmbedding, _BOS, _PAD, _NUM, _UNIT

tf_config = tf.ConfigProto(
  log_device_placement=False, # If True, all the placement of variables will be logged. 
  allow_soft_placement=True, # If True, it uses CPU instead of GPU to avoid causing errors if the specified GPU is occupied or not exists.
  gpu_options=tf.GPUOptions(
    allow_growth=True, # If False, all memories of the GPU will be occupied.
  )
)
#The default config for old models which don't have some recently added hyperparameters. (to be abolished in future)
default_config = common.recDotDict({
  'share_decoder': False,
  'target_columns': ['LB', 'UB', 'Unit', 'Rate'],
  'normalize_digits': False,
})

class Manager(object):
  @common.timewatch()
  def __init__(self, args, sess, vocab=None):
    self.sess = sess
    self.config = self.load_config(args)
    self.mode = args.mode
    self.logger = common.logManager(handler=FileHandler(args.log_file)) if args.log_file else common.logManager()

    sys.stderr.write(str(self.config) + '\n')

    if True or not args.interactive:
      self.vocab = WordVocabularyWithEmbedding(
        self.config.embeddings, 
        vocab_size=self.config.vocab_size, 
        lowercase=self.config.lowercase,
        normalize_digits=self.config.normalize_digits,
        num_init_embedding_type=self.config.num_init_embedding_type,
        unit_init_embedding_type=self.config.unit_init_embedding_type
      ) if vocab is None else vocab

      self.dataset = getattr(datasets, self.config.dataset_type)(
        self.config.dataset_type, self.config.dataset_path, 
        self.config.num_train_data, self.vocab,
        self.config.target_attribute, self.config.target_columns)

  def load_config(self, args):
    '''
    Load the config specified by args.config_path. The config will be copied into args.checkpoint_path if there is no config there. We can overwrite a few hyperparameters in the config and being listed up on the bottom of this file by specifying as arguments when runnning this code.
     e.g.
        ./run.sh checkpoints/tmp test --batch_size=30
    '''
    self.model_path = args.checkpoint_path
    self.summaries_path = self.model_path + '/summaries'
    self.checkpoints_path = self.model_path + '/checkpoints'
    self.tests_path = self.model_path + '/tests'
    self.config_path = args.config_path if args.config_path else self.model_path + '/config'

    # Read and restore config if there is no existing config in the checkpoint.
    sys.stderr.write('Reading a config from %s ...\n' % (self.config_path))
    config = pyhocon.ConfigFactory.parse_file(self.config_path)
    config_restored_path = os.path.join(self.model_path, 'config')
    if not os.path.exists(self.summaries_path):
      os.makedirs(self.summaries_path)
    if not os.path.exists(self.checkpoints_path):
      os.makedirs(self.checkpoints_path)
    if not os.path.exists(self.tests_path):
      os.makedirs(self.tests_path)

    # Overwrite configs by temporary args. They have higher priorities than those in the config of models.
    if 'dataset_type' in args and args.dataset_type:
      config['dataset_type'] = args.dataset_type
    if 'train_data_path' in args and args.train_data_path:
      config['dataset_path']['train'] = args.train_data_path
    if 'test_data_path' in args and args.test_data_path:
      config['dataset_path']['test'] = args.test_data_path
    if 'vocab_size' in args and args.vocab_size:
      config['vocab_size'] = args.vocab_size
    if 'batch_size' in args and args.batch_size:
      config['batch_size'] = args.batch_size
    if 'target_attribute' in args and args.target_attribute:
      config['target_attribute'] = args.target_attribute

    # The restored confing in the checkpoint will be overwritten with the argument --cleanup=True.
    if args.cleanup or not os.path.exists(config_restored_path):
      sys.stderr.write('Restore the config to %s ...\n' % (config_restored_path))

      with open(config_restored_path, 'w') as f:
        sys.stdout = f
        common.print_config(config)
        sys.stdout = sys.__stdout__
    config = common.recDotDict(config)

    # The default config for old models which don't have some recently added hyperparameters will be overwritten if a model has the corresponding hyperparameters.
    default_config.update(config)
    config = default_config

    print config
    return config

  def save_model(self, model, save_as_best=False):
    '''
    Restore the trained model. 
    Args: 
      - model: The model object created by create_model().
      - save_as_best: If True, the model in this epoch is copied as 'best' and will be used in testing.
    '''
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
      sys.stderr.write('Epoch %d start training ...\n' % (epoch))
    
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
    model = self.create_model(
      self.sess, self.config, self.vocab,)
      #checkpoint_path=self.checkpoints_path + '/model.ckpt.best')
    embeddings = self.sess.run(model.models[0].w_embeddings)
    
    num_vec = embeddings[self.vocab.token2id(_NUM)]
    unit_vec = embeddings[self.vocab.token2id(_UNIT)]
    num_vec2 = embeddings[self.vocab.token2id('_num')]
    unit_vec2 = embeddings[self.vocab.token2id('_unit')]
    def calc_similarity(vec):
      res = []
      for i, v in enumerate(embeddings):
        sim = common.cosine_similarity(vec, v)
        res.append((i, sim))
      return sorted(res, key=lambda x: -x[1])

    N=20
    for v in [num_vec, unit_vec, num_vec2, unit_vec2]:
      print [(self.vocab.id2token(_id), sim) for _id, sim in calc_similarity(v)][:N]
    print len(embeddings)
    print self.vocab.size
    print self.vocab.rev_vocab[:20]
    a = []
    for w in self.vocab.rev_vocab[20:]:
      e = embeddings[self.vocab.token2id(w)]
      print '-------'
      print w, np.linalg.norm(e)
      if np.linalg.norm(e) < 1:
        a.append(w)
    print a
    import math
    e= np.random.uniform(-math.sqrt(3), math.sqrt(3), 
                         size=300)
    print np.linalg.norm(e)
      #print e
    pass

  def demo(self, model=None, inp=None):
    if model is None:
      model = self.create_model(
        self.sess, self.config, self.vocab, 
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')

    def decode(origin_inp):
      origin_inp = [origin_inp]
      demo_data = datasets.create_demo_batch(
        origin_inp, self.config.dataset_type, self.vocab, 
        self.config.target_columns)
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
      #dataset = self.dataset.valid

    _, test_filename = common.separate_path_and_filename(
      dataset.path)

    if model is None: 
      model = self.create_model(
        self.sess, self.config, self.vocab, 
        checkpoint_path=self.checkpoints_path + '/model.ckpt.best')
 
    test_filename = '%s.%02d' % (test_filename, model.epoch.eval()) if in_training else '%s.best' % (test_filename)
    output_types = None

    batches = dataset.get_batch(
      config.batch_size, input_max_len=None, 
      output_max_len=config.output_max_len, shuffle=False)
    predictions = model.test(batches)
    index, sources, targets = dataset.raw_data
    predictions = [datasets.find_token_from_sentence(p, s) for p,s in zip(predictions, sources)]

    test_output_path = os.path.join(self.tests_path, test_filename)
    df, scalar_summary = dataset.show_results(
      sources, targets, predictions, verbose=verbose, 
      target_path_prefix=test_output_path, output_types=output_types)
    if in_training:
      self.summary_writer.add_summary(scalar_summary, model.epoch.eval())
    #else:
    #  self.summary_writer.add_summary(text_summary, model.epoch.eval())
    
    return df

  @common.timewatch()
  def create_model(self, sess, config, vocab, 
                   checkpoint_path=None, cleanup=False):
    ''' 
    The latest checkpoint is automatically loaded if there is, otherwise create a model with flesh parameters.
    '''
    # Instatiate the model class specified by config.model_type and define the computation graph. This must be done before running tf.global_variables_initializer().
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      m = getattr(models, config.model_type)(sess, config, vocab)

    if not checkpoint_path and not cleanup:
      ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)
      checkpoint_path = ckpt.model_checkpoint_path if ckpt else None


    # Load or create from scratch.
    self.saver = tf.train.Saver(tf.global_variables(), 
                                max_to_keep=config.max_to_keep)
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
      sys.stderr.write("Reading model parameters from %s\n" % checkpoint_path)
      self.saver.restore(sess, checkpoint_path)
    else:
      sys.stderr.write("Created model with fresh parameters.\n")
      sess.run(tf.global_variables_initializer())

    # List up all the defined variables and their shapes.
    variables_path = self.model_path + '/variables.list'
    with open(variables_path, 'w') as f:
      variable_names = sorted([v.name + ' ' + str(v.get_shape()) for v in tf.global_variables()])
      f.write('\n'.join(variable_names) + '\n')

    self.summary_writer = tf.summary.FileWriter(self.summaries_path, 
                                                sess.graph)
    return m


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
  
  parser.add_argument("--cleanup", default=False, type=common.str2bool)
  parser.add_argument("--interactive", default=False, type=common.str2bool)
  parser.add_argument("--log_file", default=None, type=str)

  # Arguments that can be dynamically overwritten if they're not None.
  parser.add_argument("--dataset_type", default=None, type=str)
  parser.add_argument("--train_data_path", default=None, type=str)
  parser.add_argument("--test_data_path", default=None, type=str)
  parser.add_argument("--batch_size", default=None, type=int)
  parser.add_argument("--vocab_size", default=None, type=int)
  parser.add_argument("--target_attribute", default=None, type=str)
  args  = parser.parse_args()
  main(args)

