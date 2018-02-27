#coding:utf-8
import tensorflow as tf
import sys, re, random, itertools
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID
from utils.common import dotDict
sys.stdout=sys.stderr

class DatasetBase(object):
  pass

class PriceDataset(DatasetBase):
  def __init__(self, path, vocab):
    data = pd.read_csv(path)
    self.vocab = vocab
    self.tokenizer = vocab.tokenizer
    #self.tokenizer = get_word_tokenizer(True, False)
    # For copying, keep unnormalized sentences too.
    self.texts = data['Text']
    self.values = data['Values']
    self.original_sources = [[_BOS] + self.tokenizer(l, normalize_digits=False) for l in data['Text']]
    self.sources = [[_BOS] + self.tokenizer(l) for l in data['Text']]
    targets = self.preprocess_target(data['Values'])
    lowerbounds, l_equals, upperbounds, u_equals, currencies, rates = zip(*targets)
    self.targets = lowerbounds #upperbounds

  def get_batch(self, batch_size, input_max_len=None, output_max_len=None, shuffle=False):
    input_max_len = input_max_len+1 if input_max_len else input_max_len 
    sources, targets = self.symbolized
    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    targets =  tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID)
    data = [(s,t) for s,t in zip(sources, targets)]
    if shuffle:
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      b_sources, b_targets = zip(*batch)
      yield dotDict({
        'sources': np.array(b_sources),
        'targets': np.array(b_targets),
      })

  @property
  def tensorized(self):
    sources, targets = self.symbolized
    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=config.input_max_len+1, padding='post', truncating='post', value=BOS_ID)
    targets =  tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=config.output_max_len, padding='post', truncating='post', value=BOS_ID)
    sources = infer_dataset = tf.data.Dataset.from_tensor_slices(sources)
    targets = infer_dataset = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((sources, targets))

  @property 
  def raw_data(self):
    return zip(self.original_sources, self.targets)
  

  @property
  def symbolized(self):
    # Find the indice of the tokens in a source sentence which was copied as a target label.
    targets = [[o.index(x) for x in t if x != '-' and x in o ] for o, t in zip(self.original_sources, self.targets)]
    sources = [self.vocab.tokens2ids(s) for s in self.sources]
    #return list(zip(sources, targets))
    return sources, targets

  def preprocess_target(self, data):
    def find_entry(l):
      return re.search('\((.+)\)',l.split(':')[0]).group(1).split('|')
    return [[self.tokenizer(x.strip(), normalize_digits=False) for x in find_entry(l)] for l in data]

  def tokens2indexs(self, source, target):
    pass
