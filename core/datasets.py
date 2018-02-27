#coding:utf-8
import tensorflow as tf
import sys, re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
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
    self.original_sources = [self.tokenizer(l, normalize_digits=False) for l in data['Text']]
    self.sources = [self.tokenizer(l) for l in data['Text']]
    targets = self.preprocess_target(data['Values'])
    lowerbounds, l_equals, upperbounds, u_equals, currencies, rates = zip(*targets)
    self.targets = lowerbounds #upperbounds

  @property
  def tensorized(self):
    return (self.sources, self.targets)
    pass
  #   print self.sources[0]
  #   print 
  #   ss = [vocab.tokens2ids(s) for s in self.sources][:3]
  #   print tf.keras.preprocessing.sequence.pad_sequences(ss)
  #   #print tf.data.Dataset.from_tensor_slices(ss, ss)
  # #def raw2tensor(self, origin, source, target):
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
