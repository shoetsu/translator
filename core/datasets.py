#coding:utf-8
import tensorflow as tf
import sys, re, random, itertools, collections
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID
from utils.common import dotDict, timewatch
from utils import evaluation

class DatasetBase(object):
  pass

text_header = 'sentence'
label_header = 'label'

class PriceDataset(DatasetBase):
  NONE = '-'
  @timewatch()
  def __init__(self, path, vocab, num_train_data=0):
    sys.stderr.write('Loading dataset from %s ...\n' % (path))
    data = pd.read_csv(path)
    if num_train_data:
      data = data[:num_train_data]
    text_data = data[text_header]
    label_data = data[label_header]
    self.vocab = vocab
    self.tokenizer = vocab.tokenizer

    # For copying, keep unnormalized sentences too.
    self.original_sources = [[_BOS] + self.tokenizer(l, normalize_digits=False) for l in text_data]
    self.sources = [[_BOS] + self.tokenizer(l) for l in text_data]
    targets = self.preprocess_target(label_data)
    lowerbounds, l_equals, upperbounds, u_equals, currencies, rates = zip(*targets)
    self.targets = [lowerbounds, upperbounds, currencies, rates] 
    self.targets = list(zip(*self.targets))
    self.targets_name = ['LB', 'UB', 'Currency', 'Rate']
    
  def get_batch(self, batch_size, input_max_len=None, output_max_len=None, shuffle=False):
    sources, targets = self.symbolized
    if input_max_len:
      paired = [(s,t) for s,t in zip(sources, targets) if not len(s) > input_max_len ]
      sources, targets = list(zip(*paired))

    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    targets = list(zip(*targets)) # to column-major. (for padding)
    targets = [tf.keras.preprocessing.sequence.pad_sequences(targets_by_column, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID) for targets_by_column in targets]
    targets = list(zip(*targets)) # to idx-major. (for shuffling)

    data = [(s,t) for s,t in zip(sources, targets)]

    if shuffle:
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      b_sources, b_targets = zip(*batch)
      b_targets = list(zip(*b_targets)) # to column-major.
      yield dotDict({
        'sources': np.array(b_sources),
        'targets': [np.array(t) for t in b_targets],
      })
  
  def create_demo_batch(self, text, output_max_len):
    source = [self.vocab.tokens2ids(text)]
    targets = [[[0] for _ in self.targets_name]]
    targets = list(zip(*targets)) # to column-major. (for padding)
    source =  tf.keras.preprocessing.sequence.pad_sequences(source, padding='post', truncating='post', value=PAD_ID)
    targets = [tf.keras.preprocessing.sequence.pad_sequences(targets_by_column, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID) for targets_by_column in targets]
    
    yield dotDict({
      'sources': np.array(source),
      'targets': [np.array(t) for t in targets],
    })

  @property 
  def raw_data(self):
    return self.original_sources, self.targets

  @property
  def symbolized(self):
    # Find the indice of the tokens in a source sentence which was copied as a target label.
    targets = [[[o.index(x) for x in tt if x != self.NONE and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]
    #targets = list(zip(*targets))
    sources = [self.vocab.tokens2ids(s) for s in self.sources]
    #return list(zip(sources, targets))
    return sources, targets

  @property
  def tensorized(self):
    sources, targets = self.symbolized
    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=config.input_max_len+1, padding='post', truncating='post', value=BOS_ID)
    targets =  tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=config.output_max_len, padding='post', truncating='post', value=BOS_ID)
    sources = tf.data.Dataset.from_tensor_slices(sources)
    targets = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((sources, targets))
    return dataset

  def preprocess_target(self, data):
    def find_entry(l):
      return re.search('\((.+)\)',l.split(':')[0]).group(1).split('|')
    return [[self.tokenizer(x.strip(), normalize_digits=False) for x in find_entry(l)] for l in data]


  def show_results(self, predictions, verbose=True):
    test_inputs_tokens, _ = self.symbolized
    golds = []
    preds = []
    for i, ((raw_s, raw_t), p) in enumerate(zip(zip(*self.raw_data), predictions)):
      token_inp = self.vocab.ids2tokens(test_inputs_tokens[i])
      inp, gold, pred = self.ids_to_tokens(raw_s, raw_t, p)
      golds.append(gold)
      preds.append(pred)
      succ_or_fail = 'EM_Success' if gold == pred else "EM_Failure"
      if verbose:
        sys.stdout.write('<%d> (%s)\n' % (i, succ_or_fail))
        sys.stdout.write('Test input       :\t%s\n' % (' '.join(inp)))
        sys.stdout.write('Test input (unk) :\t%s\n' % (' '.join(token_inp)))
        sys.stdout.write('Human label      :\t%s\n' % (' | '.join(gold)))
        sys.stdout.write('Test prediction  :\t%s\n' % (' | '.join(pred)))
    EM = evaluation.exact_match(golds, preds)
    precisions, recalls = evaluation.precision_recall(golds, preds)

    res = {'Metrics': ['EM accuracy', 'Precision', 'Recall']}
    for col, col_name in zip(zip(EM, precisions, recalls), self.targets_name):
      res[col_name] = col

    df = pd.DataFrame(res)
    df = df.ix[:,['Metrics'] + self.targets_name].set_index('Metrics')
    print df
    return df

  def ids_to_tokens(self, s_tokens, t_tokens, p_idxs):
    outs = []
    preds = []
    inp = [x for x in s_tokens if x not in [_BOS, _PAD]]
    if t_tokens is not None:
      for tt in t_tokens:
        out = ' '.join([x for x in tt if x not in [_BOS, _PAD]])
        outs.append(out)
    for pp in p_idxs:
      #print pp
      pred =  [s_tokens[k] for k in pp if len(s_tokens) > k]
      pred = ' '.join([x for x in pred if x not in [_BOS, _PAD]])
      if not pred:
        pred = self.NONE
      preds.append(pred)
    return inp, outs, preds

    
