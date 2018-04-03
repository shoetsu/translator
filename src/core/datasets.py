#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _PAD, PAD_ID, _BOS, BOS_ID, _EOS, _UNK, _NUM, _UNIT, FeatureVocab
from utils import evaluation, tf_utils, common
import datasets as self_module

EMPTY = '-' # token for empty label.

def postprocess(tokens):
  """
  Args:
    - tokens : A list of words.
  """
  # Postprocessing for normalizing numbers. (e.g. 40|million -> 40 Million)
  def separate_concatenated_tokens(tokens): 
    return common.flatten([x.split('|') for x in tokens])

  def remove_special_tokens(tokens):
    special_tokens = [_BOS, _PAD]
    return [x for x in tokens if x not in special_tokens]

  tokens = separate_concatenated_tokens(tokens)
  tokens = remove_special_tokens(tokens)
  return tokens

def find_token_from_sentence(p_idxs, s_tokens):
  '''
  Args:
   p_idxs: List of indices. (e.g. [[3,4], [3,4], [2], []])
   s_tokens: List of string. (e.g. ['it', 'costs', '$', '4', 'millions', '.'])
  '''
  sys.stdout = sys.stderr
  preds = []
  for pp in p_idxs:
    pred =  [s_tokens[k] for k in pp if len(s_tokens) > k]
    pred = ' '.join([x for x in pred if x not in [_BOS, _PAD]])
    if not pred:
      pred = EMPTY
    preds.append(pred)
  return preds


def create_demo_batch(sentences, dataset_type, vocab, 
                      attribute_name, target_columns, tmp_path='/tmp'):
  '''
  Args:
    sentences: List of string.
  '''
  # Create a temporary file.
  tmp_path = os.path.join(tmp_path, common.random_string(5))
  index = [i for i in xrange(len(sentences))]
  dic = {
    'index': index,
    'sentence': sentences,
  }
  for col in target_columns:
    dic[col] = [EMPTY]
  df = pd.DataFrame(dic).ix[:, ['index', 'sentence'] + target_columns].set_index('index')
  
  sys.stdout = sys.stderr
  with open(tmp_path, 'w') as f:
    f.write(df.to_csv() + '\n')
  pathes = common.dotDict({'train': tmp_path, 'valid':tmp_path, 'test':tmp_path})

  num_training_sentences = 0 # Fake value.
  dataset = getattr(self_module, dataset_type)(dataset_type, pathes, num_training_sentences, vocab, attribute_name, target_columns)
  dataset.test.load_data()
  os.system('rm %s' % tmp_path)
  return dataset.test

class DatasetBase(object):
  pass

class _PriceDataset(DatasetBase):
  def __init__(self, data_path, vocab, target_attribute, target_columns, 
               num_lines=0):
    self.path = data_path
    self.num_lines = num_lines 
    self.target_attribute = target_attribute
    self.target_columns = target_columns
    self.vocab = vocab

    self.indices = []
    self.original_sources = []
    self.sources = []
    self.targets = []
    self.all_columns = []

  def get_pos(self, texts, path=None):
    pos = common.get_pos(texts, output_path=path)
    pos = [[_BOS] + p[1:] for p in pos] # The POS of _BOS should be _BOS.
    return pos

  def load_data(self):
    # lazy loading to save time.
    sys.stderr.write('Loading dataset from %s ...\n' % (self.path))
    data = pd.read_csv(self.path).fillna(EMPTY)

    if self.num_lines:
      data = data[:self.num_lines]
  
    text_data = data['sentence']
    # For copying, keep unnormalized sentences too.
    self.original_sources = [[_BOS] + self.vocab.word.tokenizer(l, normalize_digits=False) for l in text_data]
    self.indices = data['index'].values
    self.sources = [[_BOS] + self.vocab.word.tokenizer(l) for l in text_data]

    self.all_columns = [x for x in data.columns if x not in ['index', 'sentence']]
    if not self.target_columns:
      self.target_columns = self.all_columns
    for c in self.target_columns:
      if not c in self.all_columns:
        raise ValueError('The name of column must be in the label columns of data. (\'%s\' not found in %s)' % (c, str(self.all_columns)))
    self.targets = [[self.vocab.word.tokenizer(x, normalize_digits=False) for x in data[col].values] for col in self.all_columns]
    self.targets = list(zip(*self.targets)) # to batch-major.

  def get_batch_data(self, input_max_len, output_max_len):
    '''
    '''
    sources, targets = self.symbolized
    if input_max_len:
      paired = [(s,t) for s,t in zip(sources, targets) if not len(s) > input_max_len]
      sources, targets = list(zip(*paired))

    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    targets = list(zip(*targets)) # to column-major. (for padding)
    targets = [tf.keras.preprocessing.sequence.pad_sequences(targets_by_column, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID) for targets_by_column in targets]
    targets = list(zip(*targets)) # to idx-major. (for shuffling)

    data = sources, targets, self.original_sources
    return data

  def yield_batch(self, batch):
    '''
    Args
      - batch: A list of a list containing 'batch_size' examples (specified as an argument to get_batch()), batch[i] contains each of the return values of get_batch_data().   (i.e. the shape of 'batch' = [len(self.get_batch_data(...)), batch_size]).

    Return : A batch as a dictionary.
    '''
    b_sources, b_targets, b_ori_sources = batch
    b_targets = list(zip(*b_targets)) # to column-major.

    return common.dotDict({
      'sources': np.array(b_sources),
      # Include only the labels in 'target_columns' to batch.
      'targets': [np.array(t) for t, col in zip(b_targets, self.all_columns) if col in self.target_columns],
      'original_sources': b_ori_sources,
    })

  def get_batch(self, batch_size,
                input_max_len=None, output_max_len=None, shuffle=False):
    if not self.sources:
      self.load_data() # lazy loading.

    # get_batch_data() and yield_batch() can be overwritten in the child to feed extra inputs. 
    data = self.get_batch_data(input_max_len=input_max_len,
                               output_max_len=output_max_len)
    data = list(zip(*data))
    if shuffle: # For training.
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      b = [x[1] for x in b] # remove 'i'.
      yield self.yield_batch(zip(*b))

  @property 
  def raw_data(self):
    return self.indices, self.original_sources, self.targets

  def manual_replace(self, s):
    # For easy inheritance.
    return s

  @property 
  def symbolized(self):
    # Find the indice of the tokens in a source sentence which was copied as a target label.
    targets = [[[o.index(x) for x in tt if x != EMPTY and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]
    sources = [self.vocab.word.tokens2ids(self.manual_replace(s)) for s in self.sources]
    return sources, targets

  # @property
  # def tensorized(self):
  #   sources, targets = self.symbolized
  #   sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=config.input_max_len+1, padding='post', truncating='post', value=BOS_ID)
  #   targets =  tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=config.output_max_len, padding='post', truncating='post', value=BOS_ID)
  #   sources = tf.data.Dataset.from_tensor_slices(sources)
  #   targets = tf.data.Dataset.from_tensor_slices(targets)
  #   dataset = tf.data.Dataset.zip((sources, targets))
  #   return dataset

  def show_results(self, original_sources, targets, predictions, 
                   verbose=True, 
                   target_path_prefix=None, output_types=None):
    """
      Args:
       - original_sources : Separated words in input sentences. 
         e.g. [['it', 'costs', '$' '30'], ... ]
       - targets : Human labels. Values of each column must be an list of str.
         e.g. [(['2', 'million'], ['2', 'million'], ['$'], ['year']), ...]
       - predictions : Predictions. Same as targets.
         e.g. [(['2'], ['2', 'million'], ['$'], ['-']), ...]
    """
    if type(original_sources[0]) == str:
      original_sources = [s.split() for s in original_sources]
    if type(targets[0][0]) == str:
      targets = [[c.split() for c in t] for t in targets]
    if type(predictions[0][0]) == str:
      predictions = [[c.split() for c in p] for p in predictions]
    golds = []
    preds = []
    try:
      assert len(original_sources) == len(targets) == len(predictions)
    except:
      print "The lengthes of sentences, human labels, predictions must be same. (%d, %d, %d)" % (len(original_sources), len(targets), len(predictions))
      print predictions
      exit(1)

    inputs = [] # Original texts.

    # Normalized texts with _UNK.
    if self.vocab:
      test_inputs_tokens, _ = self.symbolized
      token_inputs = [self.vocab.word.ids2tokens(x) for x in test_inputs_tokens]
    else:
      token_inputs = [[] for _ in range(len(original_sources))]


    for i, (s, t, p) in enumerate(zip(original_sources, targets, predictions)):
      inp = postprocess(s)
      gold = OrderedDict({col:' '.join(postprocess(t)) for col, t in zip(self.all_columns, t)})
      #if prediction_is_index:
      pred = OrderedDict({col:' '.join(postprocess(p)) for col, p in zip(self.target_columns, p)})
      inputs.append(inp)
      golds.append(gold)
      preds.append(pred)

    ###############################################################
    '''
    Functions freely defined to analyze results of complicated targets.
    (name, (func1, func2))
    func1 : A function to decide whether an example will be imported in the results.
    <Args>
       - g : A dictionary of gold labels (e.g. {'LB':30, 'UB':'40', 'Currency':'$', 'Rate':'-'})
    <Returns>  True or False.

    func2 : A function to decide an extraction to be a success or not.
    <Args>
       - g : same as func1.
       - p : A dictionary of predicted labels. Note that only the label of 'self.target_columns' is contained. (e.g. {'LB':30, 'UB':'30'})
    <Returns>  True or False.
    '''
    ###############################################################

    lower_upper_success = lambda g, p: ('LB' not in p or g['LB'] == p['LB']) and ('UB' not in p or g['UB'] == p['UB'])
    exact_cond = lambda g: g['LB'] == g['UB'] and g['LB'] != EMPTY
    range_cond = lambda g: g['LB'] != g['UB'] and g['LB'] != EMPTY and g['UB'] != EMPTY
    rate_cond = lambda g: g['Rate'] != EMPTY
    multi_cond = lambda g: len(g['LB'].split(' ')) > 1 or len(g['UB'].split(' ')) > 1
    less_cond = lambda g: g['LB'] == '-' and g['UB'] != '-'
    more_cond = lambda g: g['LB'] != '-' and g['UB'] == '-'

    all_success = lambda g, p: tuple(g.values()) == tuple(p.values())
    rate_success = lambda g, p: 'Rate' not in p or g['Rate'] == p['Rate']
    conditions = [
      ('overall', (lambda g: True, all_success)),
      ('exact', (exact_cond, lower_upper_success)),
      ('multi', (multi_cond, lower_upper_success)),
      ('range', (range_cond, lower_upper_success)),
      #('range_more_less', (lambda g: range_cond(g) or less_cond(g) or more_cond(g),lower_upper_success)),
      ('less', (less_cond, lower_upper_success)),
      ('more', (more_cond, lower_upper_success)),
      ('rate', (rate_cond, rate_success)),
      # ('exact_rate', (lambda g: exact_cond(g) and rate_cond(g), all_success)),
      # ('exact_multi', (lambda g: exact_cond(g) and multi_cond(g), all_success)),
      # ('range_rate', (lambda g: range_cond(g) and rate_cond(g), all_success)),
      # ('range_multi', (lambda g: range_cond(g) and multi_cond(g), all_success)),
      # ('more_rate', (lambda g: more_cond(g) and rate_cond(g), all_success)),
      # ('more_multi', (lambda g: more_cond(g) and multi_cond(g), all_success)),
      # ('less_rate', (lambda g: less_cond(g) and rate_cond(g), all_success)),
      # ('less_multi', (lambda g: less_cond(g) and multi_cond(g), all_success)),
    ]
    if output_types is not None:
      conditions = [c for c in conditions if c[0] in output_types]

    df_sums = []
    df_rows = []
    summaries = []
    for (name, cf) in conditions:
      with open("%s.%s" % (target_path_prefix, name), 'w') as f:
        sys.stdout = f
        df_sum, df_row = self.summarize(inputs, token_inputs, golds, preds, cf)
        sys.stdout = sys.__stdout__
      df_sums.append(df_sum)
      df_rows.append(df_row)

    df_sum_all = df_sums[0]
    df_row_all = df_rows[0]

    if output_types is None: 
      # Reshape results of testing for visualization.
      names = [c[0] for c in conditions[1:]]
      indices = [df_row.index.tolist() for df_row in df_rows[1:]]
      types = [[] for i in df_row_all.index.tolist()]
      for name, idx in zip(names, indices):
        for i in idx:
          types[i].append(name)
      types = [' '.join(x) if x else EMPTY for x in types]
      df_row_all['type'] = types
      header = df_row_all.columns.values.tolist()
      df_row_all =  df_row_all.ix[:, ['type'] + header[:-1]]

      df_em_all = [df.values.tolist()[0] for df in df_sums]
      df_em_all = list(zip(*df_em_all))
      header = ['type'] + df_sum_all.columns.tolist()
      types = tuple([c[0] for c in conditions])
      df_em_all = [types] + df_em_all
      df_em_all = pd.DataFrame({k:v for k,v in zip(header, df_em_all)}).ix[:, header].set_index('type')
      with open("%s.%s" % (target_path_prefix, 'summary.csv'), 'w') as f:
        sys.stdout = f
        print (df_row_all.to_csv())
        print (df_em_all.to_csv())
        sys.stdout = sys.__stdout__

    # Make Summary object for Tensorboard.
    type_names = [name for (name, _) in conditions]
    summary_dict = {}
    for type_name, df in zip(type_names, df_sums):
      col_names = df.columns.tolist()
      EM_rates = df.values.tolist()[0]
      for col_name, val in zip(col_names, EM_rates):
        k = "%s/%s" % (type_name, col_name)
        summary_dict[k] = val
    scalar_summary = tf_utils.make_summary(summary_dict)
    
    # header = ['sentence', 'human', 'prediction']
    # text_summary = [tf.summary.text(h, tf.convert_to_tensor(df_row_all[h].tolist())) for h in header]
    # text_summary = tf.summary.merge(text_summary)
    #   #  pass
    return df_sums[0], scalar_summary #, text_summary 

  def summarize(self, inputs, token_inputs, _golds, _preds, cf, 
                output_as_csv=False):
    '''
    Args:
     inputs: 2D array of string in original sources. [len(data), max_source_length]
     token_inputs: 2D array of string. Tokens not in the vocabulary are converted into _UNK. [len(data), max_source_length]
     _golds: 1D List of dictionary. Each dictionary must have the name of column as key, and a label string as value.
     _preds:  1D List of dictionary. Each dictionary must have the name of column as key, and a label string as value. 
     cf: Tuple of two functions. cf[0] is for the condition whether it includes a line of test into the summary, and cf[1] is to decide whether a pair of tuple (gold, prediction) for a line of testing is successful of not. 
    '''
    golds, preds = [], []
    res = []
    for i, (inp, token_inp, g, p) in enumerate(zip(inputs, token_inputs, _golds, _preds)):
      if not cf[0](g):
        continue
      #is_success = cf[1](g, p)
      is_success = OrderedDict([(col, g[col] == p[col]) for col in self.target_columns])
      
      # Output only the labels in 'target_columns'.
      g = [g[col] for col in self.target_columns] 
      p = [p[col] for col in self.target_columns] 
      golds.append(g)
      preds.append(p)
      res.append([i, is_success, ' '.join(inp), ' '.join(token_inp), ' | '.join(g), ' | '.join(p)])

    # Print results
    header = ['index', 'success', 'sentence', 'tokenized_sent', 'human', 'prediction']
    df_dict = {k:v for k, v in zip(header, list(zip(*res)))}
    output_header = ['index', 'success', 'sentence', 'human', 'prediction']
    df_row = pd.DataFrame(df_dict)
    df_row = df_row.ix[:, output_header].set_index('index')

    EM = evaluation.exact_match(golds, preds)
    precisions, recalls = evaluation.precision_recall(golds, preds)

    df_dict = {'Metrics': ['EM accuracy', 'Precision', 'Recall']}
    for col, col_name in zip(zip(EM, precisions, recalls), self.target_columns):
      df_dict[col_name] = col

    df_sum = pd.DataFrame(df_dict)
    df_sum = df_sum.ix[:,['Metrics'] + self.target_columns].set_index('Metrics')

    if output_as_csv:
      print (df_row.to_csv())
      print ('')
      print (df_sum.to_csv())
    else:
      for i, is_success, inp, token_inp, g, p in res:
        #succ_or_fail = 'EM_Success' if is_success else "EM_Failure"
        succ_or_fail = ['%s_Success' % k if v else '%s_Failure' % k for k,v in is_success.items()] 
        succ_or_fail = ', '.join(succ_or_fail)
        sys.stdout.write('<%d> (%s)\n' % (i, succ_or_fail))
        sys.stdout.write('Test input       :\t%s\n' % (inp))
        if token_inp:
          sys.stdout.write('Test input (unk) :\t%s\n' % (token_inp))
        sys.stdout.write('Human label      :\t%s\n' % (g))
        sys.stdout.write('Test prediction  :\t%s\n' % (p))
      print ('')
      print (df_sum)

    return df_sum, df_row

class _NumNormalizedPriceDataset(_PriceDataset):
  # def __init__(self, path, vocab, target_columns, num_lines=0):
  #   _PriceDataset.__init__(self, path, vocab, num_lines, target_columns)

  def load_data(self):
    _PriceDataset.load_data(self)
    self.original_sources, self.targets, self.num_indices = self.concat_numbers(
      self.original_sources, self.targets, self.pos 
    )
    self.sources = self.original_sources

  def manual_replace(self, s, idx):
    # Replace the token on a certain index to 0. These indices are those of the tokens with 'CD' POS.
    return [x if i not in idx else _NUM for i, x in enumerate(s)]

  @property 
  def symbolized(self):
    targets = [[[o.index(x) for x in tt if x != EMPTY and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]
    sources = [self.vocab.word.tokens2ids(self.manual_replace(s, idx)) for s, idx in zip(self.sources, self.num_indices)]
    return sources, targets

  def concat_numbers(self, _sources, _targets, _pos):
    # Concatenate consecutive CDs in sources and targets by '|'.
    # 'num_indices' shows the concatenated tokens and they will be normalized into '0' when being input as an embedding.
    delim = '|'
    sources = []
    targets = []
    num_indices = []
    for s, t, p in zip(_sources, _targets, _pos):
      try:
        assert len(s) == len(p)
      except:
        sys.stderr.write('Source:' + ' '.join(s) + '\n')
        sys.stderr.write('POS:' + ' '.join(p) + '\n')
        raise Exception('The sources and their POS must be same. (%d != %d)' % (len(s), len(p)))
      new_s = []
      tmp = []
      idx = []
      for i in xrange(len(s)):
        if p[i] == 'CD' and s[i] not in ['(', ')', EMPTY]:
          tmp.append(s[i])
        else:
          if len(tmp) > 0:
            concated_num = delim.join(tmp)
            new_s.append(concated_num)
            tmp = []
            idx.append(new_s.index(concated_num))
          new_s.append(s[i])
      if len(tmp) > 0:
        new_s.append(delim.join(tmp))
        idx.append(i-1)
      new_t = ([delim.join(t[0])], [delim.join(t[1])], t[2], t[3])
      sources.append(new_s)
      targets.append(new_t)
      num_indices.append(idx)
    return sources, targets, num_indices


# Dictionaries for manual replacement.
def unit_normalize(s, target_attribute):
  normalized_name = _UNIT
  if target_attribute.lower() == 'price':
    unit_names = ['yen', 'dollar', 'euro', 'franc', 'pound', 'cent', 'buck']
    unit_names += [x+'s' for x in unit_names]
    unit_symbols = ['$', '₡', '£', '¥','₦', '₩', '₫', '₪', '₭', '€', '₮', '₱', '₲', '₴', '₹', '₸', '₺', '₽', '฿',]
  elif target_attribute.lower() == 'weight':
    unit_names = ['kg', 'g', 'gram', 'grams', 'pounds', 'tons', 'ounce', 'lb', 'lbs', 'gallon', 'gallons']
    unit_symbols = []
  else:
    raise ValueError('\'args.target_attribute\' must be in the list [\'Price\', \'Weight\']. (It is \'%s\' now.)' % target_attribute)

  s = [x if x not in unit_names else normalized_name for x in s]
  s = [x if x not in unit_symbols else normalized_name for x in s]
  return s

def attribute_normalize(s, target_attribute):
  normalized_name = 'attribute'
  if target_attribute == 'Price':
    attribute_names = ['price']
    attribute_names += [x + 's' for x in attribute_names]
  elif target_attribute == 'Weight':
    attribute_names = ['weight']
    attribute_names += [x + 's' for x in attribute_names]

  s = [x if x not in attribute_names else normalized_name for x in s]
  return s

class _UnitNormalizedPriceDataset(_PriceDataset):
  def manual_replace(self, s):
    return unit_normalize(s, self.target_attribute)


class _AllNormalizedPriceDataset(_NumNormalizedPriceDataset):
  def manual_replace(self, s, idx):
    s = _NumNormalizedPriceDataset.manual_replace(self, s, idx)
    s = unit_normalize(s, self.target_attribute)
    s = attribute_normalize(s, self.target_attribute)
    return s


class _PriceDatasetWithFeatures(_PriceDataset):
  def __init__(self, data_path, vocab, 
               target_attribute, target_columns, 
               num_lines=0):
    _PriceDataset.__init__(self, data_path, vocab, target_attribute, 
                           target_columns, num_lines=num_lines)
  def load_data(self):
    super(_PriceDatasetWithFeatures, self).load_data()
    self.pos = self.get_pos(self.original_sources, self.path)

  def get_batch_data(self, input_max_len, output_max_len):
    # TODO:
    data = _PriceDataset.get_batch_data(self, input_max_len, output_max_len)
    pos = [self.vocab.pos.tokens2ids(p) for p in self.pos]
    pos = tf.keras.preprocessing.sequence.pad_sequences(pos, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    return data + tuple([pos])

  def yield_batch(self, batch_by_column):
    b_sources, b_targets, b_ori_sources, b_pos= batch_by_column
    b_targets = list(zip(*b_targets)) # to column-major.
    return common.dotDict({
      'sources': np.array(b_sources),
      # Include only the labels in 'target_columns' to batch.
      'targets': [np.array(t) for t, col in zip(b_targets, self.all_columns) 
                  if col in self.target_columns],
      'original_sources': b_ori_sources,
      'pos': b_pos,
    })


###################################################
#    Classes for dataset pair (train, valid, test)
###################################################

class PackedDatasetBase(object):
  '''
  The class contains train, valid, test dataset.
  Each dataset class has different types of .
  args:
     dataset_type: A string. It is the name of dataset class defined in config.
     pathes: A list of string. ([train_path, valid_path, test_path])
  kwargs:
     num_train_data: The upperbound of the number of training examples. If 0, all of the data will be used.
     no_train: whether to omit to load training data to save time. (in testing)
  '''
  @common.timewatch()
  def __init__(self, dataset_type, pathes, num_train_data,
               *args, **kwargs):
    self.dataset_type = getattr(self_module, '_' + dataset_type)
    train_path, valid_path, test_path = pathes.train, pathes.valid, pathes.test
    self.train = self.dataset_type(
      train_path, *args, num_lines=num_train_data, **kwargs) 
    self.valid = self.dataset_type(valid_path, *args, **kwargs)
    self.test = self.dataset_type(test_path, *args, **kwargs)

class PriceDataset(PackedDatasetBase):
  pass

class NumNormalizedPriceDataset(PackedDatasetBase):
  pass

class CurrencyNormalizedPriceDataset(PackedDatasetBase):
  pass

class AllNormalizedPriceDataset(PackedDatasetBase):
  pass

class PriceDatasetWithFeatures(PackedDatasetBase):
  @common.timewatch()
  def __init__(self, dataset_type, pathes, num_train_data, vocab,
               *args, **kwargs):
    # Create POS Vocabulary from training data.
    pos_vocab_path = pathes.train + '.pos.vocab'
    if not os.path.exists(pos_vocab_path):
      data = pd.read_csv(pathes.train).fillna(EMPTY)
      text_data = data['sentence']
      sources = [[_BOS] + vocab.word.tokenizer(l, normalize_digits=False) 
                 for l in text_data]
      pos_tokens = common.get_pos(sources, output_path=pathes.train)
    else:
      pos_tokens = []
    vocab.pos = FeatureVocab(pos_vocab_path, pos_tokens, start_vocab=[_PAD, _BOS, _EOS, _UNK])
    vocab.wtype = FeatureVocab(None, [])

    PackedDatasetBase.__init__(self, dataset_type, pathes, num_train_data, vocab, *args, **kwargs)
