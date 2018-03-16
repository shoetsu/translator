#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM, FeatureVocab
from utils import evaluation, tf_utils, common

NONE = '-' # token for empty label.

def find_entry(target_labels):
  """
  Convert label strings to sequence of values. 
  If there are two or more label pairs, use the first one.

  Args:
    - target_labels : A string with the following format.  
      (e.g. "(30|1|60|1|$|item):(100|0|-|-|$|hour)")
  """
  try:
    labels = [x.strip() for x in re.search('\((.+)\)', target_labels.split(':')[0]).group(1).split('|')]
  except:
    print "There is an example with no label."
    exit(1)
  return labels

def remove_special_tokens(tokens):
  special_tokens = [_BOS, _PAD]
  return [x for x in tokens if x not in special_tokens]


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
      pred = NONE
    preds.append(pred)
  return preds

def ids2tokens(s_tokens, t_tokens, p_idxs):
  outs = []
  preds = []
  inp = [x for x in s_tokens if x not in [_BOS, _PAD]]
  if t_tokens is not None:
    for tt in t_tokens:
      out = ' '.join([x for x in tt if x not in [_BOS, _PAD]])
      outs.append(out)
  for pp in p_idxs:
    pred =  [s_tokens[k] for k in pp if len(s_tokens) > k]
    pred = ' '.join([x for x in pred if x not in [_BOS, _PAD]])
    if not pred:
      pred = NONE
    preds.append(pred)
  return inp, outs, preds

def create_demo_batch(sentences, dataset_type, vocab, 
                      target_columns, tmp_path='/tmp'):
  '''
  Args:
    sentences: List of string.
  '''
  tmp_path = os.path.join(tmp_path, common.random_string(5))
  index = [i for i in xrange(len(sentences))]
  dic = {
    'index': index,
    'sentence': sentences,
  }
  for col in target_columns:
    dic[col] = ['']
  df = pd.DataFrame(dic)
  with open(tmp_path, 'w') as f:
    f.write(df.to_csv() + '\n')
  pathes = common.dotDict({'train': tmp_path, 'valid':tmp_path, 'test':tmp_path})
  dataset = dataset_type(pathes, vocab, no_train=True)
  os.system('rm %s' % tmp_path)
  return dataset.test

class DatasetBase(object):
  pass

class _PriceDataset(DatasetBase):
  @common.timewatch()
  def __init__(self, path, vocab, target_columns, num_lines=0):
    sys.stderr.write('Loading dataset from %s ...\n' % (path))
    data = pd.read_csv(path)
    if num_lines:
      data = data[:num_lines]
    self.vocab = vocab
    self.tokenizer = vocab.tokenizer
  
    text_data = data['sentence']
    # For copying, keep unnormalized sentences too.
    self.original_sources = [[_BOS] + self.tokenizer(l, normalize_digits=False) for l in text_data]
    self.indexs = data['index'].values
    self.sources = [[_BOS] + self.tokenizer(l) for l in text_data]

    #targets = [[self.tokenizer(x, normalize_digits=False) for x in find_entry(l)] for l in label_data]
    self.target_columns = target_columns
    self.all_columns = [x for x in data.columns if x not in ['index', 'sentence']]
    for c in target_columns:
      if not c in self.all_columns:
        raise ValueError('The name of column must be in the label columns of data. (\'%s\' not found in %s)' % (c, str(self.all_columns)))
    self.targets = [[self.tokenizer(x, normalize_digits=False) for x in data[col].values] for col in self.all_columns]
    self.targets = list(zip(*self.targets)) # to batch-major.

  def get_batch(self, batch_size,
                input_max_len=None, output_max_len=None, shuffle=False):
    sources, targets = self.symbolized
    if input_max_len:
      paired = [(s,t) for s,t in zip(sources, targets) if not len(s) > input_max_len ]
      sources, targets = list(zip(*paired))

    sources =  tf.keras.preprocessing.sequence.pad_sequences(sources, maxlen=input_max_len, padding='post', truncating='post', value=PAD_ID)
    targets = list(zip(*targets)) # to column-major. (for padding)
    targets = [tf.keras.preprocessing.sequence.pad_sequences(targets_by_column, maxlen=output_max_len, padding='post', truncating='post', value=PAD_ID) for targets_by_column in targets]
    targets = list(zip(*targets)) # to idx-major. (for shuffling)

    data = [tuple(x) for x in zip(sources, targets, self.original_sources)]

    if shuffle: # For training.
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      b_sources, b_targets, b_ori_sources = zip(*batch)
      b_targets = list(zip(*b_targets)) # to column-major.
      yield common.dotDict({
        'sources': np.array(b_sources),
        # Include only the labels in 'target_columns' to batch.
        'targets': [np.array(t) for t, col in zip(b_targets, self.all_columns) if col in self.target_columns],
        'original_sources': b_ori_sources,
        #'original_targets': b_ori_targets,
      })
  
  @property 
  def raw_data(self):
    return self.indexs, self.original_sources, self.targets

  @property 
  def symbolized(self):
    # Find the indice of the tokens in a source sentence which was copied as a target label.
    targets = [[[o.index(x) for x in tt if x != NONE and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]
    sources = [self.vocab.tokens2ids(s) for s in self.sources]
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
                   verbose=True, #prediction_is_index=True, 
                   target_path_prefix=None, output_types=None):
    test_inputs_tokens, _ = self.symbolized
    golds = []
    preds = []
    try:
      assert len(original_sources) == len(targets) == len(predictions)
    except:
      print len(original_sources), len(targets), len(predictions)
      exit(1)

    inputs = [] # Original texts.
    token_inputs = [] # Normalized texts with _UNK.
    for i, (raw_s, raw_t, p) in enumerate(zip(original_sources, targets, predictions)):
      inp = remove_special_tokens(raw_s)
      gold = OrderedDict({col:' '.join(remove_special_tokens(t)) for col, t in zip(self.all_columns, raw_t)})
      token_inp = self.vocab.ids2tokens(test_inputs_tokens[i])
      #if prediction_is_index:
      pred = OrderedDict({col:t for col, t in zip(self.target_columns, find_token_from_sentence(p, raw_s))})
      inputs.append(inp)
      token_inputs.append(token_inp)
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
    all_success = lambda g, p: tuple(g.values()) == tuple(p.values())
    rate_success = lambda g, p: 'Rate' not in p or g['Rate'] == p['Rate']
    conditions = [
      ('overall', (lambda g: True, all_success)),
      ('range', (lambda g: g['LB'] != g['UB'] and g['LB'] != '-' and g['UB'] != '-', lower_upper_success)),
      ('multi', (lambda g: len(g['LB'].split(' ')) > 1 or len(g['UB'].split(' ')) > 1, lower_upper_success)),
      ('rate', (lambda g: g['Rate'] != '-', rate_success)),
      #('less', (lambda g: g[0] == '-' and g[1] != '-', lower_upper_success)),
      #('more', (lambda g: g[0] != '-' and g[1] == '-', lower_upper_success)),
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
      types = [' '.join(x) if x else '-' for x in types]
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
        print df_row_all.to_csv()
        print df_em_all.to_csv()
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
    summary = tf_utils.make_summary(summary_dict)

      #  pass
    return df_sums[0], summary

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
      is_success = cf[1](g, p)
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
      print df_row.to_csv()
      print ''
      print df_sum.to_csv()
    else:
      for i, is_success, inp, token_inp, g, p in res:
        succ_or_fail = 'EM_Success' if is_success else "EM_Failure"
        sys.stdout.write('<%d> (%s)\n' % (i, succ_or_fail))
        sys.stdout.write('Test input       :\t%s\n' % (inp))
        sys.stdout.write('Test input (unk) :\t%s\n' % (token_inp))
        sys.stdout.write('Human label      :\t%s\n' % (g))
        sys.stdout.write('Test prediction  :\t%s\n' % (p))

      #print df_row
      print ''
      print df_sum

    return df_sum, df_row

class _NumNormalizedPriceDataset(_PriceDataset):
  def __init__(self, path, vocab, target_columns, num_lines=0):
    _PriceDataset.__init__(self, path, vocab, num_lines, target_columns)
    self.vocab.add2vocab(_NUM)
    self.pos = common.get_pos(self.original_sources, output_path=path)
    assert len(self.pos) == len(self.sources)
    self.original_sources, self.targets, self.num_indices = self.concat_numbers(
      self.original_sources, self.targets, self.pos 
    )
    self.sources = self.original_sources

  @property 
  def symbolized(self):
    def manual_replace(s, idx):
      # Replace the token on a certain index to 0. These indices are those of the tokens with 'CD' POS.
      return [x if i not in idx else '0' for i, x in enumerate(s)]
    targets = [[[o.index(x) for x in tt if x != NONE and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]
    sources = [self.vocab.tokens2ids(manual_replace(s, idx)) for s, idx in zip(self.sources, self.num_indices)]
    return sources, targets

  def concat_numbers(self, _sources, _targets, _pos):
    # Concatenate consecutive CDs in sources and targets by '|'.
    # 'num_indices' shows the concatenated tokens and they will be normalized into '0' when being input as an embedding.
    delim = '|'
    sources = []
    targets = []
    num_indices = []
    for s, t, p in zip(_sources, _targets, _pos):
      assert len(s) == len(p)
      new_s = []
      tmp = []
      idx = []
      for i in xrange(len(s)):
        if p[i] == 'CD' and s[i] not in ['(', ')', '-']:
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

class _CurrencyNormalizedPriceDataset(_PriceDataset):
  @property
  def symbolized(self):
    targets = [[[o.index(x) for x in tt if x != NONE and x in o ] for tt in t] for o, t in zip(self.original_sources, self.targets)]

    def manual_replace(s):
      normalized_token = '$'
      currency_names = ['yen', 'dollar', 'euro', 'flanc', 'pound']
      currency_names += [x+'s' for x in currency_names]
      currency_symbols = ['₡', '£', '¥','₦', '₩', '₫', '₪', '₭', '€', '₮', '₱', '₲', '₴', '₹', '₸', '₺', '₽', '฿',]
      s = [x if x not in currency_names else 'dollars' for x in s]
      s = [x if x not in currency_symbols else '$' for x in s]
    sources = [self.vocab.tokens2ids(manual_replace(s)) for s  in self.sources]
    return sources, targets

class _PriceDatasetWithFeatures(_PriceDataset):
  def __init__(self, path, vocab, target_columns, num_lines=0):
    _PriceDataset.__init__(self, path, vocab, target_columns, num_lines=num_lines)
    self.pos = common.get_pos(self.original_sources, output_path=path)
    self.pos_vocab = None # given after initialization



###################################################
#    Classes for dataset pair (train, valid, test)
###################################################

class PackedDatasetBase(object):
  '''
  The class contains train, valid, test dataset.

  Args:
     pathes: A list of string. ([train_path, valid_path, test_path])
     vocab: A vocabulary object in core/vocabularies.py
  '''
  dataset_type = None
  def __init__(self, pathes, vocab, target_columns, 
               num_train_data=0, no_train=False):
    train_path, valid_path, test_path = pathes.train, pathes.valid, pathes.test
    self.train = self.dataset_type(train_path, vocab, target_columns, num_lines=num_train_data) if not no_train else None
    self.valid = self.dataset_type(valid_path, vocab, target_columns, )
    self.test = self.dataset_type(test_path, vocab, target_columns, )

class PriceDataset(PackedDatasetBase):
  dataset_type = _PriceDataset    

class NumNormalizedPriceDataset(PackedDatasetBase):
  dataset_type = _NumNormalizedPriceDataset

class CurrencyNormalizedPriceDataset(PackedDatasetBase):
  dataset_type = _CurrencyNormalizedPriceDataset

class PriceDatasetWithFeatures(PackedDatasetBase):
  dataset_type = _PriceDatasetWithFeatures
  def __init__(self, pathes, vocab, target_columns, 
               num_train_data=0, no_train=False):
    train_path, valid_path, test_path = pathes.train, pathes.valid, pathes.test
    self.train = self.dataset_type(
      train_path, vocab, 
      pos_vocab=pos_vocab, num_lines=num_train_data) if not no_train else None
    self.valid = self.dataset_type(valid_path, vocab, target_columns, pos_vocab=pos_vocab)
    self.test = self.dataset_type(test_path, vocab, target_columns, pos_vocab=pos_vocab)

    pos_lists = Counter(common.flatten(self.train.pos)).keys() if self.train else None
    pos_vocab = FeatureVocab




