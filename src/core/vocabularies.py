# coding:utf-8
import tensorflow as tf
import collections, os, time, re, sys, math
from tensorflow.python.platform import gfile
from orderedset import OrderedSet
from nltk.tokenize import word_tokenize
import numpy as np

import utils.common as common

_PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
_UNK = "_UNK"
_NUM = "_NUM"
_UNIT = "_UNIT"

ERROR_ID = -1
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

_DIGIT_RE = re.compile(r"\d")
START_VOCAB = [_PAD, _BOS, _EOS, _UNK, _NUM, _UNIT]
UNDISPLAYED_TOKENS = [_PAD, _BOS, _EOS]

def separate_numbers(sent):
  '''
  Since for some reason nltk.tokenizer fails to separate numbers (e.g. 6.73you),
  manually separate them.
  Args:
     sent: a string.
  '''
  for m in re.findall("(\D*)(\d?[0-9\,\.]*\d)(\D*?)", sent):
  #for m in re.findall("(\D*)(\d?[0-9\,\.]*\d)(\D*)", sent):
    m = [x for x in m if x]
    sent = sent.replace(''.join(m), ' ' + ' '.join(m)+ ' ')
    sent = ' '.join(sent.split())
  return sent

def separate_symbols(sent):
  symbols = ['/', ':']
  for sym in symbols:
    sent = sent.replace(sym, " %s " % sym)
  return ' '.join(sent.split())

class WordTokenizer(object):
  def __init__(self, lowercase=False, normalize_digits=False):
    self.lowercase = lowercase
    self.normalize_digits = normalize_digits

  def __call__(self, sent, normalize_digits=None, lowercase=None, flatten=None):
    try:
      sent = sent.replace('\n', '')
      sent = separate_numbers(sent)
      sent = separate_symbols(sent)
    except Exception as e:
      print e
      print sent
      exit(1)
    normalize_digits = normalize_digits if normalize_digits is not None else self.normalize_digits
    lowercase = lowercase if lowercase is not None else self.lowercase
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
      # special tokens are not transformed to lowercase.
      for t in START_VOCAB:
        sent.replace(t.lower(), t) 
    return word_tokenize(sent)

class VocabularyBase(object):
  def __init__(self, add_bos=False, add_eos=False):
    self.vocab = None
    self.rev_vocab = None
    self.name = None
    # self.start_offset = [BOS_ID] if add_bos else []
    # self.end_offset = [EOS_ID] if add_eos else []
    # self.n_start_offset = len(self.start_offset)
    # self.n_end_offset = len(self.end_offset)

  @property
  def size(self):
    return len(self.vocab)

  # def create_vocab(self, token_list, vocab_path):
  #   # Create vocab from a list of words.
  #   start_vocab = START_VOCAB
  #   rev_vocab = OrderedSet(start_vocab + token_list) 
  #   with open(vocab_path, 'w') as f:
  #     f.write('\n'.join(list(rev_vocab) + '\n'))
  #   return rev_vocab

  # def init_vocab(self, token_list, vocab_path, skip_first=False):
  #   rev_vocab = self.load_vocab(vocab_path, skip_first=skip_first)
  #   if rev_vocab is None:
  #     rev_vocab = self.create_vocab()

  #   vocab = collections.OrderedDict()
  #   for i,t in enumerate(rev_vocab):
  #     vocab[t] = i
  #   return vocab, rev_vocab


  # def load_vocab(vocab_path, skip_first=False):
  #   pass
  #   #with open(vocab_path)


class WordVocabularyBase(VocabularyBase):
  def id2token(self, _id):
    if not type(_id) in [int, np.int32]:
      raise ValueError('ID must be an integer but %s' % str(type(_id)))
    elif _id < 0 or _id > len(self.rev_vocab):
      raise ValueError('Token ID must be an integer between 0 and %d (ID=%d)' % (len(self.rev_vocab), _id))
    # elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
    #   return None
    else:
      return self.rev_vocab[_id]

  def idx2tokens(self, idxs, refer_tokens):
    return [refer_tokens[i] for i in idxs if refer_tokens[i] and refer_tokens[i] not in UNDISPLAYED_TOKENS]

  def ids2tokens(self, ids, link_span=None, join=False, remove_special=True):
    '''
    ids: a list of word-ids.
    link_span : a tuple of the indices between the start and the end of a link.
    '''
    def _ids2tokens(ids, link_span):
      sent_tokens = [self.id2token(word_id) for word_id in ids]
      if link_span:
        for i in xrange(link_span[0], link_span[1]+1):
          sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      if remove_special:
        sent_tokens = [w for w in sent_tokens 
                       if w and w not in UNDISPLAYED_TOKENS]
      if join:
        sent_tokens = " ".join(sent_tokens)
      return sent_tokens
    return _ids2tokens(ids, link_span)

  def token2id(self, token):
    # token: a string.
    # res: an interger.

    return self.vocab.get(token, UNK_ID)

  def str2ids(self, sentence):
    # sentence : a string.
    # res :a list of integer.

    tokens = self.tokenizer(sentence) 
    return [self.token2id(word) for word in tokens]

  def tokens2ids(self, tokens):
    if type(tokens) == list:
      res = [self.token2id(word) for word in tokens]
    elif type(tokens) == tf.Tensor and self.lookup_table:
      res = self.lookup_table.lookup(tokens)
    else:
      raise ValueError
    return res


class PredefinedVocabWithEmbeddingBase(object):
  def init_vocab(self, emb_configs, vocab_size=0):
    
    pretrained = [self.load_vocab(c['path'], c['format'] == 'vec', vocab_size=vocab_size) for c in emb_configs]
    rev_vocab = common.flatten([e.keys() for e in pretrained])
    # Todo: ここでtokenizer掛けない
    start_vocab = START_VOCAB
    rev_vocab = OrderedSet(start_vocab + [self.tokenizer(w, flatten=True)[0] 
                                          for w in rev_vocab])
    #rev_vocab = OrderedSet(start_vocab + [self.tokenizer(w, flatten=True)[0] 
    #                                      for w in rev_vocab])
    if vocab_size:
      rev_vocab = OrderedSet([w for i, w in enumerate(rev_vocab) if i < vocab_size])
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    #embeddings = [common.flatten([emb[w] for emb in pretrained]) for w in vocab]
    embeddings = [np.array([emb[w] for w in vocab]) for emb in pretrained]
    embeddings = np.concatenate(embeddings, axis=-1)
    return vocab, rev_vocab, embeddings

  def load_vocab(self, embedding_path, skip_first=True, 
                 vocab_size=0):
    '''
    Load pretrained vocabularies and embeddings.
    '''
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    embedding_dict = None
    if vocab_size and skip_first:
      vocab_size += 1

    word_and_emb = []
    with open(embedding_path) as f:
      for i, line in enumerate(f.readlines()):
        if skip_first and i == 0:
          continue
        if vocab_size and i > vocab_size:
          break
        #################
        word_and_embedding = line.split()
        word = self.tokenizer(word_and_embedding[0])
        if len(word) > 1:
          continue
        else:
          word = word[0]
        embedding = [float(s) for s in word_and_embedding[1:]]
        word_and_emb.append((word, embedding))
    embedding_size = len(word_and_emb[0][1])
    zero_vector = [0.0 for _ in xrange(embedding_size)]
    all_average_vector = np.mean([v for _, v in word_and_emb], axis=0)
    embedding_dict = common.OrderedDefaultDict(
      default_factory=lambda:np.random.uniform(-math.sqrt(3), math.sqrt(3),
                                               size=embedding_size))
                                        
    for k in START_VOCAB:
      embedding_dict[k] = default_embedding
    for k, v in word_and_emb:
      embedding_dict[k] = v
    ############ TEMPORARY ####################
    num_init_embedding_type = self.num_init_embedding_type
    unit_init_embedding_type = self.unit_init_embedding_type
    if num_init_embedding_type == 'word_0':
      embedding_dict[_NUM] = embedding_dict['0']
    elif num_init_embedding_type == 'average_all':
      pass
    elif unit_init_embedding_type == 'random':
      embedding_dict[_NUM] = np.random.uniform(-math.sqrt(3), math.sqrt(3),
                                               size=embedding_size)
    elif num_init_embedding_type == 'average_selective':
      average_vec = []
      for word in embedding_dict:
        if re.match('^[0-9]+$', word):
          average_vec.append(embedding_dict[word])
      average_vec = np.mean(average_vec, axis=0)
      embedding_dict[_NUM] = average_vec
    elif num_init_embedding_type == 'zero_vector':
      embedding_dict[_NUM] = zero_vector
    else:
      raise ValueError

    if unit_init_embedding_type == 'word_unit':
      embedding_dict[_UNIT] = embedding_dict['unit']
    elif unit_init_embedding_type == 'word_0':
      embedding_dict[_UNIT] = embedding_dict['0']
    elif unit_init_embedding_type == 'average_all':
      pass
    elif unit_init_embedding_type == 'random':
      embedding_dict[_UNIT] = np.random.uniform(-math.sqrt(3), math.sqrt(3), 
                                                size=embedding_size)
    elif unit_init_embedding_type == 'average_selective':
      unit_names = ['yen', 'dollar', 'euro', 'franc', 'pound', 'cent', 'buck']
      unit_names += [x+'s' for x in unit_names]
      unit_symbols = ['$', '₡', '£', '¥','₦', '₩', '₫', '₪', '₭', '€', '₮', '₱', '₲', '₴', '₹', '₸', '₺', '₽', '฿',]
      unit_tokens = unit_names + unit_symbols
      average_vec = np.mean([embedding_dict[t] for t in unit_tokens 
                             if t in embedding_dict], axis=0)
      embedding_dict[_UNIT] = average_vec
    elif unit_init_embedding_type == 'zero_vector':
      embedding_dict[_UNIT] = zero_vector
    else:
      raise ValueError
    sys.stderr.write("Done loading word embeddings.\n")
    # print embedding_dict["0"]
    # print embedding_dict["unit"]
    # print embedding_dict[_UNIT]
    # print zero_vector
    # print self.num_init_embedding_type
    # print self.unit_init_embedding_type
    # exit(1)
    return embedding_dict


class WordVocabularyWithEmbedding(WordVocabularyBase, PredefinedVocabWithEmbeddingBase):
  def __init__(self, emb_configs,
               vocab_size=0, lowercase=False, normalize_digits=True, 
               normalize_embedding=False, 
               num_init_embedding_type=None, unit_init_embedding_type=None):
    self.tokenizer = WordTokenizer(lowercase=lowercase,
                                   normalize_digits=normalize_digits)
    self.normalize_embedding = normalize_embedding
    self.num_init_embedding_type = num_init_embedding_type
    self.unit_init_embedding_type = unit_init_embedding_type
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      emb_configs, vocab_size)

  @property
  def lookup_table(self):
    return tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(self.vocab.keys(), 
                                                  self.vocab.values()), UNK_ID)
  # def add2vocab(self, token, new_embedding=None):
  #   if token not in self.rev_vocab:
  #     self.vocab[token] = len(self.vocab)
  #     self.rev_vocab.add(token)
  #     if not new_embedding:
  #       new_embedding = np.zeros((1, len(self.embeddings[0])))
  #     self.embeddings = np.concatenate(
  #       (self.embeddings, new_embedding)
  #     )


class FeatureVocab(WordVocabularyBase):
  def __init__(self, vocab_path, source):
    if source and type(source[0]) not in [str, unicode]:
      source = common.flatten(source)
    self.tokenizer = lambda x: [x]
    self.vocab, self.rev_vocab = self.init_vocab(vocab_path, source)

  def init_vocab(self, vocab_path, source, vocab_size=0):
    if os.path.exists(vocab_path):
      sys.stderr.write('Loading word vocabulary from %s...\n' % vocab_path)
      vocab, rev_vocab = self.load_vocab(vocab_path)
    else:
      sys.stderr.write('Restoring word vocabulary to %s...\n' % vocab_path)
      vocab, rev_vocab = self.create_vocab(source, vocab_size=vocab_size)
      self.save_vocab(vocab_path, rev_vocab)
    return vocab, rev_vocab

  def create_vocab(self, source, vocab_size=0):
    '''
    Args:
     - source: List of words.
    '''
    start_vocab = START_VOCAB 
    rev_vocab, freq = zip(*collections.Counter(source).most_common())
    rev_vocab = common.flatten([self.tokenizer(w) for w in rev_vocab])
    if type(rev_vocab[0]) == list:
      rev_vocab = common.flatten(rev_vocab)
    rev_vocab = OrderedSet(start_vocab + rev_vocab)
    if vocab_size:
      rev_vocab = OrderedSet([w for i, w in enumerate(rev_vocab) if i < vocab_size])
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    return vocab, rev_vocab
    
  def save_vocab(self, vocab_path, rev_vocab):
    '''
    Args:
     - vocab_path: The path to which the vocabulary will be restored.
     - rev_vocab: List of words.
    '''
    # Restore vocabulary.
    with open(vocab_path, 'w') as f:
      for k in rev_vocab:
        if type(k) == unicode:
          k = k.encode('utf-8')
        f.write('%s\n' % (k))

  def load_vocab(self, vocab_path):
    rev_vocab = [l.replace('\n', '').split('\t')[0] for l in open(vocab_path)]
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    return vocab, rev_vocab
