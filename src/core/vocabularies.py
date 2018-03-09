# coding:utf-8
import tensorflow as tf
import collections, os, time, re, sys
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

ERROR_ID = -1
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

_DIGIT_RE = re.compile(r"\d")
START_VOCAB = [_PAD, _BOS, _EOS, _UNK]
UNDISPLAYED_TOKENS = [_PAD, _BOS, _EOS]

def separate_numbers(sent):
  # for some reason nltk.tokenizer fails to separate numbers (e.g. 6.73you)
  for m in re.findall("(\D*)(\d?[0-9\,\.]*\d)(\D*)", sent):
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
    sent = sent.replace('\n', '')
    sent = separate_numbers(sent)
    sent = separate_symbols(sent)
    normalize_digits = normalize_digits if normalize_digits is not None else self.normalize_digits
    lowercase = lowercase if lowercase is not None else self.lowercase
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
    if lowercase:
      sent = sent.lower()
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
  def init_vocab(self, emb_configs, source_dir, vocab_size=0):
    start_vocab = START_VOCAB
    # if self.tokenizer.lowercase:
    #   start_vocab = [x.lower for x in lowercase]
    pretrained = [self.load_vocab(os.path.join(source_dir, c['path']), c['format'] == 'vec') for c in emb_configs]
    rev_vocab = common.flatten([e.keys() for e in pretrained])
    rev_vocab = OrderedSet(start_vocab + [self.tokenizer(w, flatten=True)[0] 
                                          for w in rev_vocab])
    if vocab_size:
      rev_vocab = OrderedSet([w for i, w in enumerate(rev_vocab) if i < vocab_size])
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    embeddings = [common.flatten([emb[w] for emb in pretrained]) for w in vocab]
    embeddings = np.array(embeddings)
    return vocab, rev_vocab, embeddings

  def load_vocab(self, embedding_path, skip_first=True):
    '''
    Load pretrained vocabularies and embeddings.
    '''
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    embedding_dict = None
    with open(embedding_path) as f:
      for i, line in enumerate(f.readlines()):
        if skip_first and i == 0:
          continue
        #################3
        #if False and i ==100 :
        if False and i==200:
          break
        #################
        word_and_embedding = line.split()
        word = self.tokenizer(word_and_embedding[0])
        if len(word) > 1:
          continue
        else:
          word = word[0]
        vector = word_and_embedding[1:]

        if not embedding_dict:
          embedding_size = len(vector)
          default_embedding = [0.0 for _ in xrange(embedding_size)]
          embedding_dict = collections.defaultdict(lambda:default_embedding)
          assert len(word_and_embedding) == embedding_size + 1

        embedding = [float(s) for s in vector]
        if word not in embedding_dict:
          embedding_dict[word] = embedding
      sys.stderr.write("Done loading word embeddings.\n")
    return embedding_dict


class WordVocabularyWithEmbedding(WordVocabularyBase, PredefinedVocabWithEmbeddingBase):
  def __init__(self, emb_configs, source_dir="dataset/embeddings",
               vocab_size=0,
               lowercase=False, normalize_digits=True, 
               normalize_embedding=False, add_bos=False, add_eos=False):
    
    #super(WordVocabularyBase, self).__init__(add_bos=add_bos, add_eos=add_eos)
    #super().__init__(add_bos=add_bos, add_eos=add_eos)
    self.tokenizer = WordTokenizer(lowercase=lowercase,
                                   normalize_digits=normalize_digits)
    self.normalize_embedding = normalize_embedding
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      emb_configs, source_dir, vocab_size)
    # For some reason "tf.contrib.lookup.HashTable" is not successfully imported when being run from jupyter.

  @property
  def lookup_table(self):
    return tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(self.vocab.keys(), 
                                                  self.vocab.values()), UNK_ID)
  def add2vocab(self, token, new_embedding=None):
    if token not in self.rev_vocab:
      self.vocab[token] = len(self.vocab)
      self.rev_vocab.add(token)
      if not new_embedding:
        new_embedding = np.zeros((1, len(self.embeddings[0])))
      self.embeddings = np.concatenate(
        (self.embeddings, new_embedding)
      )
