# coding: utf-8
import argparse, re, sys, os
import pandas as pd
import random 
import numpy as np

random.seed(0)
np.random.seed(0)

def main(args):
  i = 0
  idx, sent, label = None, None, None
  res = []
  for l in open(args.file_path):
    m = re.match('^<(\d+)>', l)
    m2 = re.match('^Test input       :\t(.+)', l)
    m3 = re.match('^Human label      :\t(.+)', l)
    if m:
      idx = m.group(1)
    if m2: 
      sent = m2.group(1)
    if m3:
      label = [x.strip() for x in m3.group(1).split('|')]
    if idx and sent and label:
      res.append([idx, sent] + label)
      idx, sent, label = None, None, None
  indices, sents, lb, ub, currency, rate = list(zip(*res))
  df = pd.DataFrame({
    'index': indices,
    'sentence': sents,
    'LB':lb,
    'UB':ub,
    'Unit':currency,
    'Rate':rate
  }).ix[:, ['index', 'sentence', 'LB', 'UB', 'Unit', 'Rate']].set_index('index')
  if args.n_sample > 0:
    df = df.sample(args.n_sample, replace=True)
  print df.to_csv(header=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  parser.add_argument("--n_sample", default=0, type=int)
  args  = parser.parse_args()
  main(args)


