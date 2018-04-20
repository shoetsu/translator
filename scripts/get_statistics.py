# coding:utf-8
import argparse, re, sys, os, collections
import pandas as pd

empty = '-'

class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

def main(args):
  df = pd.read_csv(args.file_path).fillna(empty)
  cnt = collections.defaultdict(int)

  conditions = [
    ('none', lambda x: x.LB == empty and x.UB == empty),
    ('exact', lambda x: x.LB != empty and x.UB != empty and x.LB == x.UB),
    ('rate', lambda x: x.Rate != empty),
    ('multi', lambda x: len(x.LB.split()) > 1 or len(x.UB.split()) > 1),
    ('range', lambda x: x.LB != empty and x.UB != empty and x.LB != x.UB),
    ('more', lambda x: x.LB != empty and x.UB == empty),
    ('less', lambda x: x.LB == empty and x.UB != empty),
  ]
  for l in df.values.tolist():
    entry = dotDict({col:val for col, val in zip(df.columns.tolist(), l)})
    for name, cond in conditions:
      if cond(entry):
        cnt[name] += 1
  assert cnt['none'] + cnt['exact'] + cnt['more'] + cnt['less'] + cnt['range'] == len(df)
  sep = ', '
  for k, v in cnt.items():
    print (sep.join([k, str(v)]))
  print (sep.join(['All', str(len(df))]))
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  args  = parser.parse_args()
  main(args)
