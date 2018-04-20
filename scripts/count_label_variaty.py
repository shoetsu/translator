# coding:utf-8
import argparse, re, sys, os, collections
import pandas as pd
def main(args):
  df = pd.read_csv(args.file_path)
  examples_not_empty = df[df[args.column_type]!= '-']
  counter = collections.Counter([x.lower() for x in examples_not_empty[args.column_type]]).most_common()
  
  for k,v in counter:
    print k, v
  print ''
  print '# Examples:\t%d' %  len(df[args.column_type])
  print '# Examples with %s:\t%d' %  (args.column_type, len(examples_not_empty))
  print '# Variations of %s:\t%d' %(args.column_type, len(counter))
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  parser.add_argument("--column_type", default='Rate')
  args  = parser.parse_args()
  main(args)


