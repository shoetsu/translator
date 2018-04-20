# coding: utf-8
import argparse
import pandas as pd
def preprocess(sent):
  sent = sent.replace('\xe2\x80\x8b', '')
  sent = sent.replace('\n', '')
  return sent

def main(args):
  data = pd.read_csv(args.file_path)
  columns = [x for x in data.columns if x not in ['index', 'sentence']]
  # Noisy character appears in the sentences for some reason... 
  sentences = [preprocess(x) for x in data['sentence']]
  data['sentence'] = sentences
  print data.ix[:, ['index', 'sentence']+columns].set_index('index').to_csv()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  args  = parser.parse_args()
  main(args)

