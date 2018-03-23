# coding:utf-8

import argparse
import pandas as pd
import sys, re, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src/')

from core.datasets import _PriceDataset as Dataset
from utils import common

def main(args):
  test_file = pd.read_csv(args.test_file)
  print pd.read_csv(args.prediction_file)
  exit(1)
  prediction_file = pd.read_csv(args.prediction_file)
  _, prediction_filename = common.separate_path_and_filename(args.prediction_file)
  all_columns = ['LB', 'UB', 'Unit', 'Rate']
  dataset = Dataset(args.test_file, None, args.target_attribute, all_columns)
  dataset.all_columns = all_columns

  tmp = [test_file[col] for col in test_file]
  sources, targets = tmp[1], tmp[2:]
  targets = list(zip(*targets))
  tmp = [prediction_file[col] for col in prediction_file]
  predictions = tmp[2:]
  predictions = list(zip(*predictions))
  target_path = 'checkpoints/naive_baseline/naive_results/'
  dataset.show_results(sources, targets, predictions, 
                       target_path_prefix=target_path + '/' + prediction_filename)
  exit(1)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_file", default='dataset/test.price.csv')
  parser.add_argument("--prediction_file", default='dataset/naive_result.csv')
  parser.add_argument("--target_attribute", default='Price')
  args  = parser.parse_args()
  main(args)


