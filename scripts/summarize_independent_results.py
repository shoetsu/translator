# coding: utf-8
import argparse, re, sys, os
import pandas as pd
import commands


def main(args):
  #categories = ['overall', 'more', 'less', 'range', 'multi', 'rate', 'exact']
  categories = ['overall']
  for category in categories:
    summarize(args.root_path, args.test_filename + '.best.%s' % category)

def summarize(root_path, result_file):
  golds = []
  preds = []
  successes = []
  stats = []
  output_path = args.root_path + '/tests'
  #for model_path in commands.getoutput('ls -d %s/*' % args.root_path).split('\n'):
  for column_name in ['LB', 'UB', 'Unit', 'Rate']:
    model_path = os.path.join(args.root_path, column_name)
    if model_path == output_path:
      continue
    file_path = model_path + '/tests/' + result_file
    if not os.path.exists(file_path):
      sys.stderr.write('File not found: %s\n' % file_path)
      return
    indices, col_successes, sents, token_sents, col_golds, col_preds, col_stats = read(file_path)
    successes.append(col_successes)
    stats.append(col_stats)
    golds.append(col_golds)
    preds.append(col_preds)
  successes = zip(*successes)
  golds = zip(*golds)
  preds = zip(*preds)
  stats.insert(0, ['Metrix', 'EM accuracy', 'Precision', 'Recall'])
  stats = zip(*stats)
  header = stats[0]
  stats = zip(*stats[1:])
  df = pd.DataFrame({k:v for k,v in zip(header, stats)}).ix[:, header].set_index('Metrix')

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  with open(os.path.join(output_path, result_file), 'w') as f:
    sys.stdout = f
    for idx, success, sent, token_sent, gold, pred in zip(indices, successes, sents, token_sents, golds, preds):
      print "<%s> (%s)" % (idx, ', '.join(success))
      print sent
      print token_sent
      print 'Human label      :\t%s' % ' | '.join(gold)
      print 'Test prediction  :\t%s' % ' | '.join(pred)
    print ''
    print df
    sys.stdout = sys.__stdout__


def read(file_path):
  i = 0
  idx, success, sent, token_sent, gold, pred = None, None, None, None, None, None
  res = []
  now_in_examples = True
  stats = []
  for l in open(file_path):
    l = l.replace('\n', '')
    if now_in_examples:
      m = re.search('^<(\d+)> \((.+)\)', l)
      m2 = re.search('^Test input\s*:\s*(.+)', l)
      m3 = re.search('^Test input \(unk\)\s*:\s*(.+)', l)
      m4 = re.search('^Human label\s*:\s*(.+)', l)
      m5 = re.search('^Test prediction\s*:\s*(.+)', l)
      if m:
        idx = m.group(1)
        success = m.group(2)
      if m2: 
        sent = m2.group(0)
      if m3: 
        token_sent = m3.group(0)
      if m4:
        gold = [x.strip() for x in m4.group(1).split('|')]
        gold = gold[0]
      if m5:
        pred = [x.strip() for x in m5.group(1).split('|')]
        pred = pred[0]
      if idx and sent and token_sent and gold and pred:
        res.append([idx, success, sent,token_sent, gold, pred])
        idx, success, sent, token_sent, gold, pred = None, None, None, None, None, None
      if not l.strip():
        now_in_examples = False
    else:
      stats.append(l.strip())

  col_name = stats[0]
  pattern = '[0-9\.]+'
  try:
    em = re.search(pattern, stats[2]).group(0)
    prec = re.search(pattern, stats[3]).group(0)
    recall = re.search(pattern, stats[4]).group(0)
  except:
    print stats
    exit(1)
  indices, successes, sents, token_sents, golds, preds = zip(*res)
  col_stat = [col_name, em, prec, recall]
  return indices, successes, sents, token_sents, golds, preds, col_stat


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root_path")
  parser.add_argument('--test_filename', default='test.price.csv')
  
  args  = parser.parse_args()
  main(args)


