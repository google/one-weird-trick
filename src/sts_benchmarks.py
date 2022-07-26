"""Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Performs word similarity evaluations over a set of embeddings."""
from __future__ import print_function
import google3

from scipy.spatial import distance
from scipy.stats import pearsonr

from google3.pyglib import app
from google3.pyglib import flags
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2

import numpy as np
from numpy.linalg import norm
from numpy import dot
import csv
import pickle
import os
import collections


flags.DEFINE_string(
    'benchmark', '',
    'Benchmark on which to evaluate embeddings. To run on multiple benchmarks, enter as a comma-separated string.'
)

flags.DEFINE_string(
    'embedding_table', '',
    'Table in which pre-trained word embeddings are stored. To evaluate on multiple word embeddings, enter as a comma-separated string.'
)

flags.DEFINE_string(
    'decontextual_embedding_table', '',
    'Table in which decontextualized word embeddings are stored. To be used as a backoff if no contextual word embeddings is available.'
)


flags.DEFINE_string(
    'embedding_map', '',
    'Pickle file containing map of words to their embeddings.'
)

flags.DEFINE_string(
    'include_pattern', '',
    'Substring that must be present in embedding tables for them to be evaluated.'
)

flags.DEFINE_boolean(
    'print_baselines', False,
    'Whether or not to print baselines reported in All-But-The-Top.')

BENCHMARKS = {
    'semeval_2012': '/usr/local/google/home/agoldie/data/sts/semeval_2012/tokenized_benchmark.pkl',
    'semeval_2013': '/usr/local/google/home/agoldie/data/sts/semeval_2013/tokenized_benchmark.pkl',
    'semeval_2014': '/usr/local/google/home/agoldie/data/sts/semeval_2014/tokenized_benchmark.pkl',
    'semeval_2015': '/usr/local/google/home/agoldie/data/sts/semeval_2015/tokenized_benchmark.pkl',
    'semeval_2016': '/usr/local/google/home/agoldie/data/sts/semeval_2016/tokenized_benchmark.pkl',
    'semeval_all': '/usr/local/google/home/agoldie/data/sts/semeval_all/tokenized_benchmark.pkl',
}

FLAGS = flags.FLAGS

def get_embedding(word, embedding_table):
  # TODO(agoldie): Use SSTableService to make reads faster.
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  print('table', table.items())
  word = word.encode('utf-8')
  print('word', word)
  if word in table:
    return table[word]

def get_sentence_embedding_from_map(sentence, embedding_file):
  table = sstable.SSTable(
      embedding_file,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  dir = os.path.dirname(embedding_file)
  basename = os.path.basename(embedding_file)
  sum_embedding = np.zeros(300)
  total_embeddings = 0
  num_oovs = 0
  oovs = set()
  for word in sentence:
    word = word.lower().encode('utf-8')
    if word in table:
      total_embeddings += 1
      embedding = np.array(table[word].vector.values)
      sum_embedding += embedding
    else:
      oovs.add(word)
      num_oovs += 1
  #print('sentence', sentence)
  if total_embeddings == 0:
    return None, num_oovs, oovs
  else:
    return sum_embedding / total_embeddings, num_oovs, oovs

def get_similarity(e1, e2):
  return dot(e1, e2) / (norm(e1) * norm(e2))

def generate_oov_stats(oovs):
  num_skipped = len(oovs)
  unique_oovs = set()
  for tuple in oovs:
    for word in tuple:
      unique_oovs.add(word)
  return num_skipped, unique_oovs


def write_csv_file(filename, data):
  with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(
        csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(data)


def run_benchmark(benchmark, embedding_map, decontextual_embeddings=None):
  tokenized_benchmark_file = BENCHMARKS[benchmark]
  with open(tokenized_benchmark_file, 'rb') as f:
    tokenized_benchmark = pickle.load(f)
  predicted_sims, actual_sims, oovs = [], [], []
  total_items = 0
  all_word_oovs = set()
  for (actual_sim, s1, s2) in tokenized_benchmark:
    total_items += 1
    e1, num_oovs1, word_oovs1 = get_sentence_embedding_from_map(s1, embedding_map)
    e2, num_oovs2, word_oovs2 = get_sentence_embedding_from_map(s2, embedding_map)
    all_word_oovs.update(word_oovs1)
    all_word_oovs.update(word_oovs2)
    if e1 is None or e2 is None:
      oov = []
      if e1 is None:
        oov.append(s1)
      if e2 is None:
        oov.append(s2)
      oovs.append(oov)
      continue
    predicted_sim = get_similarity(e1, e2)
    actual_sims.append(actual_sim)
    predicted_sims.append(predicted_sim)
  rho, _ = pearsonr(actual_sims, predicted_sims)
  #print(benchmark_name, str.format('{:.2f}', rho * 100))

  num_skipped, unique_oovs = generate_oov_stats(word_oovs1.union(word_oovs2))
  if num_skipped:
    print('num_skipped', num_skipped, 'out of', total_items)
  #   print(unique_oovs)
  #print('word_oovs', all_word_oovs)
  return rho, unique_oovs


# wrote embeddings from /usr/local/google/home/agoldie/data/word2vec/embedding_map.pckl to /usr/local/google/home/agoldie/data/word2vec/embeddings.sstable
def main(_):
  benchmarks = FLAGS.benchmark
  vocab = set()
  if benchmarks == 'all':
    all_benchmarks = list(BENCHMARKS.keys())
    benchmarks = ','.join(all_benchmarks)
  contains_glove = 'glove' in FLAGS.embedding_table
  contains_word2vec = 'word2vec' in FLAGS.embedding_table
  for benchmark in benchmarks.split(','):
    print('\033[1m' + benchmark + ':' + '\033[0m')
    if FLAGS.print_baselines:
      if contains_word2vec:
        if benchmark == 'wordsim-combined':
          print('reported (word2vec): 68.29')
          print('reported (removed+subtracted): 69.05')
        elif benchmark == 'rw':
          print('reported (word2vec): 53.74')
          print('reported (removed+subtracted): 54.33')
        elif benchmark == 'men':
          print('reported (word2vec): 78.20')
          print('reported (removed+subtracted): 79.08')
        elif benchmark == 'simlex':
          print('reported (word2vec): 44.20')
          print('reported (removed+subtracted): 45.10')
        elif benchmark == 'simverb':
          print('reported (word2vec): 36.35')
          print('reported (removed+subtracted): 36.50')
      if contains_glove:
        if benchmark == 'wordsim-combined':
          print('reported (glove): 73.79')
          print('reported (removed+subtracted): 76.79')
        elif benchmark == 'rw':
          print('reported (glove): 46.41')
          print('reported (removed+subtracted): 52.04')
        elif benchmark == 'men':
          print('reported (glove): 80.49')
          print('reported (removed+subtracted): 81.78')
        elif benchmark == 'simlex':
          print('reported (glove): 40.83')
          print('reported (removed+subtracted): 44.97')
        elif benchmark == 'simverb':
          print('reported (glove): 28.33')
          print('reported (removed+subtracted): 32.23')
    embedding_table = FLAGS.embedding_table
    embedding_tables = [embedding_table]
    result_map = collections.defaultdict(list)
    is_dir = False
    if os.path.isdir(embedding_table):
      is_dir = True
      embedding_tables = [os.path.join(embedding_table, f) for f in os.listdir(embedding_table)]
      embedding_tables = [t for t in embedding_tables if t.endswith('.sstable')]
      if FLAGS.include_pattern:
        include_pattern = FLAGS.include_pattern
        embedding_tables = [t for t in embedding_tables if include_pattern in t]
    elif ',' in embedding_table:
      embedding_tables = embedding_table.split(',')
    else:
      embedding_tables = [embedding_table]
    for table in embedding_tables:
      embedding_name = os.path.basename(table).split('.')[0]
      rho, oov = run_benchmark(benchmark, table, FLAGS.decontextual_embedding_table)
      if is_dir:
        result_map[rho].append(embedding_name)
      else:
        print(embedding_name, str.format('{:.2f}', rho * 100))
    if is_dir:
      for key in sorted(result_map.keys()):
        print(str.format('{:.2f}', key * 100), result_map[key])
    print()


if __name__ == '__main__':
  app.run()
