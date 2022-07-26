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
from scipy.stats import spearmanr

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
    'simlex': {
        'description': 'SimLex-999',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/simlex/SimLex-999.txt',
        'delimiter': '\t',
        'score_col': 4,
        'has_header': True
    },
    'wordsim': {
        'description': 'WordSim353 - Similarity',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt',
        'delimiter': '\t',
        'score_col': 3,
        'has_header': False
    },
    'men': {
        'description': 'MEN',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/MEN/MEN_dataset_natural_form_full',
        'delimiter': ' ',
        'score_col': 3,
        'has_header': False},
    'rw': {
        'description': 'Rare Word (RW) Similarity Dataset by Stanford NLP',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/rw/rw.txt',
        'delimiter': '\t',
        'score_col': 3,
        'has_header': False},
    'simverb': {
        'description': 'SimVerb-3500',
        'file': '/usr/local/google/home/agoldie/data/SimVerb-3500.txt',
        'delimiter': '\t',
        'score_col': 4,
        'has_header': False},
    'rg65': {
        'description': 'RG-65',
        'file': '/usr/local/google/home/agoldie/data/RG-65.txt',
        'delimiter': ';',
        'score_col': 3,
        'has_header': False},
    'mturk': {
        'description': 'MTURK-771',
        'file': '/usr/local/google/home/agoldie/data/MTURK-771.csv',
        'delimiter': ',',
        'score_col': 3,
        'has_header': False},
}

FLAGS = flags.FLAGS


def rekey_embedding_table(old_sstable, new_sstable):
  old_sstable = sstable.SSTable(
      old_sstable,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  with sstable.SortingBuilder(new_sstable) as builder:
    i = 0
    for k, v in old_sstable.iteritems():
      i += 1
      if i == 1:
        continue
      print(i, v.token)
      builder.Add(v.token, v.SerializeToString())


def read_in_benchmark(name):
  benchmark = BENCHMARKS[name]
  if 'score_col' not in benchmark:
    print('WARNING: score_col not set, defaulting to third column!')
    benchmark['score_col'] = 3
  return (benchmark['file'], benchmark['has_header'], benchmark['delimiter'],
          benchmark['score_col'])


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

def get_embedding_from_map(word, embedding_table):
  # TODO(agoldie): Use SSTableService to make reads faster.
  #import pdb; pdb.set_trace()
  contextual_table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  dir = os.path.dirname(embedding_table)
  basename = os.path.basename(embedding_table)
  #layer = basename.split('embeddings-')[1].split('-')[0]
  layer = '0'
  decontextual_table_file = os.path.join(dir, 'decontextualized-embeddings-' + layer + '.sstable')
  if os.path.exists(decontextual_table_file):
    decontextual_table = sstable.SSTable(
        decontextual_table_file,  # Proto2
        wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  word = word.encode('utf-8')
  if word in contextual_table:
    return contextual_table[word]
  elif os.path.exists(decontextual_table_file) and word in decontextual_table:
    return decontextual_table[word]

def get_similarity(e1, e2):
  e1_array = np.array(e1.vector.values)
  e2_array = np.array(e2.vector.values)
  return dot(e1_array, e2_array) / (norm(e1_array) * norm(e2_array))

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


def run_benchmark(benchmark_name, embedding_map, decontextualized_embeddings=None):
  #print('Evaluating', embedding_map, 'on', benchmark)
  benchmark, has_header, delimiter, score_col = read_in_benchmark(benchmark_name)
  predicted_sims, actual_sims, oovs = [], [], []
  total_items = 0
  with open(benchmark, newline='') as csvfile:
    sim_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
    if has_header:
      next(sim_reader)
    for row in sim_reader:
      total_items += 1
      w1, w2, actual_sim = (row[0], row[1], float(row[score_col-1]))
      e1 = get_embedding_from_map(w1, embedding_map)
      e2 = get_embedding_from_map(w2, embedding_map)
      if not e1 or not e2:
        oov = []
        if not e1:
          oov.append(w1)
        if not e2:
          oov.append(w2)
        oovs.append(oov)
        continue
      predicted_sim = get_similarity(e1, e2)
      actual_sims.append(actual_sim)
      predicted_sims.append(predicted_sim)
  rho, _ = spearmanr(actual_sims, predicted_sims, nan_policy='omit')
  #print(benchmark_name, str.format('{:.2f}', rho * 100))
  num_skipped, unique_oovs = generate_oov_stats(oovs)
  if num_skipped:
    print('num_skipped', num_skipped, 'out of', total_items)
  #   print(unique_oovs)
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
      if 'union' in table:
        continue
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
