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
import csv
import pickle


flags.DEFINE_string(
    'benchmark', '',
    'Benchmark on which to evaluate embeddings. To run on multiple benchmarks, enter as a comma-separated string.'
)

flags.DEFINE_string(
    'input_embeddings', '',
    'Embedding file with the first token (column) containing the vocabulary item.'
)

flags.DEFINE_string(
    'input_sstable', '',
    'Embedding sstable whose keys are the vocabulary items.'
)

flags.DEFINE_string(
    'output_vocab', '',
    'Pkl file to store extracted vocabulary.'
)

BENCHMARKS = {
    'simlex': {
        'description': 'SimLex-999',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/simlex/SimLex-999.txt',
        'delimiter': '\t',
        'score_col': 4,
        'has_header': True
    },
    # 'wordsim-combined': {
    #     'description': 'WordSim353 - Combined',
    #     'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/wordsim353/combined.tab',
    #     'delimiter': '\t',
    #     'score_col': 3,
    #     'has_header': True
    # },
    'wordsim-sim': {
        'description': 'WordSim353 - Similarity',
        'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt',
        'delimiter': '\t',
        'score_col': 3,
        'has_header': False
    },
    # 'wordsim-rel': {
    #     'description': 'WordSim353 - Relatedness',
    #     'file': '/google/data/ro/teams/commonsense/wsd/word-similarity/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt',
    #     'delimiter': '\t',
    #     'score_col': 3,
    #     'has_header': False
    # },
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
    'mikolov_analogy': {
        'description': 'Mikolov (2013) Word Analogy Task - 19544',
        'file': '/usr/local/google/home/agoldie/word2vec/questions-words.txt',
        'delimiter': ' ',
        'score_col': 3,
        'has_header': True},
}


FLAGS = flags.FLAGS

def read_in_benchmark(name):
  benchmark = BENCHMARKS[name]
  if 'score_col' not in benchmark:
    print('WARNING: score_col not set, defaulting to third column!')
    benchmark['score_col'] = 3
  return (benchmark['file'], benchmark['has_header'], benchmark['delimiter'],
          benchmark['score_col'])

def collect_vocab_from_analogy_benchmark(name):
  benchmark = BENCHMARKS[name]
  file = benchmark['file']
  vocab = set()
  with open(file, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith(':'):
        continue
      vocab.update(line.split())
  return vocab

def collect_vocab_from_benchmark(benchmark):
  benchmark, has_header, delimiter, score_col = read_in_benchmark(benchmark)
  vocab = set()
  with open(benchmark, newline='') as csvfile:
    sim_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
    if has_header:
      next(sim_reader)
    for row in sim_reader:
      w1, w2, actual_sim = (row[0], row[1], float(row[score_col-1]))
      vocab.add(w1)
      vocab.add(w2)
  return vocab

def collect_vocab_from_embedding_file(embedding_file):
  import pdb; pdb.set_trace()
  vocab = set()
  with open(embedding_file, 'r') as f:
    for line in f:
      token = line.split()[0]
      vocab.add(token)
  return vocab

def collect_vocab_from_embedding_sstable(embedding_file):
  import pdb; pdb.set_trace()
  vocab = set()
  table = sstable.SSTable(
    embedding_file,  # Proto2
    wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  with open(embedding_file, 'r') as f:
    for word in table:
      vocab.add(word)
  return vocab

def main(_):
  benchmarks = FLAGS.benchmark
  input_embeddings = FLAGS.input_embeddings
  input_sstable = FLAGS.input_sstable
  vocab = set()
  if benchmarks:
    if benchmarks == 'all':
      all_benchmarks = list(BENCHMARKS.keys())
      benchmarks = ','.join(all_benchmarks)
    for benchmark in benchmarks.split(','):
      if 'analogy' in benchmark:
        new_vocab = collect_vocab_from_analogy_benchmark(benchmark)
      else:
        new_vocab = collect_vocab_from_benchmark(benchmark)
      print(benchmark, len(new_vocab))
      vocab.update(new_vocab)
  elif input_embeddings:
    new_vocab = collect_vocab_from_embedding_file(input_embeddings)
    vocab.update(new_vocab)
  elif input_sstable:
    new_vocab = collect_vocab_from_embedding_sstable(input_sstable)
    vocab.update(new_vocab)

  print('overall', len(vocab))
  print(vocab)
  import pickle
  print(len(vocab))
  with open(FLAGS.output_vocab, 'wb') as f:
    pickle.dump(vocab, f)

if __name__ == '__main__':
  app.run()
