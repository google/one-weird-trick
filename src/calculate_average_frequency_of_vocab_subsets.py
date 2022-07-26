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

from absl import app
from absl import flags

import numpy as np
import csv
import pickle
import nltk
from nltk.corpus import words
from random import sample
import os

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'sorted_tuple_file', '',
    'Pkl file containing pairs of (match_count, word) sorted in order of descending frequency.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Directory to output subset pickles to.'
)


def main(_):
  sorted_tuple_file = '/usr/local/google/home/agoldie/data/glove/descending_sorted_count_to_word_tuples.pkl'
  with open(sorted_tuple_file, 'rb') as f:
    sorted_tuples = pickle.load(f)
  word_to_count_file = '/usr/local/google/home/agoldie/data/word_to_count.pkl'
  if os.path.exists(word_to_count_file):
    with open(word_to_count_file, 'rb') as f:
      word_to_count = pickle.load(f)
  else:
    word_to_count = {}
    for (count, word) in sorted_tuples:
      word_to_count[word] = count
    with open(word_to_count_file, 'wb') as f:
      pickle.dump(word_to_count, f)

  output_dir = FLAGS.output_dir
  vocab_files = ['/usr/local/google/home/agoldie/data/men_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/mturk_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/rg65_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/rw_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/simlex_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/simverb_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/wordsim_vocab.pkl',
                 '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl']

  import pdb; pdb.set_trace()
  for vocab_file in vocab_files:
    print(vocab_file)
    with open(vocab_file, 'rb') as f:
      vocab = pickle.load(f)
      print(len(vocab))
      total_count = 0
      for word in vocab:
        count = 0 if word not in word_to_count else word_to_count[word]
        total_count += count
      average_count = total_count / len(vocab)
      print(average_count)

  print(len(sorted_tuples))
  vocab_size = 50000
  max_start_index = 250000
  stride = 1000
  start_indices = [0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 250000]
  for start_index in start_indices:
    sample_vocab = [p[1] for p in sorted_tuples[start_index:start_index + vocab_size]]
    output_vocab_file = os.path.join(output_dir, 'start' + str(start_index) + '-vocab' + str(vocab_size) + '.pkl')
    print(output_vocab_file)
    with open(output_vocab_file, 'wb') as f:
      pickle.dump(sample_vocab, f)

if __name__ == '__main__':
  app.run(main)
