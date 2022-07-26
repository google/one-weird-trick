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
    'close_words', '',
    'Pkl file containing pairs of (distance, word) sorting in order of ascending distance.'
)

def main(_):
  all_benchmarks_vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(all_benchmarks_vocab_file, 'rb') as f:
    all_benchmarks_vocab = pickle.load(f)
  print('all_benchmarks_vocab', len(all_benchmarks_vocab))
  close_words = [FLAGS.close_words]
  if ',' in FLAGS.close_words:
    close_words = FLAGS.close_words.split(',')
  for vocab_file in close_words:
    with open(vocab_file, 'rb') as f:
      vocab = set(pickle.load(f))
    print(os.path.basename(vocab_file), len(vocab))
    union_vocab = all_benchmarks_vocab.union(vocab)
    print('union', len(union_vocab))
    dir = os.path.dirname(vocab_file)
    base = os.path.basename(vocab_file[:-len('.pkl')] + '_and_all_benchmarks_vocab.pkl')
    output_file = os.path.join(dir, base)
    print('output_file', output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(union_vocab, f)

if __name__ == '__main__':
  app.run(main)
