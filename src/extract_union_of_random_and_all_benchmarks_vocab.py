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

def main(_):
  vocab_file = '/usr/local/google/home/agoldie/data/glove/full_glove_vocab.pkl'
  with open(vocab_file, 'rb') as f:
    vocab = set(pickle.load(f))
  all_benchmarks_vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(all_benchmarks_vocab_file, 'rb') as f:
    all_benchmarks_vocab = pickle.load(f)
  union_vocab = all_benchmarks_vocab.union(vocab)
  full_union_file = '/usr/local/google/home/agoldie/data/glove_and_all_benchmarks_vocab.pkl'
  with open(full_union_file, 'wb') as f:
      pickle.dump(union_vocab, f)

if __name__ == '__main__':
  app.run(main)
