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
    'Pkl file containing pairs of (match_count, word) sorted in order of ascending frequency.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Directory to write out the vocab files to.'
)

def main(_):
  sorted_tuple_file = FLAGS.sorted_tuple_file
  output_dir = FLAGS.output_dir
  #'/usr/local/google/home/agoldie/data/glove/all_sorted_count_to_word_tuples.pkl'
  with open(sorted_tuple_file, 'rb') as f:
    sorted_tuples = pickle.load(f)
  print(len(sorted_tuples))
  stride = 5000
  end_index = 1085000
  vocab_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
  vocab_sizes = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1087156]
  for vocab_size in vocab_sizes:
    sample_vocab = [p[1] for p in sorted_tuples[:vocab_size]]
    output_vocab_file = os.path.join(output_dir, 'start0-vocab' + str(vocab_size) + '.pkl')
    print(output_vocab_file)
    with open(output_vocab_file, 'wb') as f:
      pickle.dump(sample_vocab, f)
    #print(len(sample_vocab), vocab_size)

if __name__ == '__main__':
  app.run(main)
