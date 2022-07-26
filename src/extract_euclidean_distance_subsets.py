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

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'distance_to_words', '',
    'Pkl file containing pairs of (distance, word) sorting in order of ascending distance.'
)


def main(_):
  #import pdb; pdb.set_trace()
  with open(FLAGS.distance_to_words, 'rb') as f:
    distance_to_words = pickle.load(f)
  vocab_size = 10
  while vocab_size <= 10000000:
    sample_vocab = [p[1] for p in distance_to_words[:vocab_size]]
    output_vocab_file = '/usr/local/google/home/agoldie/data/euclidean_distance-' + str(vocab_size) + '.pkl'
    with open(output_vocab_file, 'wb') as f:
      pickle.dump(sample_vocab, f)
    print(len(sample_vocab), vocab_size)
    vocab_size *= 10

if __name__ == '__main__':
  app.run(main)
