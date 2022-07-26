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

# Lint as: python3
"""Aggregate frequency of word embeddings.
"""

from absl import app
from absl import flags

import numpy as np
import csv
import pickle
import nltk
import os
from nltk.corpus import words
from random import sample
import collections

FLAGS = flags.FLAGS

flags.DEFINE_string('vocab_file', '', 'Pkl file containing vocabulary')

flags.DEFINE_string('frequency_dir', '', 'Pkl file containing vocabulary')

def main(argv):
  vocab_file = FLAGS.vocab_file
  with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

  frequency_dir = FLAGS.frequency_dir
  shards = os.listdir(frequency_dir)
  #shards = ['googlebooks-eng-all-1gram-20120701-b']
  word_to_count = collections.defaultdict(int)
  for shard in shards:
    ch = shard[-1]
    path = os.path.join(frequency_dir, shard)
    print(path)
    with open(path, 'r') as f:
      for i, line in enumerate(f):
        #line = line.strip()
        word, year, mc, vc = line.strip().split('\t')
        if word in vocab:
          word_to_count[word] += int(mc)
    count_to_word_tuples = []
    for word in word_to_count:
      count_to_word_tuples.append((word_to_count[word], word))
    count_to_word_tuples.sort()
    output_file = '/usr/local/google/home/agoldie/data/glove/sorted_count_to_word_tuples_' + ch + '.pkl'
    with open(output_file, 'wb') as f:
      pickle.dump(count_to_word_tuples, f)
    print(len(count_to_word_tuples))
    print('output_file', output_file)



if __name__ == '__main__':
  app.run(main)
