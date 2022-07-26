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
"""Sort all vocab frequencies.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags


FLAGS = flags.FLAGS

import numpy as np
import csv
import pickle
import nltk
import os
from nltk.corpus import words
from random import sample
import collections
import string


def main(argv):
  all_tuples = set()
  alphabet = list(string.ascii_lowercase)
  print(alphabet)
  for ch in list(string.ascii_lowercase):
    tuple_file = '/usr/local/google/home/agoldie/data/glove/sorted_count_to_word_tuples_' + ch + '.pkl'
    with open(tuple_file, 'rb') as f:
      tuples = pickle.load(f)
    all_tuples.update(tuples)

  all_tuples = list(all_tuples)
  all_tuples.sort()
  output_file = '/usr/local/google/home/agoldie/data/glove/all_sorted_count_to_word_tuples.pkl'
  with open(output_file, 'wb') as f:
    pickle.dump(all_tuples, f)
  print(len(all_tuples))
  print('output_file', output_file)


if __name__ == '__main__':
  app.run(main)
