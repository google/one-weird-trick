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
"""Separate syntactic from semantic features.
"""

from absl import app
from absl import flags

import numpy as np
import csv
import pickle
import os
from random import sample
import collections
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2
from google3.sstable.python import pywrapsstable
from scipy.spatial import distance
import sys

FLAGS = flags.FLAGS

flags.DEFINE_string('word_analogy_file', '', 'Text file containing word analogies - e.g. Athens Greece Baghdad Iraq</n>')

flags.DEFINE_string('semantic_output_file', '', 'Text file containing semantic word analogies tasks.')

flags.DEFINE_string('syntactic_output_file', '', 'Text file containing syntactic word analogies tasks.')

semantic_categories = [': capital-common-countries', ': capital-world', ': currency', ': city-in-state', ': family']
syntactic_categories = [': gram1-adjective-to-adverb', ': gram2-opposite', ': gram3-comparative', ': gram4-superlative', ': gram5-present-participle', ': gram6-nationality-adjective', ': gram7-past-tense', ': gram8-plural', ': gram9-plural-verbs']

def separate_word_analogy_file(file_path):
  semantic_list = []
  syntactic_list = []

  with open(file_path, 'r') as f:
    category = None
    for i, line in enumerate(f):
      line = line.strip()
      if line.startswith(':'):
        if line in semantic_categories:
          category = 'semantic'
        elif line in syntactic_categories:
          category = 'syntactic'
        continue
      if category == 'semantic':
        semantic_list.append(line)
      elif category == 'syntactic':
        syntactic_list.append(line)
      else:
        print('no category?!')
  return semantic_list, syntactic_list

def main(argv):
  file_path = FLAGS.word_analogy_file
  semantic_list, syntactic_list = separate_word_analogy_file(file_path)
  with open(FLAGS.semantic_output_file, 'w') as f:
    for line in semantic_list:
      f.write(line + '\n')
  with open(FLAGS.syntactic_output_file, 'w') as f:
    for line in syntactic_list:
      f.write(line + '\n')

if __name__ == '__main__':
  app.run(main)
