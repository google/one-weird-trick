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
"""Aggregates contextual word embeddings with mean pooling (for now).
"""
from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import os
import pickle
import numpy as np
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2

import collections

FLAGS = flags.FLAGS

# ASSUMPTIONS: (1) mean-pooling over contexts.
def main(argv):
  embedding_dir = '/usr/local/google/home/agoldie/data/bert-24/'
  word_dir = os.path.join(embedding_dir, 'words')
  NUM_LAYERS = 25
  for layer in range(NUM_LAYERS):
    print(layer)
    embedding_sstable = os.path.join(embedding_dir, 'embeddings-' + str(layer) + '-100k.sstable')
    with sstable.SortingBuilder(embedding_sstable) as builder:
      word_files = os.listdir(word_dir)
      word_files = [w for w in word_files if '-' + str(layer) + '.txt' in w]
      for i, word_file in enumerate(word_files):
        if i % 100 == 0:
          print(i, '/', len(word_files))
        word_path = os.path.join(word_dir, word_file)
        embeddings = []
        with open(word_path, 'r') as f:
          for e in f:
            embedding = e.strip()
            embedding = list(map(float, embedding.split(', ')))
            embeddings.append(embedding)
        mean_embedding = list(np.sum(embeddings, axis=0) / len(embeddings))
        token_embedding = embedding_pb2.TokenEmbedding()
        word = word_file.split('-')[0]
        token_embedding.token = word.encode('utf-8')
        token_embedding.vector.values.extend(mean_embedding)
        builder.Add(token_embedding.token,
                    token_embedding.SerializeToString())





if __name__ == '__main__':
  app.run(main)
