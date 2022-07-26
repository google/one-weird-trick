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
"""Trains word2vec embeddings on input data files and outputs these embeddings
to an sstable in a format that can be evaluated with
lexical_similarity_benchmarks.py.

Example Usage:
blaze build -c opt experimental/users/agoldie/embedding_benchmarks:save_embeddings_to_sstable && \
./blaze-bin/experimental/users/agoldie/embedding_benchmarks/save_embeddings_to_sstable \
    --input_embeddings /usr/local/google/home/embeddings.txt \
    --vocab_file /tmp/vocab.txt \
    --embedding_sstable /tmp/embedding.sstable
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags
import csv
import math
import random
import numpy as np
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import os.path
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2
import pickle

flags.DEFINE_string(
    'input_embeddings', '',
    'file containing raw embeddings. format is <WORD>\t?<EMBEDDING></n>'
)

flags.DEFINE_string(
    'input_embedding_sstable', '',
    'sstable containing embeddings'
)

flags.DEFINE_string(
    'embedding_sstable', '',
    'file to write embedding sstable into.'
)

flags.DEFINE_string(
    'embedding_map_file', '',
    'pkl file containing map of words to embeddings'
)

flags.DEFINE_string(
    'vocab_file', '',
    'file to read vocab from.'
)

FLAGS = flags.FLAGS

def write_embeddings_to_sstable(embedding_map, embedding_sstable, vocab):
  print('vocab', len(vocab))
  print('embedding_map', len(embedding_map))
  with sstable.SortingBuilder(embedding_sstable) as builder:
    for word, embedding in embedding_map.items():
      #print(word, embedding)
      if word == '*OOV*' or not word in vocab:
        continue
      token_embedding = embedding_pb2.TokenEmbedding()
      token_embedding.token = word.encode('utf-8')
      token_embedding.vector.values.extend(embedding)
      builder.Add(token_embedding.token,
                  token_embedding.SerializeToString())


def read_embedding_sstable(embedding_sstable):
  embedding_map = {}
  table = sstable.SSTable(
      embedding_sstable,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  for k, v in table.iteritems():
    print(i, v.token, v.SerializeToString())
  return embedding_map

def load_embedding_map(embedding_map_file):
  with open(embedding_map_file, 'rb') as g:
    embedding_map = pickle.load(g)
  return embedding_map

def parse_embeddings(embedding_file):
  embedding_map = {}
  degenerate_map = {}
  with open(embedding_file, 'rb') as f:
    for i, line in enumerate(f):
      print(line)
      split_line = line.split()
      if len(split_line) != 301:
        degenerate_map[i] = line
        continue
      else:
        word = split_line[0]
        embedding = list(map(float, split_line[1:]))
        embedding_map[word] = embedding
  print('num lines', i)
  print('embedding_map', len(embedding_map))
  print('degenerate_map', len(degenerate_map))
  return embedding_map, degenerate_map

def main(_):
  vocab_file = FLAGS.vocab_file
  with open(vocab_file, 'rb') as g:
    vocab = pickle.load(g)
  print('vocab', len(vocab))
  embedding_file = ''
  if FLAGS.embedding_map_file:
    embedding_file = FLAGS.embedding_map_file
    embedding_map = load_embedding_map(embedding_file)
  elif FLAGS.input_embeddings:
    embedding_file = FLAGS.input_embeddings
    print('reading embeddings', embedding_file)
    embedding_map, degenerate_map = parse_embeddings(embedding_file)
  else:
    print('No embedding file provided, must specify either --embedding_map_file or --input_embeddings!')
    exit(0)

  write_embeddings_to_sstable(embedding_map, FLAGS.embedding_sstable, vocab)
  print('wrote embeddings from', embedding_file, 'to', FLAGS.embedding_sstable)

if __name__ == '__main__':
  app.run(main)
