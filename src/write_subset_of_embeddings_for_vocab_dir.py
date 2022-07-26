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
import codecs

flags.DEFINE_string(
    'full_embeddings', '',
    'sstable file containing all embeddings.'
)

flags.DEFINE_string(
    'subset_embeddings', '',
    'sstable file to write subset of embeddings into.'
)

flags.DEFINE_string(
    'vocab_dir', '',
    'pkl file to read vocab from.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Directory to write sstables out to.'
)

FLAGS = flags.FLAGS

def write_subset_of_embeddings(full_embedding_map, subset_sstable, vocab):
  with sstable.SortingBuilder(subset_sstable) as builder:
    for word, embedding in embedding_map.items():
      #print(word, embedding)
      if word == '*OOV*' or not word in vocab:
        continue
      token_embedding = embedding_pb2.TokenEmbedding()
      token_embedding.token = word.encode('utf-8')
      token_embedding.vector.values.extend(embedding)
      builder.Add(token_embedding.token,
                  token_embedding.SerializeToString())

def load_embedding_map(embedding_map_file):
  with open(embedding_map_file, 'rb') as g:
    embedding_map = pickle.load(g)
  return embedding_map

def read_embeddings(embedding_file):
  embedding_map = {}
  degenerate_map = {}
  with open(embedding_file, 'r') as f:
    for i, line in enumerate(f):
      split_line = line.split()
      if len(split_line) != 301:
        degenerate_map[i] = line
        continue
      else:
        word = split_line[0]
        embedding = list(map(float, split_line[1:]))
        embedding_map[word] = embedding
  return embedding_map, degenerate_map

def write_subset_of_embeddings(old_sstable, new_sstable, subset_vocab):
  print('subset_vocab', len(subset_vocab))
  old_sstable = sstable.SSTable(
      old_sstable,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  with sstable.SortingBuilder(new_sstable) as builder:
    for i, word in enumerate(subset_vocab):
      if i % 1000 == 0:
        print(i, '/', len(subset_vocab))
      if type(word) == str:
        word = word.encode('utf-8')
      if word in old_sstable:
        v = old_sstable[word]
        #print('adding', word)
        builder.Add(v.token, v.SerializeToString())

def main(_):
  #import pdb; pdb.set_trace()
  full_embeddings = FLAGS.full_embeddings
  output_dir = FLAGS.output_dir
  vocab_dir = FLAGS.vocab_dir
  assert(os.path.isdir(FLAGS.vocab_dir))
  # TODO: Should probably remove this special logic requiring that filename start with 'union'!
  vocab_files = [v for v in os.listdir(vocab_dir) if v.startswith('union') and v.endswith('.pkl')]
  for vocab_file in vocab_files:
    vocab_file = os.path.join(vocab_dir, vocab_file)
    print('vocab_file', vocab_file)
    with open(vocab_file, 'rb') as g:
      vocab = pickle.load(g)
    print('vocab', len(vocab))

    base = os.path.basename(vocab_file)
    subset_embeddings = os.path.join(output_dir, base[:-len('.pkl')] + '.sstable')
    write_subset_of_embeddings(full_embeddings, subset_embeddings, vocab)

    print('wrote embeddings from', full_embeddings, 'to', subset_embeddings)

if __name__ == '__main__':
  app.run(main)
