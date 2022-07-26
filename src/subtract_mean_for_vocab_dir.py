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
import google3

from scipy.spatial import distance
from scipy.stats import spearmanr

from google3.pyglib import app
from google3.pyglib import flags
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2
from google3.sstable.python import pywrapsstable

import numpy as np
from numpy.linalg import norm
from numpy import dot
import csv
import pickle
import os

flags.DEFINE_string(
    'input_dir', '',
    'Directory containing sstables in which pre-trained word embeddings are stored.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Table in which pre-trained word embeddings are stored (with the mean vector subtracted).'
)

flags.DEFINE_integer(
    'embedding_dim', 300,
    'dimensionality of the word embedding'
)

flags.DEFINE_string(
    'vocab_dir', '', 'Directory in which all of the vocabulary pickles are stored.')

flags.DEFINE_boolean(
    'overwrite_output', False, 'If True, for already written files, overwrite with new results. If False, skip ahead.')



FLAGS = flags.FLAGS

def get_embedding(word, embedding_table):
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  print('table', table.items())
  word = word.encode('utf-8')
  print('word', word)
  if word in table:
    return table[word]

def get_average(embedding_table, embedding_dim, vocab):
  norms = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = set(table.iterkeys())
  sum_embedding = np.zeros(embedding_dim)
  print(len(vocab))
  for i, key in enumerate(keys):
    if i % 1000 == 0:
      print(i, '/', len(keys))
    #print(key)
    if vocab and key.decode().lower() not in vocab:
      continue
    #print(key)
    #import pdb; pdb.set_trace()
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    norms.append(norm(embedding))
    sum_embedding += embedding
  mean_embedding = sum_embedding / len(keys)
  print('mean_embedding', mean_embedding)
  return mean_embedding, norms

def get_embeddings(embedding_table):
  embeddings = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = list(table.iterkeys())
  for key in keys:
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    embeddings.append(embedding)
  return keys, embeddings

def write_embeddings_to_sstable(embedding_map, embedding_sstable, vocab):
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

def subtract_mean(input_file, output_file, mean):
  input_table = sstable.SSTable(
      input_file,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  # builder = sstable.SortingBuilder(output_file)
  builder = pywrapsstable.SSTableBuilder(
    output_file, None, pywrapsstable.SSTableBuilderOptions())
  for i, key in enumerate(input_table.keys()):
    if i % 1000 == 0:
      print('Adding', i, 'th subtracted embedding to sstable.')
    input_embedding = np.array(input_table[key].vector.values)
    output_embedding = input_embedding - mean # make sure that this subtracts element-wise.
    output_embedding_pb = embedding_pb2.TokenEmbedding()
    output_embedding_pb.token = key
    output_embedding_pb.vector.values.extend(output_embedding)
    builder.Add(key,
                output_embedding_pb.SerializeToString())

def read_mean_embedding_file(mean_embedding_file):
  mean_embedding = []
  with open(mean_embedding_file, 'r') as f:
    for line in f:
      components = line.strip().split(' ')
      components = [x for x in components if x]
      floats = list(map(float, components))
      mean_embedding.extend(floats)
  return mean_embedding


def main(_):
  assert(FLAGS.input_dir)
  assert(os.path.isdir(FLAGS.input_dir))
  input_dir = FLAGS.input_dir
  output_dir = FLAGS.output_dir
  input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
  input_files = [f for f in input_files if 'subtracted' not in f and 'removed' not in f]

  assert(FLAGS.vocab_dir)
  assert(os.path.isdir(FLAGS.vocab_dir))
  vocab_dir = FLAGS.vocab_dir

  print('input_dir', input_dir)
  print('vocab_dir', vocab_dir)

  for input_file in input_files:
    print('input_file', input_file)
    if os.path.basename(input_file) == 'glove.sstable':
      print('Skipping glove.sstable...')
      continue
    input_table_name = os.path.basename(input_file)[len('union-all-benchmarks-'):]
    output_file = os.path.join(output_dir, 'subtracted-' + input_table_name)
    print('output_file', output_file)
    if os.path.exists(output_file):
      print('Skipping...')
      continue
    base = input_table_name[:-len('.sstable')] + '.pkl'
    vocab_file = os.path.join(vocab_dir, base)
    print('vocab_file', vocab_file)
    with open(vocab_file, 'rb') as f:
      vocab = pickle.load(f)
    mean_embedding, norms = get_average(input_file, FLAGS.embedding_dim, vocab)
    vocab_name = os.path.basename(vocab_file[:-len('.pkl')])
    print('vocab', vocab_name)
    subtract_mean(input_file, output_file, mean_embedding)


if __name__ == '__main__':
  app.run()
