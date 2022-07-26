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
    'mean_embedding_file', '',
    'Cached mean embedding to save time. Stored in plaintext with a few floats per line.'
)

flags.DEFINE_string(
    'input_embedding_table', '',
    'Table in which pre-trained word embeddings are stored.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Directory to save results into'
)

flags.DEFINE_integer(
    'embedding_dim', 300,
    'dimensionality of the word embedding'
)

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

def get_average(embedding_table, embedding_dim):
  norms = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = set(table.iterkeys())
  sum_embedding = np.zeros(embedding_dim)
  for key in keys:
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    norms.append(norm(embedding))
    sum_embedding += embedding
  if len(keys) == 0:
    print(embedding_table)
    print('WHOA!')
    exit(0)
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
  input_path = FLAGS.input_embedding_table
  output_dir = FLAGS.output_dir
  if os.path.isdir(input_path):
    input_tables = [t for t in os.listdir(input_path) if 'subtract' not in t and 'remove' not in t and 'decontext' not in t and 'random' not in t]
    input_tables = [t for t in input_tables if t.endswith('.sstable')]
    output_tables = [os.path.join(output_dir, 'subtracted-' + t) for t in input_tables]
    input_tables = [os.path.join(input_path, t) for t in input_tables]
  else:
    input_tables = [input_path]
    output_tables = [os.path.join(os.path.dirname(output_dir), 'subtracted-' + os.path.basename(input_path))]
  for input_table, output_table in zip(input_tables, output_tables):
    print('input_table', input_table)
    print('output_table', output_table)
    basename = os.path.basename(input_table)
    layer = int(basename.split('-')[1])
    mean_embedding, norms = get_average(input_table, FLAGS.embedding_dim)
    subtract_mean(input_table, output_table, mean_embedding)


if __name__ == '__main__':
  app.run()
