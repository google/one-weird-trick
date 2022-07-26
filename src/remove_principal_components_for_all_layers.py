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
    'input_embedding_table', '',
    'Table in which pre-trained word embeddings are stored.'
)

flags.DEFINE_string(
    'output_dir', '',
    'Directory in which pre-trained word embeddings are stored (with the mean vector subtracted).'
)

flags.DEFINE_integer(
    'D', 3,
    'Number of principal components to remove.'
)

FLAGS = flags.FLAGS

def get_embedding(word, embedding_table):
  # TODO(agoldie): Use SSTableService to make reads faster.
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  print('table', table.items())
  word = word.encode('utf-8')
  print('word', word)
  if word in table:
    return table[word]

def get_embeddings(embedding_table):
  embeddings = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = list(table.iterkeys())
  for key in keys:
    # if vocab and key not in vocab:
    #   continue
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    embeddings.append(embedding)
  return keys, embeddings

def calculate_principal_components(embeddings):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=300)
  _ = pca.fit_transform(embeddings)
  return pca

def remove_principal_components(keys, embeddings, principal_components, output_file, D):
  builder = pywrapsstable.SSTableBuilder(
    output_file, None, pywrapsstable.SSTableBuilderOptions())
  for i in range(len(embeddings)):
    embedding = embeddings[i]
    output_embedding = np.array(embedding)
    for j in range(D):
      principal_component = principal_components[j]
      difference = dot(principal_component, embedding) * principal_component
      output_embedding -= difference
    output_embedding_pb = embedding_pb2.TokenEmbedding()
    output_embedding_pb.token = keys[i]
    output_embedding_pb.vector.values.extend(output_embedding)
    builder.Add(keys[i],
               output_embedding_pb.SerializeToString())

def main(_):
  input_path = FLAGS.input_embedding_table
  output_dir = FLAGS.output_dir
  # vocab_file = FLAGS.vocab_file
  # vocab = None
  # if vocab_file:
  #   with open(vocab_file, 'rb') as f:
  #     vocab = pickle.load(f)
  if os.path.isdir(input_path):
    input_tables = [t for t in os.listdir(input_path) if 'subtract' in t and 'remove' not in t]
    input_tables = [t for t in input_tables if t.endswith('.sstable')]
    output_tables = [os.path.join(output_dir, 'removed-' + str(FLAGS.D) + '-' + t) for t in input_tables]
    input_tables = [os.path.join(input_path, t) for t in input_tables]
  else:
    print('unexpected!')
    exit(0)
    # input_tables = [input_path]
    # output_tables = [os.path.join(os.path.dirname(output_dir), 'removed-' + str(FLAGS.D) + os.path.basename(input_path))]
  for input_table, output_table in zip(input_tables, output_tables):
    print('input_table', input_table)
    print('output_table', output_table)
    keys, embeddings = get_embeddings(input_table)
    print('embeddings', len(embeddings), len(embeddings[0]))
    pca = calculate_principal_components(embeddings)
    principal_components = pca.components_
    singular_values = pca.singular_values_
    print('pca', len(principal_components), len(principal_components[0]))
    print(len(singular_values), singular_values)
    remove_principal_components(keys, embeddings, principal_components, output_table, FLAGS.D)



if __name__ == '__main__':
  app.run()
