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

"""Removes principal components for vocab_subsets."""
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
    'Directory of tables in which pre-trained word embeddings are stored.'
)

flags.DEFINE_integer(
    'D', 3,
    'Number of principal components to remove.'
)

flags.DEFINE_integer(
    'embedding_dim', 300,
    'Dimensionality of embedding.'
)

flags.DEFINE_string(
    'vocab_dir', '',
    'File containing vocabulary subset from which to calculate pcs to remove.'
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
  for i, key in enumerate(keys):
    if i % 100 == 0:
      print(i, '/', len(keys))
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    embeddings.append(embedding)
  return keys, embeddings

def calculate_principal_components(embeddings, num_components):
  #import pdb; pdb.set_trace()
  from sklearn.decomposition import PCA
  pca = PCA(n_components=num_components)
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

def get_subset_of_embeddings(keys, embeddings, vocab):
  subset_embeddings = []
  for key, embedding in zip(keys, embeddings):
    if key.decode().lower() not in vocab:
      continue
    subset_embeddings.append(embedding)
  return subset_embeddings

def main(_):
  assert(os.path.isdir(FLAGS.input_dir))
  input_dir = FLAGS.input_dir

  # For now, assume that the results will be stored in the same directory.
  output_dir = input_dir

  subtracted_files = [s for s in os.listdir(input_dir) if 'subtracted' in s]
  subtracted_files = [os.path.join(input_dir, s) for s in subtracted_files]

  assert(os.path.isdir(FLAGS.vocab_dir))
  vocab_dir = FLAGS.vocab_dir

  for input_table in subtracted_files:
    print('input_table', input_table)
    # subtracted-simverb_vocab-subset.sstable
    vocab_name = os.path.basename(input_table).split('-')[1]
    print('vocab_name', vocab_name)
    vocab_file = os.path.join(vocab_dir, vocab_name + '.pkl')
    with open(vocab_file, 'rb') as f:
      vocab = pickle.load(f)

    keys, embeddings = get_embeddings(input_table)
    print('embeddings', len(embeddings), len(embeddings[0]))

    subset_embeddings = get_subset_of_embeddings(keys, embeddings, vocab)
    num_components = min(FLAGS.embedding_dim, len(subset_embeddings))
    pca = calculate_principal_components(subset_embeddings, num_components)
    principal_components = pca.components_
    print('pca', len(principal_components), len(principal_components[0]))
    input_table_base = os.path.basename(input_table)
    output_file = os.path.join(output_dir, 'removed-' + str(FLAGS.D) + '-' + input_table_base)
    print('output_file', output_file)
    remove_principal_components(keys, embeddings, principal_components, output_file, FLAGS.D)



if __name__ == '__main__':
  app.run()
