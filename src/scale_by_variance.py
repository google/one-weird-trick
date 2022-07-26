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
import math

flags.DEFINE_string(
    'full_embedding_table', '',
    'Table containing all embeddings, including those not in word similarity benchmarks.'
)

flags.DEFINE_string(
    'input_embedding_table', '',
    'Table in which pre-trained word embeddings are stored.'
)

flags.DEFINE_string(
    'output_embedding_table', '',
    'Table in which pre-trained word embeddings are stored (with the mean vector subtracted).'
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

def get_average(embedding_table):
  norms = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = set(table.iterkeys())
  sum_embedding = np.zeros(300)
  for key in keys:
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    norms.append(norm(embedding))
    sum_embedding += embedding
  mean_embedding = sum_embedding / len(keys)
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

def calculate_mean(embeddings):
  embedding_dim = len(embeddings[0])
  sum_embedding = np.zeros(embedding_dim)
  for embedding in embeddings:
    sum_embedding += embedding
  mean_embedding = sum_embedding / len(embeddings)
  return mean_embedding

def read_mean(mean_file):
  with open(mean_file, 'r') as f:
    mean_str = f.read()
  mean_str = mean_str.strip().strip('[').strip(']')
  mean_embedding = list(map(float, mean_str.split()))
  return mean_embedding

# def calculate_variance(embeddings, mean):
#   embedding_dim = len(embeddings[0])
#   sum_variances = np.zeros(embedding_dim)
#   for d in range(embedding_dim):
#     for e in embeddings:
#       variance = math.pow(e[d] - mean[d], 2)
#       sum_variances[d] += variance
#   variances = sum_variances / 300
#   return variances

def calculate_principal_components(embeddings):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=300)
  _ = pca.fit_transform(embeddings)
  return pca

def calculate_variance(embeddings, mean):
  import statistics
  embedding_dim = len(embeddings[0])
  embeddings = np.array(embeddings)
  variances = []
  for i in range(embedding_dim):
    dim = embeddings[:,i]
    #print(len(dim))
    variances.append(statistics.pvariance(dim))
  return variances

def scale_by_variance(input_file, output_file, variance, standard_devs):
  input_table = sstable.SSTable(
      input_file,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  builder = pywrapsstable.SSTableBuilder(
    output_file, None, pywrapsstable.SSTableBuilderOptions())
  for key in input_table.iterkeys():
    input_embedding = np.array(input_table[key].vector.values)
    output_embedding = input_embedding / variance
    output_embedding_pb = embedding_pb2.TokenEmbedding()
    output_embedding_pb.token = key
    output_embedding_pb.vector.values.extend(output_embedding)
    builder.Add(key,
                output_embedding_pb.SerializeToString())

def main(_):
  keys, embeddings = get_embeddings(FLAGS.input_embedding_table)
  pca = calculate_principal_components(embeddings)
  principal_components = pca.components_
  pca_mean = calculate_mean(principal_components)
  variance = calculate_variance(principal_components, pca_mean)
  standard_devs = list(map(math.sqrt, variance))
  print('variance', len(variance), variance, standard_devs)
  print('singular_values', len(pca.singular_values_), pca.singular_values_)
  #scale_by_variance(FLAGS.input_embedding_table, FLAGS.output_embedding_table, variance, standard_devs)

if __name__ == '__main__':
  app.run()
