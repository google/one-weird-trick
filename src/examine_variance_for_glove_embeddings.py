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
"""Examine variance for GLOVE embeddings.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2
import numpy as np
import sys
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'embedding_sstable', '',
    'SSTable containing static word embeddings.'
)

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
  mean_embedding = sum_embedding / len(keys)
  print('mean_embedding', mean_embedding)
  return mean_embedding, norms

def main(argv):
  embedding_table = FLAGS.embedding_sstable
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = set(table.iterkeys())
  embeddings = []
  for i, key in enumerate(keys):
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    embeddings.append(embedding)
  embeddings = np.array(embeddings)
  mean_embeddings = list(np.mean(embeddings, axis=0))
  np.set_printoptions(threshold=sys.maxsize)
  print('mean', mean_embeddings)
  std_embeddings = list(np.std(embeddings, axis=0))
  np.set_printoptions(threshold=sys.maxsize)
  print('std', std_embeddings)
  min_embeddings = list(np.amin(embeddings, axis=0))
  np.set_printoptions(threshold=sys.maxsize)
  print('min', min_embeddings)
  max_embeddings = list(np.amax(embeddings, axis=0))
  np.set_printoptions(threshold=sys.maxsize)
  print('max', max_embeddings)
  stats_file = '/usr/local/google/home/agoldie/data/glove/stats.txt'
  stats = {}
  with open(stats_file, 'r') as f:
    for line in f:
      line = line.strip().strip(']')
      category, data = line.split(' [')
      embedding = [float(t) for t in data.split(', ')]
      print(category)
      print(type(embedding), embedding)
      stats[category] = embedding
  print(stats.keys())
  for k in stats.keys():
    print(k, len(stats[k]))
  mean_embedding = np.array(stats['mean'])
  std_embedding = np.array(stats['std'])
  num_stds = 1.5
  lower_bound = mean_embedding - num_stds * std_embedding
  upper_bound = mean_embedding + num_stds * std_embedding
  print(lower_bound)
  print(upper_bound)
  embedding_table = FLAGS.embedding_sstable
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = set(table.iterkeys())
  for i, key in enumerate(keys):
    print(i)
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    if (embedding > lower_bound).all() and (embedding < upper_bound).all():
      print(key)
      break

  embedding_table = FLAGS.embedding_sstable
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  distance_to_words = []
  for word in table:
    embedding = table[word]
    embedding = np.array(embedding.vector.values)
    euclidean_dist = np.linalg.norm(embedding)
    distance_to_words.append((euclidean_dist, word))
  distance_to_words.sort()
  import pickle
  distance_to_words_file = '/usr/local/google/home/agoldie/data/glove/distance_to_words.pkl'
  with open(distance_to_words_file, 'wb') as f:
    pickle.dump(distance_to_words, f)
  print(distance_to_words)
  with open(distance_to_words_file, 'rb') as f:
    distance_to_words = pickle.load(f)
  n = 5000
  vocab = set([pair[1].decode() for pair in distance_to_words[:n]])
  print(vocab)



if __name__ == '__main__':
  app.run(main)
