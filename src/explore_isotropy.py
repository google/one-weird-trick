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
"""Explore isotropy of word embeddings.
"""

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

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def get_embeddings(embedding_table):
  print('getting embeddings', embedding_table)
  embeddings = []
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  keys = list(table.iterkeys())
  for i, key in enumerate(keys):
    if i % 1000 == 0:
      print(i, '/', len(keys))
    embedding = table[key]
    embedding = np.array(embedding.vector.values)
    embeddings.append(embedding)
  return keys, embeddings

def calculate_principal_components(embeddings):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=300)
  _ = pca.fit_transform(embeddings)
  return pca

def main(argv):
  before = '/usr/local/google/home/agoldie/data/glove/glove.sstable'
  after_ours = '/usr/local/google/home/agoldie/data/glove/removed-2-subtracted-start0-vocab50000.sstable'
  after_abtt = '/usr/local/google/home/agoldie/data/glove/removed-subtracted-full.sstable'

  tables = [before, after_ours, after_abtt]
  table_map = {}
  for table in tables:
    print(table)
    keys, embeddings = get_embeddings(table)
    print('embeddings', len(embeddings), len(embeddings[0]))
    pca = calculate_principal_components(embeddings)
    principal_components = pca.components_
    singular_values = pca.singular_values_
    print('pca', len(principal_components), len(principal_components[0]))
    print(len(singular_values), singular_values)
    table_map, [table] = (principal_components, singular_values)

  import pickle
  pcs_and_svs_map_file = '~/data/glove/pcs_and_svs_map.pkl'
  with open(pcs_and_svs_map_file, 'wb') as f:
    pickle.dump(table_map, f)


if __name__ == '__main__':
  app.run(main)
