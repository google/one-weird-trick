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
"""Evaluation of word analogy task.
"""

from absl import app
from absl import flags

import numpy as np
import csv
import pickle
import os
from random import sample
import collections
from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2
from google3.sstable.python import pywrapsstable
from scipy.spatial import distance
import sys

FLAGS = flags.FLAGS

flags.DEFINE_string('word_analogy_file', '', 'Text file containing word analogies - e.g. Athens Greece Baghdad Iraq</n>')

flags.DEFINE_string('embedding_table', '', 'SSTable containing embeddings.')

flags.DEFINE_string('embedding_map_pickle', '', 'Pickle file containing a cached mapping of word to embedding (for fast iteration).')

BENCHMARKS = {
    'mikolov-combined': {
        'description': 'mikolov - combined (19544 analogies)',
        'file': '/usr/local/google/home/agoldie/word2vec/questions-words.txt',
        'delimiter': '\t',
        'score_col': 4,
        'has_header': True},
    'mikolov-semantic': {
        'description': 'mikolov - semantic (8869 analogies)',
        'file': '/usr/local/google/home/agoldie/word2vec/questions-words-semantic.txt',
        'delimiter': ' ',
        'score_col': 3,
        'has_header': False},
    'mikolov-syntactic': {
        'description': 'mikolov - syntactic (10675 analogies)',
        'file': '/usr/local/google/home/agoldie/word2vec/questions-words-syntactic.txt',
        'delimiter': '\t',
        'score_col': 3,
        'has_header': False},
    'bats': {
        'description': 'BATS 3.0',
        'file': '/usr/local/google/home/agoldie/data/BATS_3.0/',
        'delimiter': '\t',
        'score_col': 3,
        'has_header': False}
}

def get_embedding_from_map(word, embedding_table):
  # TODO(agoldie): Use SSTableService to make reads faster.
  #import pdb; pdb.set_trace()
  contextual_table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  dir = os.path.dirname(embedding_table)
  basename = os.path.basename(embedding_table)
  #layer = basename.split('embeddings-')[1].split('-')[0]
  layer = '0'
  decontextual_table_file = os.path.join(dir, 'decontextualized-embeddings-' + layer + '.sstable')
  if os.path.exists(decontextual_table_file):
    decontextual_table = sstable.SSTable(
        decontextual_table_file,  # Proto2
        wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  word = word.encode('utf-8')
  if word in contextual_table:
    return contextual_table[word]
  elif os.path.exists(decontextual_table_file) and word in decontextual_table:
    return decontextual_table[word]

def get_embedding(word, embedding_table):
  table = sstable.SSTable(
      embedding_table,  # Proto2
      wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
  print('table', table.items())
  word = word.encode('utf-8')
  print('word', word)
  if word in table:
    return table[word]

def get_embeddings(embedding_table, analogy_words, pickle_file):
  missing = 0
  if pickle_file and os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
      embedding_map = pickle.load(f)
  else:
    table = sstable.SSTable(
        embedding_table,  # Proto2
        wrapper=sstable.TableWrapper(embedding_pb2.TokenEmbedding.FromString))
    embedding_map = {}
    for i, key in enumerate(analogy_words):
      if i % 1000 == 0:
        print(i, '/', len(analogy_words), key)
      key = key.encode('utf-8')
      if not key in table:
        print('skipping', key)
        missing += 1
        continue
      embedding = table[key]
      embedding = np.array(embedding.vector.values)
      embedding_map[key] = embedding
    with open(pickle_file, 'wb') as f:
      pickle.dump(embedding_map, f)
  print('missing', missing)
  return embedding_map

def extract_analogy_words(file_path):
  """
  Args:
    file_path:
  """
  all_words = []
  with open(file_path, 'r') as f:
    for line in f:
      if line.startswith(':'):
        continue
      words = line.strip().split()
      words = [w for w in words]
      all_words.extend(words)
  print(all_words, len(all_words))
  return all_words

def extract_analogy_words_from_directory(analogy_dir):
  """
  Args:
    file_path:
  """
  all_words = []
  for category in os.listdir(analogy_dir):
    if not os.path.isdir(os.path.join(analogy_dir, category)):
      continue
    for file in os.listdir(os.path.join(analogy_dir, category)):
      words = extract_analogy_words(os.path.join(os.path.join(analogy_dir, category), file))
      all_words.extend(words)
  print(all_words, len(all_words))
  return all_words

# ”To find a word that is similar to small in the same sense as biggest is
# similar to big, we can simply compute vector
# X = vector(”biggest”)−vector(”big”) + vector(”small”).
# Then, we search in the vector space for the word closest to X
# measured by cosine distance, and use it as the answer to the question
# (we discard the input question words during this search).”
def find_closest_analogy_word(e_a1, e_a2, e_b1, a1, a2, b1, embedding_map):
  x = (e_a2 - e_a1) + e_b1
  closest_distance, closest_word, closest_embedding = sys.float_info.max, None, None
  for word, embedding in embedding_map.items():
    word = word.decode('utf-8')
    if word == b1:
      continue
    # if word in [a1, a2, b1]:
    #   continue
    # Do I need to decode the word? I forget
    dist = distance.cosine(embedding, x)
    if dist < closest_distance:
      closest_distance = dist
      closest_word = word
      closest_embedding = embedding
  return x, closest_distance, closest_word, closest_embedding

def parse_word_analogy_file(file_path, embedding_table, embedding_map_pickle):
  print('extracting analogy words...')
  analogy_words = extract_analogy_words(file_path)
  print('loading embeddings...')
  embedding_map = get_embeddings(embedding_table, analogy_words, embedding_map_pickle)
  total = 0
  num_correct = 0
  with open(file_path, 'r') as f:
    for i, line in enumerate(f):
      if i % 100 == 0:
        print(i, '/', 20000)
      if line.startswith(':'):
        continue
      line = line.strip()
      a1, a2, b1, b2 = line.split()
      e_a1 = embedding_map[a1.encode('utf-8')]
      e_a2 = embedding_map[a2.encode('utf-8')]
      e_b1 = embedding_map[b1.encode('utf-8')]
      e_b2 = embedding_map[b2.encode('utf-8')]
      x, dist, word, embedding = find_closest_analogy_word(e_a1, e_a2, e_b1, a1, a2, b1, embedding_map)
      total += 1
      if word == b2:
        num_correct += 1
  print(num_correct, '/', total)

def extract_pairs_and_embeddings(bats_file, embedding_table):
  pairs = []
  with open(bats_file, 'r') as f:
    for line in f:
      line = line.strip()
      pair = line.split()
      pairs.append(pair)
  return pairs, embedding_map

def evaluate_pairs(pairs, embedding_map):
  total = 0
  num_correct = 0
  for i in range(len(pairs)):
    for j in range(len(pairs)):
      if i == j:
        continue
      a1, a2 = pairs[i]
      b1, b2 = pairs[j]
      e_a1 = embedding_map[a1.encode('utf-8')]
      e_a2 = embedding_map[a2.encode('utf-8')]
      e_b1 = embedding_map[b1.encode('utf-8')]
      e_b2 = embedding_map[b2.encode('utf-8')]
      x, dist, word, embedding = find_closest_analogy_word(e_a1, e_a2, e_b1, a1, a2, b1, embedding_map)
      total += 1
      if word == b2:
        num_correct += e_b1
  print(num_correct, '/', total)

def evaluate_bats(bats_dir, embedding_table, embedding_map_pickle):
  analogy_words = extract_analogy_words_from_directory(bats_dir)
  embedding_map = get_embeddings(embedding_table, analogy_words, embedding_map_pickle)
  all_pairs = []
  for bats_file in os.listdir(bats_dir):
    pairs, embedding_map = extract_pairs_and_embeddings(os.path.join(bats_dir, bats_file), embedding_table)
    result = evaluate_pairs(pairs, embedding_map)
    all_pairs.append(pairs)
  print(len(all_pairs))
  return pairs


def main(argv):
  file_path = FLAGS.word_analogy_file
  embedding_table = FLAGS.embedding_table
  embedding_map_pickle = FLAGS.embedding_map_pickle
  analogy_tasks = [file_path]
  if ',' in file_path:
    analogy_tasks = file_path.split(',')
  for analogy_task in analogy_tasks:
    if 'BATS' in analogy_task:
      evaluate_bats(analogy_task, embedding_table, embedding_map_pickle)
    else:
      parse_word_analogy_file(analogy_task, embedding_table, embedding_map_pickle)


if __name__ == '__main__':
  app.run(main)
