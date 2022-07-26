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
"""Write contextual word embeddings.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags
import os
import pickle

from google3.sstable.python import sstable
from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_dir = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/'
  word_to_embedding_file = os.path.join(data_dir, 'word_to_embedding_100k.pkl')
  with open(word_to_embedding_file, 'rb') as f:
    word_to_embedding = pickle.load(f)

  embedding_dir = os.path.join('/usr/local/google/home/agoldie/data/gpt2')
  embedding_sstable = os.path.join(embedding_dir, 'embeddings-v2.sstable')
  with sstable.SortingBuilder(embedding_sstable) as builder:
    for i, word in enumerate(word_to_embedding):
      print(i, word)
      embedding = word_to_embedding[word]
      token_embedding = embedding_pb2.TokenEmbedding()
      token_embedding.token = word.encode('utf-8')
      token_embedding.vector.values.extend(embedding)
      builder.Add(token_embedding.token,
                  token_embedding.SerializeToString())


if __name__ == '__main__':
  app.run(main)
