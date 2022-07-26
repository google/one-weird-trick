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
"""Write out any pickle map of word to embedding to a given sstable.

Example usage:
blaze build -c opt experimental/users/agoldie/embedding_benchmarks/emnlp2021:write_embeddings_to_sstable
&& ./blaze-bin/experimental/users/agoldie/embedding_benchmarks/emnlp2021/write_embeddings_to_sstable
  --embedding_map ~/data/gpt2/word_to_decontextualized_embedding.pkl
  --output_sstable ~/data/gpt2/decontextualized-embeddings.sstable
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

flags.DEFINE_string(
    'embedding_map', '',
    'Pickle file containing map of words to their embeddings.'
)

flags.DEFINE_string(
    'output_sstable', '',
    'Output sstable in which to write out the embeddings.'
)

def main(argv):
  # if len(argv) > 1:
  #   raise app.UsageError('Too many command-line arguments.')

  with open(FLAGS.embedding_map, 'rb') as f:
    word_to_embedding = pickle.load(f)

  with sstable.SortingBuilder(FLAGS.output_sstable) as builder:
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
