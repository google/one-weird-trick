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
"""Generates decontextualized embeddings for each word in the input vocab and write to an sstable
"""
from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
#from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

#import transformers
import os
import pickle
import numpy
#from google3.sstable.python import sstable
#from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2


#from nltk.tokenize import word_tokenize
#from transformers import GPT2Tokenizer, TFGPT2Model
#from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
import collections

FLAGS = flags.FLAGS

def generate_random_embedding(model, tokenizer, word):
  encoded_input = tokenizer(word, return_tensors='tf')
  ids = list(encoded_input['input_ids'].numpy())

  output_hidden_states = False
  output_attentions = False
  output = model(encoded_input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
  last_hidden_state = output['last_hidden_state'][0]
  embeddings = [list(e.numpy()) for e in last_hidden_state]
  decontextualized_embedding = list(np.sum(embeddings, axis=0) / len(embeddings))
  return decontextualized_embedding

def main(argv):
  vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

  word_to_random_embedding = {}
  for word in vocab:
    word_to_random_embedding[word] = numpy.random.rand(768)

  # save this as a pickled dictionary and then use another binary to write it to sstable.
  word_to_random_embedding_file = '/usr/local/google/home/agoldie/data/gpt2/word_to_random_embedding.pkl'
  with open(word_to_random_embedding_file, 'wb') as f:
    pickle.dump(word_to_random_embedding, f)


if __name__ == '__main__':
  app.run(main)
