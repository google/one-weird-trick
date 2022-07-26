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
"""Generate contextual word embeddings with GPT-2.
"""
# transformers.GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, \
#                         n_embd=768, n_layer=12, n_head=12, n_inner=None, \
#                         activation_function='gelu_new', resid_pdrop=0.1, \
#                         embd_pdrop=0.1, attn_pdrop=0.1, \
#                         layer_norm_epsilon=1e-05, initializer_range=0.02, \
#                         summary_type='cls_index', summary_use_proj=True, \
#                         summary_activation=None, summary_proj_to_labels=True,\
#                         summary_first_dropout=0.1, \
#                         gradient_checkpointing=False, \
#                         use_cache=True, \
#                         bos_token_id=50256, eos_token_id=50256, **kwargs)

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
#from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import transformers
import os
import pickle

from nltk.tokenize import word_tokenize
import collections

FLAGS = flags.FLAGS

# def load_occurrence_map(occurrence_map_file):
#   with open(occurrence_map_file, 'rb') as g:
#     occurrence_map = pickle.load(g)
#   return occurrence_map

# def generate_occurrence_map(input_file):
#   with open(input_file, 'r') as f:
#     for sent_id, sentence in enumerate(f):
#       if sent_id == 100000:
#         break
#       tokens = word_tokenize(sentence)
#       #print(i, sentence)
#       for word in vocab:
#         if word in tokens:
#           occurrence_map[word].append(sent_id)
#   occurrence_map_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/occurrence_map_100k.pkl'
#   with open(occurrence_map_file, 'wb') as o:
#     pickle.dump(occurrence_map, o)

def main(argv):
  # map of word -> {sentence_index -> token_index} - be careful with that token index
  occurrence_map = collections.defaultdict(list)
  vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(vocab_file, 'rb') as g:
    vocab = pickle.load(g)
  #vocab = set(['smelter', 'cat'])
  input_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.txt'
  with open(input_file, 'r') as f:
    for sent_id, sentence in enumerate(f):
      if sent_id == 100000:
        break
      tokens = word_tokenize(sentence)
      #print(i, sentence)
      for word in vocab:
        if word in tokens:
          occurrence_map[word].append(sent_id)
  occurrence_map_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/occurrence_map_100k.pkl'
  print(occurrence_map)
  with open(occurrence_map_file, 'wb') as o:
    pickle.dump(occurrence_map, o)


if __name__ == '__main__':
  app.run(main)
