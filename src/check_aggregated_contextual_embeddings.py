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
"""Aggregates contextual word embeddings with mean pooling (for now).
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
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import transformers
import os
import pickle
import numpy as np

from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer, TFGPT2Model
import collections

FLAGS = flags.FLAGS


def main(argv):
  data_dir = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/'
  occurrence_map_file = os.path.join(data_dir, 'occurrence_map_100k.pkl')
  with open(occurrence_map_file, 'rb') as g:
    occurrence_map = pickle.load(g)
  tokenization_map_file = os.path.join(data_dir, 'tokenizations_100k.pkl')
  with open(tokenization_map_file, 'rb') as h:
    tokenization_map = pickle.load(h)
  word_to_token_ids_file = os.path.join(data_dir, 'word_to_token_ids.pkl')
  with open(word_to_token_ids_file, 'rb') as h:
    word_to_token_ids = pickle.load(h)

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

  if 'smelter' in word_to_token_ids:
    print('smelter is here!')
  else:
    print('no smelter!')

  for word in word_to_token_ids.keys():
    print(word)

    sent_id_to_token_ids = word_to_token_ids[word]

    for sent_id in sent_id_to_token_ids.keys():
      ids = tokenization_map[sent_id]
      ids = ids.split(', ')
      ids = list(map(int, ids))
      tokens = tokenizer.convert_ids_to_tokens(ids)
      str_tokens = []
      for word_token in tokens:
        str_token = tokenizer.convert_tokens_to_string(word_token).strip()
        str_tokens.append(str_token)
      tokenized_sentence = ' '.join(str_tokens)
      print(tokenized_sentence)

      token_ids = sent_id_to_token_ids[sent_id]
      word_tokens = [tokens[t] for t in token_ids]
      for word_token in word_tokens:
        word_string = tokenizer.convert_tokens_to_string(word_token).strip()
        if word_string != word:
          print(word_string)


if __name__ == '__main__':
  app.run(main)
