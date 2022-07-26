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

import os
import pickle
import numpy as np

from transformers import GPT2Tokenizer, TFGPT2Model
import collections

FLAGS = flags.FLAGS

def get_str_tokens(tokenizer, ids):
  ids = ids.split(', ')
  ids = list(map(int, ids))
  tokens = tokenizer.convert_ids_to_tokens(ids)
  token_string = tokenizer.convert_tokens_to_string(tokens)
  str_tokens = []
  for word_token in tokens:
    str_token = tokenizer.convert_tokens_to_string(word_token).strip()
    str_tokens.append(str_token)
  return str_tokens

def build_word_to_token_indices(tokenizer, word_to_sent_ids, sent_id_to_token_ids):
  word_to_token_indices = collections.defaultdict(map)
  for i, word in enumerate(word_to_sent_ids.keys()):
    #print(word)
    num_tokens = 0
    sent_id_to_token_indices = collections.defaultdict(list)
    for sent_id in word_to_sent_ids[word]:
      #print(sent_id)
      ids = sent_id_to_token_ids[sent_id]
      str_tokens = get_str_tokens(tokenizer, ids)

      for start_index, str_token in enumerate(str_tokens):
        subword = word
        end_index = start_index
        # if the target word doesn't start with the current token, continue
        # otherwise...
        while end_index < len(str_tokens) and subword:
          token = str_tokens[end_index]
          if not subword.startswith(token):
            break
          subword = subword[len(token):]
          end_index += 1

        if not subword:
          #print(word, 'is present at', range(start_index, end_index), ':', str_tokens[start_index:end_index])
          sent_id_to_token_indices[sent_id].append((start_index, end_index))
    word_to_token_indices[word] = sent_id_to_token_indices
  return word_to_token_indices

def main(argv):
  data_dir = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/'
  embedding_dir = os.path.join(data_dir, 'embeddings')

  word_to_sent_ids_file = os.path.join(data_dir, 'occurrence_map_100k.pkl')
  with open(word_to_sent_ids_file, 'rb') as g:
    word_to_sent_ids = pickle.load(g)

  sent_id_to_token_ids_file = os.path.join(data_dir, 'tokenizations_100k.pkl')
  with open(sent_id_to_token_ids_file, 'rb') as h:
    sent_id_to_token_ids = pickle.load(h)

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

  word_to_token_indices = build_word_to_token_indices(tokenizer, word_to_sent_ids, sent_id_to_token_ids)

  embedding_sstable = '/usr/local/google/home/agoldie/data/gpt2/embeddings.sstable'
  word_to_embedding = {}

  for i, word in enumerate(word_to_token_indices):
    print(i, word)
    sent_id_to_token_indices = word_to_token_indices[word]
    for j, sent_id in enumerate(sent_id_to_token_indices):
      ids = sent_id_to_token_ids[sent_id]
      for token_indices in sent_id_to_token_indices[sent_id]:
        start_index = token_indices[0]
        end_index = token_indices[1]
        str_tokens = get_str_tokens(tokenizer, ids)
        word_sequence = str_tokens[start_index: end_index]
        reconstructed_word = ''.join(word_sequence)
        if reconstructed_word != word:
          print('mismatch!')
          exit(0)
        # Now we know the correct indices for this instance of the word.
        # So we should take the mean of the embeddings.
        sent_file = os.path.join(embedding_dir, str(sent_id) + '.sent')
        sum_embedding = np.zeros(768)
        with open(sent_file, 'r') as f:
          for k, embedding in enumerate(f):
            if k == end_index:
              break
            if k >= start_index:
              embedding = embedding.split(', ')
              embedding = [float(e.strip()) for e in embedding]
              sum_embedding += embedding
          num_embeddings = end_index - start_index
          mean_embedding = sum_embedding / num_embeddings
          word_to_embedding[word] = mean_embedding

  word_to_embedding_file = os.path.join(data_dir, 'word_to_embedding_100k.pkl')
  with open(word_to_embedding_file, 'wb') as f:
    pickle.dump(word_to_embedding, f)


if __name__ == '__main__':
  app.run(main)
