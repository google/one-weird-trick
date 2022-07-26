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
"""Generate contextual word embeddings with HuggingFace transformers library.
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
import collections

from nltk.tokenize import sent_tokenize
import numpy as np
import random
import string
import pickle

FLAGS = flags.FLAGS

def get_str_tokens(tokenizer, ids):
  #print('IDS', ids)
  tokens = tokenizer.convert_ids_to_tokens(ids)
  token_string = tokenizer.convert_tokens_to_string(tokens)
  #print(token_string)
  str_tokens = []
  for word_token in tokens:
    str_token = tokenizer.convert_tokens_to_string(word_token).strip()
    str_tokens.append(str_token.lower())
  return str_tokens

# Figure out why I am missing any word that has been split into subwords.
# Fix the random seed to make these experiments easier?
def generate_word_to_token_indices(tokenizer, ids, words, sentence):
  word_to_token_indices = collections.defaultdict(list)
  str_tokens = get_str_tokens(tokenizer, ids)

  for word in words:
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
        word_to_token_indices[word].append((start_index, end_index))
    if not word_to_token_indices[word]:
      print('missing word:', word)
      print('str_tokens', str_tokens)
      print('sentence', sentence)
      exit(0)
  return word_to_token_indices, str_tokens

def main(argv):
  from transformers import GPT2Tokenizer, TFGPT2Model
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = TFGPT2Model.from_pretrained('gpt2')

  # One source of discrepancy could be using a different vocabulary (theirs is 2k, mine is a different 5k).
  vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

  input_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.txt'
  sentences = []
  with open(input_file, 'r') as f:
    for i, sentence in enumerate(f):
      sentences.append(sentence)

  random.Random(7).shuffle(sentences)

  # this maps an index to a sentence that contains at least one vocabulary instance, could really just be a list
  index_to_sentence = {}
  # this maps the same index to a list of words that the corresponding sentence contains
  index_to_words = collections.defaultdict(list)

  N = 100000
  # Extract N contexts.
  print('0', '/', N)
  while len(index_to_sentence) < N:
    index = len(index_to_sentence)
    sentence = sentences.pop()
    punct = string.punctuation
    # maybe we should apply the same tokenization here? and find the tokens here too? why recompute it?
    # but it is quite expensive so maybe this makes sense as a quick first pass?
    # am I handling lowercase correctly in the other location?
    words = [w.strip(punct) for w in sentence.lower().split(' ')]
    for word in vocab:
      if word in words:
        index_to_words[index].append(word)

    if index_to_words[index]:
      index_to_sentence[index] = sentence

    step = len(index_to_sentence)
    if step % 1000 == 0:
      print(step, '/', N)
  assert(len(index_to_sentence) == N)

  word_dir = '/usr/local/google/home/agoldie/data/gpt2/words/'

  # for each sentence that contains at least one vocabulary item, append (mean-pooled) embeddings to the corresponding word files.
  for index in index_to_sentence:
    sentence = index_to_sentence[index]
    words = index_to_words[index]

    # Generate embeddings for these tokens.
    encoded_input = tokenizer(sentence, return_tensors='tf')
    ids = list(encoded_input['input_ids'].numpy().squeeze())

    output_hidden_states = False
    output_attentions = False
    output = model(encoded_input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

    #print(output.keys())

    # Make sure that taking the first element makes sense here. I think it does.
    last_hidden_state = output['last_hidden_state'][0]
    embeddings = [list(e.numpy()) for e in last_hidden_state]
    import pdb; pdb.set_trace()

    # Generate a mapping that contains all of the token indices for each word that is present in the tokenized sentence.
    word_to_token_indices, str_tokens = generate_word_to_token_indices(tokenizer, ids, words, sentence)
    assert(len(words) == len(word_to_token_indices))

    # Then, for each word, perform subword pooling and append an embedding to its embedding file.
    for word in word_to_token_indices:
      #print('word', word)
      # Store that embedding in the file of that word.
      # First get that file and open a handle to it.
      # Then append a single line to it.
      word_file = os.path.join(word_dir, word + '.txt')
      with open(word_file, 'a') as w:
        #print('word_file', word_file)
        for token_indices in word_to_token_indices[word]:
          start_index, end_index = token_indices
          subtokens = str_tokens[start_index:end_index]
          assert(''.join(subtokens) == word)
          subword_embeddings = embeddings[start_index:end_index]
          mean_embedding = list(np.sum(subword_embeddings, axis=0) / len(subword_embeddings))
          #print('mean_embedding', type(mean_embedding))
          embedding_str = str(mean_embedding).strip(']').strip('[')
          w.write(embedding_str + '\n')

if __name__ == '__main__':
  app.run(main)
