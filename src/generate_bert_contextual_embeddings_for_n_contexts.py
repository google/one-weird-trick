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
"""Generate contextual BERT word embeddings with HuggingFace transformers library.
"""
# >>> import tensorflow as tf
# >>> from transformers import BertTokenizer, TFBertForPreTraining

# >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# >>> model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
# >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
# >>> outputs = model(input_ids)
# >>> prediction_scores, seq_relationship_scores = outputs[:2]

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
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
  #import pdb; pdb.set_trace()
  tokens = tokenizer.convert_ids_to_tokens(ids)
  tokens = [t.strip('#').strip('Ġ').lower() for t in tokens]
  return tokens
  # str_tokens = []
  # for word_token in tokens:
  #   str_token = tokenizer.convert_tokens_to_string(word_token).strip().lower()
  #   str_tokens.append(str_token)
  # return str_tokens

# Figure out why I am missing any word that has been split into subwords.
# Fix the random seed to make these experiments easier?
def generate_word_to_token_indices(tokenizer, ids, words, sentence):
  word_to_token_indices = collections.defaultdict(list)
  str_tokens = get_str_tokens(tokenizer, ids)
  #print('str_tokens', str_tokens)

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
  # import tensorflow as tf

  from transformers import BertTokenizer, TFBertForPreTraining
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  model = TFBertForPreTraining.from_pretrained('bert-large-uncased', output_hidden_states=True)

  # from transformers import GPT2Tokenizer, TFGPT2Model
  # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
  # model = TFGPT2Model.from_pretrained('gpt2-medium')

  vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

  NUM_LAYERS = 25

  index_to_sentence_file = '/tmp/index_to_sentence.pkl'
  index_to_words_file = '/tmp/index_to_words.pkl'
  if os.path.exists(index_to_sentence_file) and os.path.exists(index_to_words_file):
    with open(index_to_sentence_file, 'rb') as f:
      index_to_sentence = pickle.load(f)
    with open(index_to_words_file, 'rb') as f:
      index_to_words = pickle.load(f)
  else:
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
    # this maps a word to the indices corresponding to all sentences in which it appears
    # is this useful??
    word_to_indices = collections.defaultdict(list)

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
          # this is never used?? do I need this?
          word_to_indices[word].append(index)

      if index_to_words[index]:
        index_to_sentence[index] = sentence
      step = len(index_to_sentence)
      if step % 1000 == 0:
        print(step, '/', N)
    assert(len(index_to_sentence) == N)

    with open(index_to_words_file, 'wb') as f:
      pickle.dump(index_to_words, f)
    with open(index_to_sentence_file, 'wb') as f:
      pickle.dump(index_to_sentence, f)

  # why do I need / want a tokenization file? I should just do it all right here...
  # tokenization_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/tokenizations.txt'
  # t = open(tokenization_file, 'w')

  word_dir = '/usr/local/google/home/agoldie/data/bert-24/words/'
  #word_dir = '/usr/local/google/home/agoldie/data/bert/words/'

  # for each sentence that contains at least one vocabulary item, append (mean-pooled) embeddings to the corresponding word files.
  for index in index_to_sentence:
    if index % 100 == 0:
      print('writing', index, '/', len(index_to_sentence), 'sentences')
    sentence = index_to_sentence[index]
    #print('index, sentence', index, sentence)
    words = index_to_words[index]

    # Generate embeddings for these tokens.
    encoded_input = tokenizer(sentence, return_tensors='tf')
    ids = list(encoded_input['input_ids'].numpy().squeeze())

    #import pdb; pdb.set_trace()

    output_attentions = False
    output = model(encoded_input, output_hidden_states=True) #, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

    #print(output.keys())

    # Make sure that taking the first element makes sense here. I think it does.
    hidden_states = output['hidden_states']
    embeddings_by_layer = [list(h[0].numpy()) for h in hidden_states]
    #embeddings = [list(e.numpy()) for e in last_hidden_state]
    #embeddings = embeddings_by_layer[-1]

    # Generate a mapping that contains all of the token indices for each word that is present in the tokenized sentence.
    word_to_token_indices, str_tokens = generate_word_to_token_indices(tokenizer, ids, words, sentence)
    assert(len(words) == len(word_to_token_indices))

    # Then, for each word, perform subword pooling and append an embedding to its embedding file.
    #print(word_to_token_indices)
    for word in word_to_token_indices:
      #print('word', word)
      # Store that embedding in the file of that word.
      # First get that file and open a handle to it.
      # Then append a single line to it.
      for layer in range(13, NUM_LAYERS):
        word_file = os.path.join(word_dir, word + '-' + str(layer) + '.txt')
        with open(word_file, 'a') as w:
          #print('word_file', word_file)
          for token_indices in word_to_token_indices[word]:
            start_index, end_index = token_indices
            subtokens = str_tokens[start_index:end_index]
            assert(''.join(subtokens) == word)
            subword_embeddings = embeddings_by_layer[layer][start_index:end_index]
            mean_embedding = list(np.sum(subword_embeddings, axis=0) / len(subword_embeddings))
            #print('mean_embedding', type(mean_embedding))
            embedding_str = str(mean_embedding).strip(']').strip('[')
            w.write(embedding_str + '\n')

if __name__ == '__main__':
  app.run(main)
