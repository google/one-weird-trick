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
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import os
import pickle
import numpy as np
#from google3.sstable.python import sstable
#from google3.learning.dist_belief.experimental.embedding_client import embedding_pb2


#from nltk.tokenize import word_tokenize
#from transformers import GPT2Tokenizer, TFGPT2Model
#from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
import collections

FLAGS = flags.FLAGS

#     output = model(encoded_input, output_hidden_states=True) #, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

#     #print(output.keys())

#     # Make sure that taking the first element makes sense here. I think it does.
#     hidden_states = output['hidden_states']
#     embeddings_by_layer = [list(h[0].numpy()) for h in hidden_states]

def generate_decontextualized_embedding(model, tokenizer, word):
  encoded_input = tokenizer(word, return_tensors='tf')
  ids = list(encoded_input['input_ids'].numpy())

  output = model(encoded_input, output_hidden_states=True)
  hidden_states = output['hidden_states']
  embeddings_by_layer = [list(h[0].numpy()) for h in hidden_states]
  #embeddings = [list(e.numpy()) for e in last_hidden_state]
  decontextualized_embedding_by_layer = list(np.mean(embeddings_by_layer, axis=1))
  return decontextualized_embedding_by_layer

def main(argv):
  # from transformers import GPT2Tokenizer, TFGPT2Model
  # tokenizer = GPT2Tokenizer.from_pretrained('gpt')
  # model = TFGPT2Model.from_pretrained('gpt')
  from transformers import BertTokenizer, TFBertForPreTraining
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  model = TFBertForPreTraining.from_pretrained('bert-large-uncased', output_hidden_states=True)


  vocab_file = '/usr/local/google/home/agoldie/data/all_benchmarks_vocab.pkl'
  with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

  layer_to_word_to_decontextualized_embedding = collections.defaultdict(dict)
  for i, word in enumerate(vocab):
    if i % 100 == 0:
      print(i, '/', len(vocab))
    decontextualized_embedding_by_layer = generate_decontextualized_embedding(model, tokenizer, word)
    for layer_index, layer in enumerate(decontextualized_embedding_by_layer):
      layer_to_word_to_decontextualized_embedding[layer_index][word] = layer

  # save this as a pickled dictionary and then use another binary to write it to sstable.
  layer_to_word_to_decontextualized_embedding_file = '/usr/local/google/home/agoldie/data/bert-24/layer_to_word_to_decontextualized_embedding.pkl'
  with open(layer_to_word_to_decontextualized_embedding_file, 'wb') as f:
    pickle.dump(layer_to_word_to_decontextualized_embedding, f)

  # embedding_sstable = '~/data/gpt2/decontextualized_embeddings.sstable'
  # with sstable.SortingBuilder(embedding_sstable) as builder:
  #   for word in vocab:
  #     decontextualized_embedding = generate_decontextualized_embedding(model, word)
  #     token_embedding = embedding_pb2.TokenEmbedding()
  #     token_embedding.token = word.encode('utf-8')
  #     token_embedding.vector.values.extend(decontextualized_embedding)
  #     builder.Add(token_embedding.token,
  #                 token_embedding.SerializeToString())


if __name__ == '__main__':
  app.run(main)
