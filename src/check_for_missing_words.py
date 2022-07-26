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

from nltk.tokenize import sent_tokenize

FLAGS = flags.FLAGS


def main(argv):

  oov = []
  with open('/tmp/oov.txt', 'r') as f:
    for word in f:
      oov.append(word.strip())

  input_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.txt'
  print('iterating through sentences')
  with open(input_file, 'r') as f:
    for i, sentence in enumerate(f):
      sentence = sentence.strip()
      #print(sentence)
      if i == 100:
        exit(0)
      for word in oov:
        if word in sentence:
          word_start_index = sentence.index(word)
          prev_char_index = word_start_index - 1
          next_char_index = word_start_index + len(word)
          if next_char_index >= len(sentence) and prev_char_index < 0:
            print(word, ':', sentence)
          else:
            next_char = sentence[next_char_index]
            if next_char.isalpha():
              continue
            print(word, ':', sentence)
            print('[', next_char, ']')



if __name__ == '__main__':
  app.run(main)
