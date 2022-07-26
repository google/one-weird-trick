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
  from transformers import GPT2Tokenizer, TFGPT2Model
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = TFGPT2Model.from_pretrained('gpt2')
  # text = "I am a cat. Who are you? Jealousy does not become a proletariat. proletariat"
  # sentences = sent_tokenize(text)
  # print(sentences)
  embedding_dir = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/embeddings/'
  tokenization_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/tokenizations.txt'
  t = open(tokenization_file, 'w')

  input_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.txt'
  with open(input_file, 'r') as f:
    for i, sentence in enumerate(f):
      if i == 1:
        exit(0)
      sentence_file = os.path.join(embedding_dir, str(i) + '.sent')
      with open(sentence_file, 'w') as s:

        tokens = tokenizer(sentence)
        tokenization = tokens['input_ids']
        # t.write(str(tokenization).strip('[').strip(']'))
        # t.write('\n')

        encoded_input = tokenizer(sentence, return_tensors='tf')

        output_hidden_states = False
        output_attentions = False
        output = model(encoded_input, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        print(output.keys())

        embeddings = output['last_hidden_state'][0]
        for embedding in embeddings:
          embedding = list(embedding.numpy())
          # s.write(str(embedding).strip('[').strip(']'))
          # s.write('\n')
  t.close()


if __name__ == '__main__':
  app.run(main)
