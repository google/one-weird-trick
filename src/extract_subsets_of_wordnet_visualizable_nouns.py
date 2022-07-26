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

"""Performs word similarity evaluations over a set of embeddings."""
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import csv
import pickle
import nltk
from nltk.corpus import words
from random import sample
from nltk.corpus import wordnet as wn

#flags.DEFINE_string('nltk_vocab', '', 'Pkl file containing vocabulary')

def get_hyponyms(ss):
  location = wn.synset('location.n.01')
  if ss == location:
    return set()
  hyponyms = set()
  for hyponym in ss.hyponyms():
    hyponyms |= set(get_hyponyms(hyponym))
  return hyponyms | set(ss.hyponyms())

def main(_):
  ss = wn.synset('physical_entity.n.01')
  hyponyms = get_hyponyms(ss)
  # print('hyponyms', len(hyponyms))
  # for hyponym in hyponyms:
  #   if 'location' in hyponym.name():
  #     print(hyponym)
  wordnet_lemmas = set()
  for hyponym in hyponyms:
    wordnet_lemmas.update(hyponym.lemma_names())
  wordnet_lemmas = [l for l in wordnet_lemmas if '_' not in l]
  print(wordnet_lemmas, len(wordnet_lemmas))
  for vocab_size in range(5000, len(wordnet_lemmas) + 5000, 5000):
    vocab_size = min(vocab_size, 28086)
    sample_vocab = sample(wordnet_lemmas, vocab_size)
    output_vocab_file = '/usr/local/google/home/agoldie/data/wordnet-' + str(vocab_size) + '.pkl'
    with open(output_vocab_file, 'wb') as f:
      pickle.dump(sample_vocab, f)

if __name__ == '__main__':
  app.run(main)
