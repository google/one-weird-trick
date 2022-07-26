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
"""Sort vocab by frequency.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import urllib
import requests
import pickle

FLAGS = flags.FLAGS

def main(argv):
  frequency_word_tuples = []
  vocab_pkl = '/usr/local/google/home/agoldie/data/glove/full_glove_vocab.pkl'
  with open(vocab_pkl, 'rb') as f:
    vocab = pickle.load(f)

  # Load the full vocabulary for GloVe from the SSTable and get the keys to populate words.
  for i, word in enumerate(vocab):
    if i % 100 == 0:
      print(i, '/', len(vocab))
    word = word.decode()
    match_count = 0
    encoded_query = urllib.parse.quote(word)
    params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 1, 'format': 'tsv'}
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items())
    response = requests.get('https://api.phrasefinder.io/search?' + params)
    #print(response.text)
    if not response.text:
      continue
    # seems like match count is what matters most.
    tks, mc, vc, fy, ly, id, fc = response.text.split('\t')
    tt, tg = tks.split('_')
    mc = int(mc)

    if word.casefold() != tt.casefold():
      continue

    tuple = (match_count, word)
    frequency_word_tuples.append(tuple)

  #print(frequency_word_tuples)
  frequency_word_tuples.sort()
  #print(frequency_word_tuples)
  print(len(frequency_word_tuples))
  frequency_word_tuples_file = '/usr/local/google/home/agoldie/data/sorted_frequency_word_tuples.pkl'
  with open(frequency_word_tuples_file, 'wb') as f:
    pickle.dump(frequency_word_tuples, f)


if __name__ == '__main__':
  app.run(main)
