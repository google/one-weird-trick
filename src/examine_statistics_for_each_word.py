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
from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import os
import pickle
import numpy as np
import collections

FLAGS = flags.FLAGS


def main(argv):

  word_dir = '/usr/local/google/home/agoldie/data/gpt2/words/'
  word_to_stats = collections.defaultdict(list)
  overall_min = 133433555
  overall_max = -325342523
  
  for i, word_file in enumerate(os.listdir(word_dir)):
    word_path = os.path.join(word_dir, word_file)
    embeddings = []
    with open(word_path, 'r') as f:
      for e in f:
        embedding = e.strip()
        embedding = list(map(float, embedding.split(', ')))
        embeddings.append(embedding)
    max_embedding = np.max(embeddings)
    min_embedding = np.min(embeddings)
    std_embedding = np.std(embeddings)
    if min_embedding < overall_min:
      overall_min = min_embedding
    if max_embedding > overall_max:
      overall_max = max_embedding
    print('min', min_embedding)
    print('max', max_embedding)

  print('overall_min', overall_min)
  print('overall_max', overall_max)


if __name__ == '__main__':
  app.run(main)
