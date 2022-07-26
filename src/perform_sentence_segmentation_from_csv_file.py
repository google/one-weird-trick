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
"""Perform sentence segmentation from CSV file.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pickle

# FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     'input_file', '',
#     'CSV file containing raw input data.'
# )



def main(argv):
  input_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.csv'
  output_file = '/usr/local/google/home/agoldie/data/filtered_english_wikipedia/wikipedia_utf8_filtered_20pageviews.txt'
  sentences = []
  with open(input_file, 'r') as f:
    for i, line in enumerate(f):
      partition_index = line.index(',')
      page_content = line[partition_index + 1:]
      page_content = page_content.strip().strip('\"')
      new_sentences = sent_tokenize(page_content)
      sentences.extend(new_sentences)
  num_filtered = 0
  filtered_sentences = []
  unfiltered_length = len(sentences)
  for sentence in sentences:
    words = word_tokenize(sentence)
    if len(words) >= 7 and len(words) <= 75:
      filtered_sentences.append(sentence)
  filtered_length = len(filtered_sentences)

  with open(output_file, 'w') as f:
    for sentence in filtered_sentences:
      f.write(sentence)
      f.write('\n')

  print('wrote', filtered_length, 'sentences to', output_file)


if __name__ == '__main__':
  app.run(main)
