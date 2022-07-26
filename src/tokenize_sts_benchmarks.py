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
"""Tokenize STS benchmarks.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

import pickle
import csv
import os

from nltk.parse.corenlp import CoreNLPParser


FLAGS = flags.FLAGS

flags.DEFINE_string('benchmark', '', 'Benchmark(s) for which to generate tokenized pickles.')

flags.DEFINE_string('output_dir', '', 'Directory in which to save tokenized pickles.')


BENCHMARKS = {
    'semeval_2012': {
        'description': 'SemEval STS 2012',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_2012/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
    'semeval_2013': {
        'description': 'SemEval STS 2013',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_2013/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
    'semeval_2014': {
        'description': 'SemEval STS 2014',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_2014/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
    'semeval_2015': {
        'description': 'SemEval STS 2015',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_2015/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
    'semeval_2016': {
        'description': 'SemEval STS 2016',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_2016/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
    'semeval_all': {
        'description': 'SemEval STS - All',
        'file': '/usr/local/google/home/agoldie/data/sts/semeval_all/all.tsv',
        'delimiter': '\t',
        'score_col': 1,
        'has_header': False
    },
}

FLAGS = flags.FLAGS

def read_in_benchmark(name):
  benchmark = BENCHMARKS[name]
  if 'score_col' not in benchmark:
    print('WARNING: score_col not set, defaulting to third column!')
    benchmark['score_col'] = 3
  return (benchmark['file'], benchmark['has_header'], benchmark['delimiter'],
          benchmark['score_col'])

def tokenize_sentence(sentence, tokenizer):
  #print(sentence)
  if '\x12' in sentence:
    sentence = ''.join(sentence.split('\x12'))
  tokenized = tokenizer.tokenize(sentence)
  tokenized = list(tokenized)
  #print(tokenized)
  return tokenized

def read_gs_answers(file):
  answers = {}
  with open(file, 'r') as f:
    for i, line in enumerate(f):
      answers[i] = float(line.strip())
  return answers

def tokenize_benchmark(benchmark_name, output_file, tokenizer):
  tokenized_benchmark = []
  # if benchmark_name == 'semeval_2015':
  #   dir = '/usr/local/google/home/agoldie/data/sts/semeval_2015/test_evaluation_task2a'
  #   gs_file_to_answers_map = {}
  #   gs_files = ['STS.gs.answers-forums.txt', 'STS.gs.answers-students.txt',
  #               'STS.gs.belief.txt' 'STS.gs.headlines.txt', 'STS.gs.images.txt']
  #   for gs_file in gs_files:
  #     gs_file = os.path.join(dir, gs_file)
  #     gs_answers = read_gs_answers(benchmark_name)
  #     gs_file_to_answers_map[gs_file] = gs_answers
  benchmark, has_header, delimiter, score_col = read_in_benchmark(benchmark_name)
  with open(benchmark, newline='') as csvfile:
    sim_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
    if has_header:
      next(sim_reader)
    for i, row in enumerate(sim_reader):
      print(i, row)
      if not row:
        continue
      sim, s1, s2 = (row[0], row[1], row[2])
      if not sim:
        continue

      sim = float(sim)
      t1 = tokenize_sentence(s1, tokenizer)
      t2 = tokenize_sentence(s2, tokenizer)
      tokenized_benchmark.append((sim, t1, t2))
  print('writing out', output_file)
  with open(output_file, 'wb') as f:
    pickle.dump(tokenized_benchmark, f)


def main(argv):
  tokenizer = CoreNLPParser()
  benchmarks = [FLAGS.benchmark]
  if FLAGS.benchmark == 'all':
    benchmarks = list(BENCHMARKS.keys())
  elif ',' in FLAGS.benchmark:
    benchmarks = FLAGS.benchmark.split(',')
  for benchmark in benchmarks:
    # ~/data/sts/semeval_2012/tokenized_benchmark.pkl
    benchmark_dir = os.path.join(FLAGS.output_dir, benchmark)
    output_file = os.path.join(benchmark_dir, 'tokenized_benchmark.pkl')
    print('output_file', output_file)

    tokenize_benchmark(benchmark, output_file, tokenizer)


if __name__ == '__main__':
  app.run(main)
