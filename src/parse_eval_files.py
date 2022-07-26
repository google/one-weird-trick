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
"""Parse eval files.
"""


from absl import app
from absl import flags
import collections
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_file', '', 'Evaluation file to parse.')
flags.DEFINE_string('output_file', '', 'Pickle file to save map to.')


def load_tuples(eval_file):
  benchmarks = ['mturk', 'rg65', 'simverb', 'rw', 'men', 'wordsim', 'simlex']

  benchmark = ''
  benchmark_to_tuples = collections.defaultdict(list)
  with open(eval_file, 'r') as f:
    for line in f:
      if not line:
        continue
      line = line.strip()
      sets_benchmark = False
      for b in benchmarks:
        if b in line:
          benchmark = b
          sets_benchmark = True
          print('benchmark', benchmark)
          break
      if sets_benchmark:
        continue
      if not line:
        continue
      if 'removed' not in line:
        continue
      if line.startswith('reported'):
        continue
      if line.startswith('num_skipped'):
        continue
      if len(line.split(' ')) > 2:
        segments = line.split(' ')
        score = segments[0]
        score = float(score)
        table_names = segments[1:]
        for table_name in table_names:
          table_name = file_array.strip('[').strip(']').strip('\'')
          segments = table_name.split('-')
          for segment in segments:
            if 'start' in segment:
              start_index = int(segment[len('start'):])
            elif 'vocab' in segment:
              vocab_size = int(segment[len('vocab'):])
          benchmark_to_tuples[benchmark].append((score, start_index, vocab_size))
      else:
        score, file_array = line.split(' ')
        score = float(score)
        table_name = file_array.strip('[').strip(']').strip('\'')
        segments = table_name.split('-')
        for segment in segments:
          if 'start' in segment:
            start_index = int(segment[len('start'):])
          elif 'vocab' in segment:
            vocab_size = int(segment[len('vocab'):])
        benchmark_to_tuples[benchmark].append((score, start_index, vocab_size))
  return benchmark_to_tuples


def main(argv):
  benchmarks = ['mturk', 'rg65', 'simverb', 'rw', 'men', 'wordsim', 'simlex']


  eval_file = FLAGS.eval_file
  benchmark_to_tuples = load_tuples(eval_file)
  for benchmark in benchmarks:
    print(len(benchmark_to_tuples[benchmark]))

  with open(FLAGS.output_file, 'wb') as f:
    pickle.dump(benchmark_to_tuples, f)
  mturk_tuples = benchmark_to_tuples['mturk']
  print(mturk_tuples)
  sorted_by_size = sorted(mturk_tuples, key=lambda tup: tup[2])
  print(sorted_by_size)


if __name__ == '__main__':
  app.run(main)
