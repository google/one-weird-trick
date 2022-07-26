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

# BUILD dependencies.
package(
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = "aggregate_contextual_embeddings",
    srcs = ["aggregate_contextual_embeddings.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
        "//third_party/py/transformers",
    ],
)

py_binary(
    name = "word_analogy_evaluation",
    srcs = ["word_analogy_evaluation.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
        "//third_party/py/transformers",
    ],
)

py_binary(
    name = "separate_syntactic_and_semantic",
    srcs = ["separate_syntactic_and_semantic.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
        "//third_party/py/transformers",
    ],
)

py_binary(
    name = "extract_vocab",
    srcs = ["extract_vocab.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "subtract_mean",
    srcs = ["subtract_mean.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "subtract_mean_for_multiple_vocabs",
    srcs = ["subtract_mean_for_multiple_vocabs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "subtract_mean_for_vocab_dir",
    srcs = ["subtract_mean_for_vocab_dir.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "subtract_mean_for_random_subsets",
    srcs = ["subtract_mean_for_random_subsets.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "subtract_mean_for_benchmark_transfer",
    srcs = ["subtract_mean_for_benchmark_transfer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "subtract_mean_for_all_layers",
    srcs = ["subtract_mean_for_all_layers.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy",
    srcs = ["explore_isotropy.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_glove",
    srcs = ["explore_isotropy_glove.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_glove_abtt",
    srcs = ["explore_isotropy_glove_abtt.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_word2vec_abtt",
    srcs = ["explore_isotropy_word2vec_abtt.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_word2vec_100k",
    srcs = ["explore_isotropy_word2vec_100k.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_word2vec_before",
    srcs = ["explore_isotropy_word2vec_before.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_glove_before",
    srcs = ["explore_isotropy_glove_before.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "explore_isotropy_word2vec",
    srcs = ["explore_isotropy_word2vec.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components",
    srcs = ["remove_principal_components.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components_for_vocab_subset",
    srcs = ["remove_principal_components_for_vocab_subset.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/nltk",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components_for_vocab_dir",
    srcs = ["remove_principal_components_for_vocab_dir.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/nltk",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components_for_random_subsets",
    srcs = ["remove_principal_components_for_random_subsets.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/nltk",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components_for_benchmark_transfer",
    srcs = ["remove_principal_components_for_benchmark_transfer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/nltk",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "remove_principal_components_for_all_layers",
    srcs = ["remove_principal_components_for_all_layers.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
    ],
)

py_binary(
    name = "write_subset_of_embeddings",
    srcs = ["write_subset_of_embeddings.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "write_subset_of_embeddings_for_vocab_dir",
    srcs = ["write_subset_of_embeddings_for_vocab_dir.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "aggregate_contextual_embeddings_for_each_word",
    srcs = ["aggregate_contextual_embeddings_for_each_word.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "generate_decontextualized_embeddings",
    srcs = ["generate_decontextualized_embeddings.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "write_contextual_embeddings",
    srcs = ["write_contextual_embeddings.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "write_embeddings_to_sstable",
    srcs = ["write_embeddings_to_sstable.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "write_decontextual_embeddings_to_sstable",
    srcs = ["write_decontextual_embeddings_to_sstable.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "examine_variance_for_glove_embeddings",
    srcs = ["examine_variance_for_glove_embeddings.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "lexical_similarity_benchmarks",
    srcs = ["lexical_similarity_benchmarks.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)

py_binary(
    name = "sts_benchmarks",
    srcs = ["sts_benchmarks.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//learning/dist_belief/experimental/embedding_client:embedding_py_pb2",
        "//pyglib",
        "//sstable/python:pysstable",
        "//third_party/py/matplotlib",
        "//third_party/py/scipy",
    ],
)
