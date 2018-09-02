# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for calculating loss, accuracy, and other model metrics.

Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import six
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin


def _convert_to_eval_metric(metric_fn):
    """Wrap a metric fn that returns scores and weights as an eval metric fn.

    The input metric_fn returns values for the current batch. The wrapper
    aggregates the return values collected over all of the batches evaluated.

    Args:
      metric_fn: function that returns scores and weights for the current batch's
        logits and predicted labels.

    Returns:
      function that aggregates the scores and weights from metric_fn.
    """

    def problem_metric_fn(*args):
        """Returns an aggregation of the metric_fn's returned values."""
        scores = metric_fn(*args)

        # The tf.metrics.mean function assures correct aggregation.
        return tf.metrics.mean(scores)

    return problem_metric_fn


def get_eval_metrics(logits, labels, params):
    """Return dictionary of model evaluation metrics."""
    metrics = {
        "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
        "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
            logits, labels),
        "accuracy_per_sequence": _convert_to_eval_metric(
            padded_sequence_accuracy)(logits, labels)
    }

    # Prefix each of the metric names with "metrics/". This allows the metric
    # graphs to display under the "metrics" category in TensorBoard.
    metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
    return metrics


def padded_accuracy(logits, labels):
    """Percentage of times that predictions matches labels."""
    with tf.variable_scope("padded_accuracy", values=[logits, labels]):
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        padded_labels = tf.to_int32(labels)
        return tf.to_float(tf.equal(outputs, padded_labels))


def padded_accuracy_topk(logits, labels, k):
    """Percentage of times that top-k predictions matches labels on."""
    with tf.variable_scope("padded_accuracy_topk", values=[logits, labels]):
        effective_k = tf.minimum(k, tf.shape(logits)[-1])
        _, outputs = tf.nn.top_k(logits, k=effective_k)
        outputs = tf.to_int32(outputs)
        padded_labels = tf.to_int32(labels)
        padded_labels = tf.expand_dims(padded_labels, axis=-1)
        padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
        same = tf.to_float(tf.equal(outputs, padded_labels))
        same_topk = tf.reduce_sum(same, axis=-1)
        return same_topk


def padded_accuracy_top5(logits, labels):
    return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
    """Percentage of times that predictions matches labels everywhere."""
    with tf.variable_scope("padded_sequence_accuracy", values=[logits, labels]):
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        padded_labels = tf.to_int32(labels)
        not_correct = tf.to_float(tf.not_equal(outputs, padded_labels))
        axis = list(range(1, len(outputs.get_shape())))
        correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
        return correct_seq
