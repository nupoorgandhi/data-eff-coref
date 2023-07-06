from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
from collections import defaultdict
import util
import coref_ops
import conll
import metrics
import optimization
from bert import tokenization
from bert import modeling
from pytorch_to_tf import load_from_pytorch_checkpoint
import sys
import itertools
import debug_util
from tasks.coref import CorefModel
import metrics

class MentionDetectionModel(CorefModel):

    def filter_tvars(self, tvars):
        return tvars

    def get_loss(self, predictions):
        binary_candidate_labels = self.get_candidate_labels(predictions['candidate_starts'],
                                                            predictions['candidate_ends'],
                                                            predictions['gold_starts'],
                                                            predictions['gold_ends'],
                                                            predictions['cluster_ids'],
                                                            md=True)  # [num_candidates]
        loss = self.sigmoid_loss(predictions['candidate_mention_scores'],
                                 binary_candidate_labels)
        return loss
        

    def sigmoid_loss(self, span_scores, span_labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(span_labels), logits=span_scores)
        loss = tf.reduce_sum(loss)
        return loss

    def setup_evaluation(self):
        self.md_evaluator = metrics.MentionDetectionEvaluator(self.config)
        self.evaluators = [self.md_evaluator]
        return self.md_evaluator

    def evaluate(self, predictions, example):
        clusters = example['clusters']
        self.md_evaluator.update(predictions, example)

