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
from debug.span_analysis import *
import debug_util
import span_util
import antecedent_linking_analysis
from emb_master import INCOMPAT_PAIRS, COMPAT_PAIRS
from collections import Counter
from scipy import stats
import analysis

CONCEPTS = ['Person', 'Treatment', 'Test', 'Problem']

class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.max_span_width = config["max_span_width"]
        self.max_segment_len = config['max_segment_len']
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=config['vocab_file'], do_lower_case=False)
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.input_props = self.get_input_props()
        self.model_id = config['model_id']
        self.conf_location = config['conf_location']


    def filter_tvars(self, tvars):
        if self.config['cl_freeze_md']:
            tvars = util.drop_md_variables(tvars)
        if self.config['cl_freeze_bert']:
            tvars = util.drop_bert_variables(tvars)
        return tvars

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training):
        pprefix = '[tensorize_example]'
        clusters = example["clusters"]

        if self.config['filter_singletons']:
            clusters = [cl for cl in clusters if len(cl) > 1]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        speaker_dict = self.get_speaker_dict(util.flatten(speakers))
        sentence_map = example['sentence_map']  

        max_sentence_length = self.max_segment_len
        text_len = np.array([len(s) for s in sentences])  

        input_ids, input_mask, speaker_ids = [], [], []
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)  
        input_mask = np.array(input_mask)  
        speaker_ids = np.array(speaker_ids)  

        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        self.gold[doc_key] = example["clusters"]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (
        input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids,
        sentence_map)

        limit_mentions = 'max_sen' in self.config
        if limit_mentions:
            max_sen, last_doc_key = self.config['max_sen'] , self.config['last_doc_key']

            if is_training and doc_key == last_doc_key and len(sentences) > max_sen:
                if self.config['single_example']:
                    return self.truncate_example(*(example_tensors + (max_sen, max_sen,)))
                else:
                    offsets = range(max_sen, len(sentences),
                                    max_sen)
                    tensor_list = [self.truncate_example_custom(*(example_tensors + (max_sen, offset,))) for offset in
                                   offsets]
                    return tensor_list

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*(example_tensors+(self.config["max_training_sentences"],)))
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (self.config["max_training_sentences"], offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors




    def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts, gold_ends,
                         cluster_ids, sentence_map, max_training_sentences, sentence_offset=None):
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0,
                                         num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts, gold_ends, cluster_ids, sentence_map

    def get_input_props(self):
        input_props = []
        input_props.append((tf.int32, [None, None]))  # input_ids.
        input_props.append((tf.int32, [None, None]))  # input_mask
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None, None]))  # Speaker IDs.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.
        input_props.append((tf.int32, [None]))  # Sentence Map
        return input_props

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels, md=False):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        if md:
            candidate_labels = tf.reduce_any(same_span, axis=0)  # [num_candidates]
        else:
            candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
            candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]

        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                             0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]
        if self.config['use_prior']:
            antecedent_distance_buckets = self.bucket_distance(antecedent_offsets)  # [k, c]
            with tf.variable_scope('antecedent_distance_emb', reuse=tf.AUTO_REUSE):
                distance_scores = util.projection(tf.nn.dropout(
                    tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1,
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))  # [10, 1]
            antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1),
                                                   antecedent_distance_buckets)  # [k, c]
            fast_antecedent_scores += antecedent_distance_scores

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def package_predictions(self, mention_doc, candidate_starts, candidate_ends, candidate_mention_scores, candidate_labels,
                top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, top_antecedent_labels,
                cluster_ids, gold_starts, gold_ends):
        return dict(zip('mention_doc, candidate_starts, candidate_ends, candidate_mention_scores, candidate_labels,\
                top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, top_antecedent_labels,\
                cluster_ids, gold_starts, gold_ends'.replace(" ", "").split(','), [mention_doc, candidate_starts, candidate_ends, candidate_mention_scores, candidate_labels,
                top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, top_antecedent_labels,
                cluster_ids, gold_starts, gold_ends]))

    def get_predictions(self, input_ids, input_mask, text_len, speaker_ids, is_training, gold_starts,
                                 gold_ends, cluster_ids, sentence_map):
        pprefix = '[get_predictions_and_loss]'

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        all_encoder_layers = model.get_all_encoder_layers()
        mention_doc = model.get_sequence_output()  # [batch_size, seq_length, hidden_size]

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        num_sentences = tf.shape(mention_doc)[0]
        max_sentence_length = tf.shape(mention_doc)[1]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
        num_words = util.shape(mention_doc, 0)
        antecedent_doc = mention_doc

        flattened_sentence_indices = sentence_map
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
                                                     flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]

        candidate_span_emb, candidate_head_attn_reps = self.get_span_emb(mention_doc, mention_doc, candidate_starts,
                                                                         candidate_ends)  # [num_candidates, emb]

        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
        if not self.config['filter_singletons']:
            k = tf.minimum(k, util.shape(candidate_mention_scores, 0))
        c = tf.minimum(self.config["max_top_antecedents"], k)
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   num_words,
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]

        if self.config['mention_selection_method'] == 'high_f1':
            sigmoid_span_mention_scores = tf.nn.sigmoid(top_span_mention_scores)
            threshold_mask = tf.greater_equal(sigmoid_span_mention_scores, self.config['mention_selection_threshold'])
            top_span_starts = tf.boolean_mask(top_span_starts, threshold_mask)
            top_span_ends = tf.boolean_mask(top_span_ends, threshold_mask)
            top_span_emb = tf.boolean_mask(top_span_emb, threshold_mask)
            top_span_cluster_ids = tf.boolean_mask(top_span_cluster_ids, threshold_mask)
            top_span_mention_scores = tf.boolean_mask(top_span_mention_scores, threshold_mask)
            k = util.shape(top_span_mention_scores, 0)
            c = tf.minimum(self.config["max_top_antecedents"], k)

        if self.config['use_metadata']:
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
            top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]i
        else:
            top_span_speaker_ids = None

        dummy_scores = tf.fill([k,1], -1.0)
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
            top_span_emb, top_span_mention_scores, c)
        num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
        word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
        flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
        mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1)  # [k, 1]
        antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents))  # [k, c]
        segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0,
                                            self.config['max_training_sentences'] - 1) if self.config[
            'use_segment_distance'] else None  # [k, c]
        if self.config['fine_grained']:
            for i in range(self.config["coref_depth"]):
                with tf.variable_scope("coref_layer", reuse=(i > 0)):
                    top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                    top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                         top_antecedents,
                                                                                                         top_antecedent_emb,
                                                                                                         top_antecedent_offsets,
                                                                                                         top_span_speaker_ids,
                                                                                                         segment_distance)  # [k, c]
                    top_antecedent_weights = tf.nn.softmax(
                        tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                    top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                                   1)  # [k, c + 1, emb]
                    attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                      1)  # [k, emb]
                    with tf.variable_scope("f", reuse=tf.AUTO_REUSE):
                        # f determines for each dimension whether to keep the current span information or to integrate new information from its expected antecedent
                        f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                       util.shape(top_span_emb, -1)))  # [k, emb]
                        top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]
        else:
            top_antecedent_scores = top_fast_antecedent_scores

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        candidate_labels = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts,
                                                     gold_ends, cluster_ids, md=True)  # [num_candidates]

        return self.package_predictions(*[mention_doc, candidate_starts, candidate_ends, candidate_mention_scores, candidate_labels,
                top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, top_antecedent_labels,
                cluster_ids, gold_starts, gold_ends])

    def get_loss(self, predictions):
        pprefix = '[get_loss]'
        loss = self.softmax_loss(predictions['top_antecedent_scores'], predictions['top_antecedent_labels'])  # [k]
        loss = tf.reduce_sum(loss)  # []

        return  loss

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            with tf.variable_scope('span_emb', reuse=tf.AUTO_REUSE):
                span_width_emb = tf.gather(
                    tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs)  # [K, T]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]

        return span_emb, head_attn_reps  # [k, emb]

    def get_mention_scores(self, span_emb, span_starts, span_ends):
        with tf.variable_scope("mention_scores", reuse=tf.AUTO_REUSE):
            span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                    self.dropout)  # [k, 1]
        if self.config['use_prior']:
            with tf.variable_scope('span_width_prior_embeddings', reuse=tf.AUTO_REUSE):
                span_width_emb = tf.get_variable("span_width_prior_embeddings",
                                                 [self.config["max_span_width"], self.config["feature_size"]],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))  # [W, emb]
            span_width_index = span_ends - span_starts  # [NC]
            with tf.variable_scope("width_scores", reuse=tf.AUTO_REUSE):
                width_scores = util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                         self.dropout)  # [W, 1]
            width_scores = tf.gather(width_scores, span_width_index)
            span_scores += width_scores
        return span_scores

    def get_width_scores(self, doc, starts, ends):
        distance = ends - starts
        span_start_emb = tf.gather(doc, starts)
        hidden = util.shape(doc, 1)
        with tf.variable_scope('span_width'):
            span_width_emb = tf.gather(
                tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), distance)  # [W, emb]
        scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
        return scores

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = util.shape(encoded_doc, 0)  # T
        num_c = util.shape(span_starts, 0)  # NC
        doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1])  # [K, T]
        mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1),
                                      doc_range <= tf.expand_dims(span_ends, 1))  # [K, T]
        with tf.variable_scope("mention_word_attn", reuse=tf.AUTO_REUSE):
            word_attn = tf.squeeze(
                util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
        mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
        return mention_word_attn

    def sigmoid_loss(self, span_scores, span_labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(span_labels), logits=span_scores)
        loss = tf.reduce_sum(loss)
        return loss

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        pprefix = '[softmax_loss]'
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, segment_distance=None):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]],
                                                         initializer=tf.truncated_normal_initializer(stddev=0.02)),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)



        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]

            with tf.variable_scope('antecedent_distance_emb', reuse=tf.AUTO_REUSE):
                antecedent_distance_emb = tf.gather(
                    tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02)),
                    antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)
        if segment_distance is not None:
            with tf.variable_scope('segment_distance', reuse=tf.AUTO_REUSE):
                segment_distance_emb = tf.gather(tf.get_variable("segment_distance_embeddings",
                                                                 [self.config['max_training_sentences'],
                                                                  self.config["feature_size"]],
                                                                 initializer=tf.truncated_normal_initializer(
                                                                     stddev=0.02)), segment_distance)  # [k, emb]
            feature_emb_list.append(segment_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores", reuse=tf.AUTO_REUSE):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection", reuse=tf.AUTO_REUSE):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents
     
    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents, filter_singletons):
        mention_to_predicted = {}
        predicted_clusters = []
        failed_to_cluster = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                failed_to_cluster.append((int(top_span_starts[i]), int(top_span_ends[i])))
                continue
            assert i > predicted_index, (i, predicted_index)
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                # antecedent was already clustered
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                # create a new cluster for antecedent
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            # add mention to cluster
            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster
        if not filter_singletons:
            for i, predicted_index in enumerate(predicted_antecedents):
                mention = (int(top_span_starts[i]), int(top_span_ends[i]))
                ext_mentions = list(mention_to_predicted.keys())
                if not debug_util.search_overlapping(ext_mentions, mention):
                # if mention not in mention_to_predicted:
                    predicted_cluster = len(predicted_clusters)
                    predicted_clusters.append([mention])
                    mention_to_predicted[mention] = predicted_cluster
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted, failed_to_cluster

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents,
                       gold_clusters, doc_key, example
                       ):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted, failed_to_cluster = self.get_predicted_clusters(top_span_starts,
                                                                                                  top_span_ends,
                                                                                                  predicted_antecedents,
                                                                                                  filter_singletons=False)
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        self.coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold, example)
        self.doc_level_coref_evaluator[doc_key].update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold, example)
        for concept in CONCEPTS:
            if concept not in example['concept_clusters']:
                concept_spans = []
            else:
                concept_spans = [tuple(x) for x in example['concept_clusters'][concept]]
            self.concept_coref_evaluators[concept].update(predicted_clusters, 
                                                          self.filter_concept_clusters(gold_clusters, concept_spans),
                                                          mention_to_predicted, mention_to_gold, example)



        ns_predicted_clusters, ns_mention_to_predicted, ns_failed_to_cluster = self.get_predicted_clusters(top_span_starts,
                                                                                                  top_span_ends,
                                                                                                  predicted_antecedents,
                                                                                                  filter_singletons=True)
        ns_gold_clusters = [cl for cl in gold_clusters if len(cl) > 1]
        ns_mention_to_gold = {}
        for gc in ns_gold_clusters:
            for mention in gc:
                ns_mention_to_gold[mention] = gc
        self.ns_coref_evaluator.update(ns_predicted_clusters, ns_gold_clusters, ns_mention_to_predicted, ns_mention_to_gold, example)
        self.md_evaluator.update(list(zip(top_span_starts, top_span_ends)), gold_clusters)

        return predicted_clusters, failed_to_cluster

    def setup_evaluation(self):
        self.coref_evaluator = metrics.CorefEvaluator(self.config)
        self.md_evaluator = metrics.MentionDetectionEvaluator(self.config)
        self.ns_coref_evaluator = metrics.NonSingletonCorefEvaluator(self.config)
        self.evaluators = [self.coref_evaluator, self.md_evaluator, self.ns_coref_evaluator]
        self.doc_level_coref_evaluator = {}
        self.concept_coref_evaluators = {
            'Person':metrics.CorefEvaluator(self.config, 'person'),
            'Test':metrics.CorefEvaluator(self.config, 'test'),
            'Treatment':metrics.CorefEvaluator(self.config, 'treatment'),
            'Problem':metrics.CorefEvaluator(self.config, 'problem')
        }

        return self.ns_coref_evaluator



    def filter_concept_clusters(self, clusters, concept_spans):
        concept_spans_ = concept_spans
        clusters_ = []
        for cl in clusters:
            for sp in cl:
                if sp in concept_spans_:
                    clusters_.append(cl)
                    break
        return clusters_


    def evaluate(self, predictions, example):
        clusters = example['clusters']
        self.doc_level_coref_evaluator[example['doc_key']] = metrics.CorefEvaluator(self.config)
        
        predicted_antecedents = self.get_predicted_antecedents(predictions['top_antecedents'],
                                                               predictions['top_antecedent_scores'])

        predicted_clusters, failed_to_cluster = self.evaluate_coref(predictions['top_span_starts'],
                                                                   predictions['top_span_ends'],
                                                                   predicted_antecedents,
                                                                   clusters,
                                                                    example['doc_key'],
                                                                    example)

        self.coref_evaluator.subtoken_maps[example['doc_key']] = example['subtoken_map']
        self.coref_evaluator.coref_predictions[example['doc_key']] = predicted_clusters
        self.coref_evaluator.gold_clusters[example['doc_key']] = clusters
        self.doc_level_coref_evaluator[example['doc_key']].subtoken_maps[example['doc_key']] = example['subtoken_map']
        self.doc_level_coref_evaluator[example['doc_key']].coref_predictions[example['doc_key']] = predicted_clusters
        self.doc_level_coref_evaluator[example['doc_key']].gold_clusters[example['doc_key']] = clusters

        ns_clusters = [cl for cl in clusters if len(cl) > 1]
        ns_predicted_clusters = [cl for cl in predicted_clusters if len(cl) > 1]
        self.ns_coref_evaluator.subtoken_maps[example['doc_key']] = example['subtoken_map']
        self.ns_coref_evaluator.coref_predictions[example['doc_key']] = ns_predicted_clusters
        self.ns_coref_evaluator.gold_clusters[example['doc_key']] = ns_clusters

