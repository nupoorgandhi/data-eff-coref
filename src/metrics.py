from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import util
import numpy as np
from collections import Counter
import os
from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment
import util
import conll


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class MLMEvaluator(object):
    def __init__(self):
        self.per_example_loss = []
    def update(self, per_example_loss_new):
        self.per_example_loss.append(per_example_loss_new)
    def make_summary(self):
        return {'per_example_loss':np.average(self.per_example_loss)}
class MentionDetectionEvaluator(object):
    def __init__(self, config):
        self.tp, self.fn, self.fp = 0,0,0
        self.ns_tp, self.ns_fp, self.ns_fn = 0,0,0
        self.config = config
        self.silver_examples = []
        self.doc_level_results = {}

    def update(self, predictions, example):
        if type(predictions) == list:
            predicted_cl = predictions
        else:
            predicted_cl = list(zip(predictions['top_span_starts'], predictions['top_span_ends']))

        if type(example) == dict:
            gold_cl = example['clusters']
        else:
            gold_cl = example
        gold_mentions = set([(m[0], m[1]) for cl in gold_cl for m in cl])

        pred_mentions = set([(s, e) for s, e in predicted_cl])

        self.tp += len(gold_mentions & pred_mentions)
        self.fn += len(gold_mentions - pred_mentions)
        self.fp += len(pred_mentions - gold_mentions)

        ns_gold_mentions = set([(m[0], m[1]) for cl in gold_cl for m in cl if len(cl) > 1])
        self.ns_tp += len(ns_gold_mentions & pred_mentions)
        self.ns_fn += len(ns_gold_mentions - pred_mentions)
        self.ns_fp += len(pred_mentions - ns_gold_mentions)

        if self.config['save_silver_mentions']:
            if not example['given']:
                example['clusters'] = [list(pred_mentions)]
            self.silver_examples.append(example)
        
    def get_precision(self, ns=False):
        if not ns:
            p = 0 if (self.tp + self.fp) == 0 else float(self.tp) / (self.tp + self.fp)
        else:
            p = 0 if (self.ns_tp + self.ns_fp) == 0 else float(self.ns_tp) / (self.ns_tp + self.ns_fp)
        return p

    def get_recall(self, ns=False):
        if not ns:
            r = 0 if (self.tp + self.fn) == 0 else float(self.tp) / (self.tp + self.fn)
        else:
            r = 0 if (self.ns_tp + self.ns_fn) == 0 else float(self.ns_tp) / (self.ns_tp + self.ns_fn)
        return r

    def get_f1(self, ns=False):
        if (self.get_recall(ns) + self.get_precision(ns)) == 0:
            return 0
        f1 = 2.0 * self.get_recall(ns) * self.get_precision(ns) / (self.get_recall(ns) + self.get_precision(ns))
        return f1
    def get_prf(self, ns=False):
        return self.get_precision(ns), self.get_recall(ns), self.get_f1(ns)

    def make_summary(self):
        summary_dict = {}
        p, r, f = self.get_prf(ns=False)
        summary_dict["MentionDetection f1"] = f
        summary_dict["MentionDetection p"] = p
        summary_dict["MentionDetection r"] = r

        p, r, f = self.get_prf(ns=True)
        summary_dict["MentionDetection ns_f1"] = f
        summary_dict["MentionDetection ns_p"] = p
        summary_dict["MentionDetection ns_r"] = r

        return summary_dict

    def save_results(self):
        log_dir = self.config["log_dir"]

        util.save_results(self.make_summary(), 'md', log_dir)
        if 'save_silver_mentions' in self.config and self.config['save_silver_mentions']:
            util.save_md_silver(self.config, self.silver_examples)

    def get_precision_dk(self, doc_key, ns=False):
        if not ns:
            p = 0 if (self.doc_level_results[doc_key]['tp'] +
                      self.doc_level_results[doc_key]['fp']) == 0 \
                else float(self.doc_level_results[doc_key]['tp']) / \
                     (self.doc_level_results[doc_key]['tp'] +
                      self.doc_level_results[doc_key]['fp'])
        else:
            p = 0 if (self.doc_level_results[doc_key]['ns_tp'] +
                      self.doc_level_results[doc_key]['ns_fp']) == 0 \
                else float(self.doc_level_results[doc_key]['ns_tp']) / \
                     (self.doc_level_results[doc_key]['ns_tp'] +
                      self.doc_level_results[doc_key]['ns_fp'])
        return p

    def get_recall_dk(self, doc_key, ns=False):
        if not ns:
            r = 0 if (self.doc_level_results[doc_key]['tp'] + self.doc_level_results[doc_key]['fn'])\
                     == 0 else float(self.doc_level_results[doc_key]['tp']) / (self.doc_level_results[doc_key]['tp'] +
                                                                               self.doc_level_results[doc_key]['fn'])
        else:
            r = 0 if (self.doc_level_results[doc_key]['ns_tp'] + self.doc_level_results[doc_key]['ns_fn']) \
                     == 0 else float(self.doc_level_results[doc_key]['ns_tp']) / (self.doc_level_results[doc_key]['ns_tp'] +
                                                                               self.doc_level_results[doc_key]['ns_fn'])

        return r

    def get_f1_dk(self, doc_key, ns=False):
        f1 = 2.0 * self.get_recall_dk(doc_key, ns) * self.get_precision_dk(doc_key, ns) / (self.get_recall_dk(doc_key, ns)
                                                                                           + self.get_precision_dk(doc_key, ns))
        return f1

class CorefEvaluator(object):
    def __init__(self, config, concept=None):


        self.eval_metrics = 'lea,muc,b_cubed,ceafe'.split(',')
        self.evaluators = [Evaluator(m) for m in (lea,muc, b_cubed, ceafe)]
        self.subtoken_maps = {}
        self.coref_predictions = {}
        self.gold_clusters = {}
        self.config = config
        self.examples = []
        self.concept = concept

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold, example):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)
        example['predictions'] = predicted
        self.examples.append(example)


    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def make_detailed_summary(self):
        summary_dict = {}
        for eval_metric_name, evaluator in zip(self.eval_metrics, self.evaluators):
            summary_dict['{} r'.format(eval_metric_name)] = evaluator.get_recall()
            summary_dict['{} p'.format(eval_metric_name)] = evaluator.get_precision()
            summary_dict['{} f1'.format(eval_metric_name)] = evaluator.get_f1()
        p, r, f = self.get_prf()
        summary_dict['Average r'] = r
        summary_dict['Average p'] = p
        summary_dict['Average f1'] = f
        return summary_dict


    def make_summary(self):
        summary_dict = {}
        p, r, f = self.get_prf()
        summary_dict['Coref Average r'] = r
        summary_dict['Coref Average p'] = p
        summary_dict['Coref Average f1'] = f
        return summary_dict
    def save_results(self):
        log_dir = self.config["log_dir"]

        print('[eval_conll] save_results to {}...'.format(self.config["log_dir"]))
        if self.concept == None:
            util.save_results(self.make_detailed_summary(), 'coref', log_dir)
        else:
            if 'white' in self.config['cl_eval_path']:
                util.save_results(self.make_detailed_summary(), '{}_white_coref'.format(self.concept), log_dir)
            else:
                util.save_results(self.make_detailed_summary(), '{}_black_coref'.format(self.concept), log_dir)

            return
        new_gold_path = os.path.join(log_dir, 'gold.conll')
        conll.write_new_gold_file(self.config["conll_eval_path"], new_gold_path,
                                  self.gold_clusters, self.subtoken_maps)

        conll.write_predictions_file(new_gold_path, self.coref_predictions, self.subtoken_maps,
                                             output_conll_file=os.path.join(log_dir,
                                                                            '{}.conll'.format('predictions')),
                                             )


class NonSingletonCorefEvaluator(CorefEvaluator):
    def save_results(self):
        log_dir = self.config["log_dir"]

        print('[eval_conll] save_results to {}...'.format(self.config["log_dir"]))
        util.save_results(self.make_detailed_summary(), 'ns_coref', log_dir)


        new_gold_path = os.path.join(log_dir, 'ns_gold.conll')
        conll.write_new_gold_file(self.config["conll_eval_path"], new_gold_path,
                                  self.gold_clusters, self.subtoken_maps)
        
        conll.write_predictions_file(new_gold_path, self.coref_predictions, self.subtoken_maps,
                                             output_conll_file=os.path.join(log_dir,
                                                                            '{}.conll'.format('ns_predictions')),
                                             )

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count
        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)

    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
