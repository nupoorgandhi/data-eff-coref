from tasks.coref import CorefModel
from tasks.md import MentionDetectionModel
from tasks.mlm import MaskingModel
from tasks.coref_ns import NonsingletonCorefModel
from tasks.coref_gm import GoldMentionCorefModel
#from tasks.coref_prec import CustomPrecisionCorefModel
from bert import tokenization, modeling, create_pretraining_data
import tensorflow as tf
from pytorch_to_tf import load_from_pytorch_checkpoint
import optimization
import json
import itertools
import threading
import random
import util
import numpy as np
import os
def combine_losses(loss):
    combined_loss = tf.add_n([v for k,v in loss.items()])
    return combined_loss


class DataHandler(object):
    def __init__(self, config):
        self.config = config
        self.task_ids = [x.lower().strip() for x in config['task_ids'].split('_')]
        self.primary_task = config['primary_task']
        self.eval_data = None # load eval data lazily
        self.task_train_examples = {}

    def get_silver_path(self, task_id):
        silver_data_dir = os.path.join(self.config['data_dir'], 'silver_jsonlines')
        if not os.path.exists(silver_data_dir):
            os.makedirs(silver_data_dir)
        train_jsonlines_filename = 'silver.{}.'.format(self.config['model_id']) + os.path.basename(self.config['{}_train_path'.format(task_id)])
        silver_path = os.path.join(silver_data_dir, train_jsonlines_filename)
        return silver_path

    """
    for each task_id, check if there are multiple datasets
        if so, for each dataset, check if you need to limit mentions
    """
    def get_task_train_examples(self, task_id, task):
        if self.config["{}_load_silver".format(task_id)]:
            task_train_path = self.get_silver_path(task_id)
        else:
            task_train_path = self.config["{}_train_path".format(task_id)]
        print('loading...', task_train_path)
        with open(task_train_path) as f:
            task_train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        if self.config["{}_max_mentions".format(task_id)] > 0 and not (self.config['{}_num_datasets'.format(task_id)] == 2: 
            task_max_mentions = self.config["{}_max_mentions".format(task_id)]
            task_train_examples = self.limit_mentions(task_train_examples, task,
                                                      task_max_mentions)
        if self.config['{}_num_datasets'.format(task_id)] == 2 :
            secondary_train_path = self.config["{}_train_path_2".format(task_id)]

            with open(secondary_train_path) as f:
                secondary_task_train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
            if self.config["{}_max_mentions_2".format(task_id)] > 0:
                task_max_mentions = self.config["{}_max_mentions_2".format(task_id)]
                secondary_task_train_examples = self.limit_mentions(secondary_task_train_examples,
                                                                    task,
                                                                    task_max_mentions)
            print('loading {} examples from {}, {} examples from {}'.format(len(task_train_examples),
                                                                            task_train_path,
                                                                            len(secondary_task_train_examples),
                                                                            secondary_train_path))
            task_train_examples = task_train_examples + secondary_task_train_examples
        return task_train_examples


    """
    creates a list of lists [# of examples in max task, # tasks]. Each element is a tuple of task_id, example
    """
    def get_train_examples(self, tasks):
        task_train_examples = {}
        for task_id in self.task_ids:
            task_train_examples[task_id] = self.get_task_train_examples(task_id, tasks[task_id])
        self.task_train_examples = task_train_examples

        # merge together the examples from each domain
        train_set_len = max([len(v) for k,v in task_train_examples.items()])
        task_train_iter = [(task_id, itertools.cycle(task_train_examples[task_id])) for task_id in self.task_ids]
        train_examples = []
        for i in range(train_set_len):
            train_examples.append([(task_id, next(task_train_example)) for task_id, task_train_example in task_train_iter])
        return train_examples

    """ given a list of json objects train examples, we need to limit the number of mentions
        truncates the set of training docs, reports the max sen on the last doc, doc_key of last doc
    """
    def limit_mentions(self, train_examples, task, task_max_mentions):
        last_doc = len(train_examples)
        last_sentence = self.config['max_training_sentences']
        mention_count = 0
        random.shuffle(train_examples)
        last_doc_key = train_examples[-1]['doc_key']
        for doc_idx, example in enumerate(train_examples):
            sentence_idx_map = [[i] * len(sen) for i, sen in enumerate(example["sentences"])]
            sentence_idx_map = util.flatten(sentence_idx_map)
            if self.config['filter_singletons']:
                example_mentions_ = util.flatten([x for x in example['clusters'] if len(x) > 1])
            else:
                example_mentions_ = util.flatten(example['clusters'])

            # filter mentions for last training sen
            example_mentions = []
            for m in example_mentions_:
                if sentence_idx_map[m[0]] > self.config['max_training_sentences']:
                    continue
                example_mentions.append(m)

            if len(example_mentions) + mention_count > task_max_mentions:
                # this is the stopping doc and we need to truncate it
                example_mentions = sorted(example_mentions)
                # trim to only max-mention_count mentions
                last_mention = example_mentions[min(len(example_mentions) - 1,
                                                    task_max_mentions - mention_count)]

                sentence_idx_map = [[i] * len(sen) for i, sen in enumerate(example["sentences"])]
                sentence_idx_map = util.flatten(sentence_idx_map)
                last_sentence = sentence_idx_map[last_mention[0]]
                last_doc = doc_idx
                last_doc_key = example['doc_key']
                print(pprefix, 'last_sen:{}, last_doc:{}, mention_count:{}, max_mentions:{}'.format(last_sentence, last_doc, mention_count, task_max_mentions))
                break
            mention_count += len(example_mentions)
        train_examples = train_examples[:last_doc]
        task.config['max_sen'] = last_sentence
        task.config['last_doc_key'] = last_doc_key
        return train_examples

    """ 
    load a dict where key=task_id and value=list of json examples
    if eval_mode=True we load the eval path, otherwise we load the dev path
    """
    def get_eval_examples(self, task, eval_mode=False):
        if self.eval_data is None:
            self.eval_data = {}

            def load_line(line):
                example = json.loads(line)
                return task.tensorize_example(example, is_training=False), example

            for task_id in self.task_ids:
                if eval_mode:
                    print('eval mode true:', self.config["{}_eval_path".format(task_id)], task_id)
                    with open(self.config["{}_eval_path".format(task_id)]) as f:
                        self.eval_data[task_id] = [load_line(l) for l in f.readlines()]

                else:
                    print('eval mode false:', self.config["{}_dev_path".format(task_id)], task_id)
                    with open(self.config["{}_dev_path".format(task_id)]) as f:
                        self.eval_data[task_id] = [load_line(l) for l in f.readlines()]
               
            print("Loaded {} eval examples.".format(len(self.eval_data[self.primary_task])))


class MultiTaskTrainer(object):
    def __init__(self, config):
        self.config = config
        self.dh = DataHandler(config)
        self.max_span_width = config["max_span_width"]
        self.eval_data = None  # Load eval data lazily.
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.task_ids = [x.lower().strip() for x in config['task_ids'].split('_')]
        self.primary_task = config['primary_task']
        self.queue_input_tensors = {}
        self.enqueue_op = {}
        self.input_tensors = {}
        self.tasks = {}
        self.predictions = {}
        self.loss = {}
        queue = {}
        for task_id in self.task_ids:
            self.tasks[task_id] = self.setup_task(task_id, config)
            dtypes, shapes = zip(*self.tasks[task_id].input_props)

            self.queue_input_tensors[task_id] = [tf.placeholder(dtype, shape) for dtype, shape in self.tasks[task_id].input_props]
            queue[task_id] = tf.PaddingFIFOQueue(capacity=len(self.tasks[task_id].input_props), dtypes=dtypes, shapes=shapes)
            self.enqueue_op[task_id] = queue[task_id].enqueue(
                self.queue_input_tensors[task_id])  # op that enqueues examples one at a time into teh graph

            self.input_tensors[task_id] = queue[task_id].dequeue()

            self.predictions[task_id], self.loss[task_id] = self.get_predictions_and_loss(task_id)
        # bert stuff
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config[
            'tf_checkpoint'])
        print('initializing from ', config['tf_checkpoint'])



        init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith(
            'ckpt') else load_from_pytorch_checkpoint

        init_from_checkpoint(config['init_checkpoint'], assignment_map)

        task_tvars = {}
        for task_id in self.task_ids:
            task_tvars[task_id] = self.tasks[task_id].filter_tvars(tvars)

        print("**** Trainable Variables ****")
        tvars = self.filter_tvars(tvars, initialized_variable_names)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            modifiable_by = ''
            for task_id in self.task_ids:
                if var in task_tvars[task_id]:
                    modifiable_by += (task_id + '_')

            print("  name = %s, shape = %s%s, modified by = %s" % (var.name, var.shape, init_string, modifiable_by))

        num_train_steps = int(
            self.config['num_docs'] * self.config['num_epochs'])
        num_warmup_steps = int(num_train_steps * 0.1)
        print('num_warmup_steps', num_warmup_steps)

        self.global_step = tf.train.get_or_create_global_step()
        self.combined_loss = combine_losses(self.loss)

        if not self.config['interleave']:
            train_ops = []
            for task_id in self.task_ids:
                train_ops.append(optimization.create_custom_optimizer(task_tvars[task_id],
                                                                      self.loss[task_id], self.config['bert_learning_rate'],
                                                                      self.config['task_learning_rate'],
                                                                      num_train_steps, num_warmup_steps, False,
                                                                      self.global_step, freeze=freeze_bert_layer,
                                                                      task_opt=self.config['task_optimizer'],
                                                                      eps=config['adam_eps']))



            self.train_op = tf.group(train_ops)
        else:
            self.train_op = optimization.create_custom_optimizer(tvars,
                                                                 self.combined_loss, self.config['bert_learning_rate'],
                                                                 self.config['task_learning_rate'],
                                                                 num_train_steps, num_warmup_steps, False,
                                                                 self.global_step, freeze=freeze_bert_layer,
                                                                 task_opt=self.config['task_optimizer'],
                                                                 eps=config['adam_eps'])


    """
    drop some subset of parameters corresponding to a specific model component (encoder, mention detector, or linker)
    """
    def filter_tvars(self, tvars, initialized_variable_names):
        if self.config['freeze_antecedent']:
            tvars = util.drop_ante_link_variables(tvars, initialized_variable_names)
        if self.config['freeze_bert']:
            tvars = util.drop_bert_variables(tvars, initialized_variable_names)
        if self.config['freeze_md']:
            tvars = util.drop_md_variables(tvars)

        return tvars

    def restore(self, session):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() ]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        if self.config['primary_task'] == 'pcl':
            checkpoint_path = os.path.join(self.config["alt_log_dir"], "model.max.ckpt")

        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def setup_task(self, task_id, config):
        elif task_id == 'cl':
            task = CorefModel(config)
        elif task_id == 'md':
            task = MentionDetectionModel(config)
        elif task_id == 'mlm':
            task= MaskingModel(config)
        else:
            print('[setup_task] task_id:{} not implemented'.format(task_id))
            exit(0)
        return task

    def get_predictions_and_loss(self, task_id):
        input_tensors = self.input_tensors[task_id]
        predictions = self.tasks[task_id].get_predictions(*input_tensors)
        loss = self.tasks[task_id].get_loss(predictions)
        return predictions, loss



    """
    return list of tf summaries and list of f-scores for each task
    """
    def run_eval(self, session, eval_mode=False, save_predictions=False):
        pprefix = '[evaluate]'
        self.dh.get_eval_examples(self.setup_task(self.primary_task, self.config),eval_mode=eval_mode)
        evaluators = {}
        analyzers = {}

        for task_id in self.task_ids:
            evaluators[task_id] = self.tasks[task_id].setup_evaluation()
            analyzers[task_id] = self.tasks[task_id].setup_analysis()


        for example_num, (tensorized_example, example) in enumerate(self.dh.eval_data[self.primary_task]):

            for task_id in self.task_ids:
                tensorized_example = self.tasks[task_id].tensorize_example(example, is_training=False)
                feed_dict = {i: t for i, t in zip(self.input_tensors[task_id], tensorized_example)}

                loss, predictions = session.run(
                    [self.loss[task_id], self.predictions[task_id]], feed_dict=feed_dict)
                predictions['loss'] = loss
                if self.config['save_silver_mentions'] and task_id == 'md':
                    given_doc_keys = [x['doc_key'] for x in self.dh.task_train_examples[task_id]]
                    example['given'] = example['doc_key'] in given_doc_keys


                self.tasks[task_id].evaluate(predictions, example)

            if example_num % 10 == 0:
                print("Evaluated {} examples.".format(example_num + 1))

        if eval_mode or self.config['save_silver_mentions']:
            print(evaluators[self.primary_task].get_prf())
            for evaluator in self.tasks[self.primary_task].evaluators:
                evaluator.save_results()

        if 'cl' in self.tasks:
            for concept, evaluator in self.tasks['cl'].concept_coref_evaluators.items():
                evaluator.save_results()

        return [evaluators[task_id].make_summary() for task_id in self.task_ids], \
               evaluators[self.primary_task].get_f1()



    def start_enqueue_thread(self, session):
        train_examples = self.dh.get_train_examples(self.tasks)

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                if self.config['single_example']:
                    for task_example_set in train_examples:
                        for task_id, example in task_example_set:
                            tensorized_example = self.tasks[task_id].tensorize_example(example, is_training=True)
                            feed_dict = dict(zip(self.queue_input_tensors[task_id], tensorized_example))
                            session.run(self.enqueue_op[task_id], feed_dict=feed_dict)
                else:
                    print('[_enqueue_loop] only handling single example as of now')
                    exit(0)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()



