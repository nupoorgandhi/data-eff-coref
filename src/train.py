#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

import util
import logging
import sys
from tensorflow.python import debug as tf_debug
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
import random
from collections import defaultdict

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

if __name__ == "__main__":

  model_config = util.initialize_from_env()
  report_frequency = model_config["report_frequency"]
  eval_frequency = model_config["eval_frequency"]
  random.seed(model_config['rs'])

  model = util.get_trainer(model_config)
  saver = tf.train.Saver()

  log_dir = model_config["log_dir"]
  max_steps = model_config['num_epochs']*model_config['num_docs']

  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  print('max_steps', max_steps)

  max_f1 = 0
  prev_f1 = -1
  prev_f1_worse = False
  mode = 'w'


  # try to solve GPu issue
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  tf.set_random_seed(model_config['rs'])
  with tf.Session(config=config) as session:

    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)


    accumulated_loss = 0.0
    ckpt = tf.train.get_checkpoint_state(log_dir)
    print('ckpt:', ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    else:
      print('did not find existing model, training from scratch')

    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    finegrained_losses = defaultdict(float)
    task_ids = [x.lower().strip() for x in model_config['task_ids'].split('_')]
    while True:


      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])

      accumulated_loss += tf_loss[model_config['primary_task']]
      for task_id in task_ids:
        finegrained_losses[task_id] += tf_loss[task_id]


      if tf_global_step % report_frequency == 0 or tf_global_step > max_steps:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency

        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step,average_loss,steps_per_second))
        for task_id in task_ids:
          logger.info("[{}] {}_loss={:.2f}, steps/s={:.2f}".format(tf_global_step, task_id, finegrained_losses[task_id] / report_frequency, steps_per_second))

        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0
        finegrained_losses = defaultdict(float)



      if (tf_global_step  > 0 and tf_global_step % eval_frequency == 0) or tf_global_step > max_steps:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summaries, eval_f1 = model.run_eval(session)

        if eval_f1 > max_f1:
          saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)

          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        for eval_summary in eval_summaries:
          writer.add_summary(util.make_summary(eval_summary), tf_global_step)
          log_msgs = ['[{}] {}={:.2f}'.format(tf_global_step, k, v) for k,v in eval_summary.items()]
          for log_msg in log_msgs:
            logger.info(log_msg)

        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))

        if eval_f1 > max_f1 or tf_global_step > max_steps:
          saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
          print('[main] finished saving')
          print('[main] tf global_step', tf_global_step, 'max_steps', max_steps)


        if eval_f1 < prev_f1 and prev_f1_worse:#and ('tuneTrue' in sys.argv[1] :
            saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)

            break

        elif eval_f1 < prev_f1:
            prev_f1_worse = True
            prev_f1 = eval_f1
        else:
            prev_f1_worse = False
            prev_f1 = eval_f1

        if tf_global_step > max_steps: 
          break

  print('gpu memory used',session.run(tf.contrib.memory_stats.MaxBytesInUse()))
