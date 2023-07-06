#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import util


if __name__ == "__main__":

  model_config = util.initialize_from_env()
  model = util.get_trainer(model_config)
  saver = tf.train.Saver()
  with tf.Session() as session:
    model.restore(session)
    # Make sure eval mode is True if you want official conll results
    model.run_eval(session, eval_mode=True)
