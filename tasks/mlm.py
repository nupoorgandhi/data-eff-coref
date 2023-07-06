
from bert import tokenization, modeling, create_pretraining_data
import numpy as np
import tensorflow as tf
import random
import metrics
def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

class MaskingModel(object):
	def __init__(self, config):
		self.config = config
		self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
		self.max_segment_len = config['max_segment_len']

		self.tokenizer = tokenization.FullTokenizer(
			vocab_file=config['vocab_file'], do_lower_case=False)
		self.input_props = self.get_input_props()

	def filter_tvars(self, tvars):
		return tvars

	def package_predictions(self, log_probs, one_hot_labels, label_weights):
		return dict(zip('log_probs, one_hot_labels, label_weights'.replace(" ", "").split(','),
						[log_probs, one_hot_labels, label_weights]))

	def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
							 label_ids, label_weights):
		"""Get loss and log probs for the masked LM."""
		input_tensor = gather_indexes(input_tensor, positions)

		with tf.variable_scope("cls/predictions"):
			# We apply one more non-linear transformation before the output layer.
			# This matrix is not used after pre-training.
			with tf.variable_scope("transform"):
				input_tensor = tf.layers.dense(
					input_tensor,
					units=bert_config.hidden_size,
					activation=modeling.get_activation(bert_config.hidden_act),
					kernel_initializer=modeling.create_initializer(
						bert_config.initializer_range))
				input_tensor = modeling.layer_norm(input_tensor)

			# The output weights are the same as the input embeddings, but there is
			# an output-only bias for each token.
			output_bias = tf.get_variable(
				"output_bias",
				shape=[bert_config.vocab_size],
				initializer=tf.zeros_initializer())
			logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
			logits = tf.nn.bias_add(logits, output_bias)
			log_probs = tf.nn.log_softmax(logits, axis=-1)

			label_ids = tf.reshape(label_ids, [-1])
			label_weights = tf.reshape(label_weights, [-1])

			one_hot_labels = tf.one_hot(
				label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

			# The `positions` tensor might be zero-padded (if the sequence is too
			# short to have the maximum number of predictions). The `label_weights`
			# tensor has a value of 1.0 for every real prediction and 0.0 for the
			# padding predictions.
			# per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
			# numerator = tf.reduce_sum(label_weights * per_example_loss)
			# denominator = tf.reduce_sum(label_weights) + 1e-5
			# loss = numerator / denominator

		return self.package_predictions(log_probs, one_hot_labels, label_weights)
		# return (loss, per_example_loss, log_probs)

	def get_predictions(self, input_ids, input_mask, is_training,
						masked_lm_positions, masked_lm_ids, masked_lm_weights):

		model = modeling.BertModel(
			config=self.bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			use_one_hot_embeddings=False,
			scope='bert')

		return self.get_masked_lm_output(self.bert_config,
										 model.get_sequence_output(),
										 model.get_embedding_table(),
										 masked_lm_positions,
										 masked_lm_ids,
										 masked_lm_weights)

	def get_loss(self, predictions):
		per_example_loss = -tf.reduce_sum(predictions['log_probs'] * predictions['one_hot_labels'], axis=[-1])
		per_example_loss = tf.cast(per_example_loss, tf.float32)
		predictions['label_weights'] = tf.cast(predictions['label_weights'], tf.float32)

		numerator = tf.reduce_sum(predictions['label_weights'] * per_example_loss)
		denominator = tf.reduce_sum(predictions['label_weights']) + 1e-5
		loss = numerator / denominator
		return loss

	def get_input_props(self):
		input_props = []
		input_props.append((tf.int32, [None, None]))  # input_ids.
		input_props.append((tf.int32, [None, None]))  # input_mask
		input_props.append((tf.bool, []))  # Is training.
		input_props.append((tf.int32, [None, None]))  # masked_lm_positions.
		input_props.append((tf.int32, [None, None]))  # masked_lm_ids.
		input_props.append((tf.int32, [None, None]))  # masked_lm_weights
		return input_props


	def tensorize_example(self, example, is_training):
		sentences = example["sentences"]  # [:stopping_sentence]

		input_ids, input_mask = [], []
		masked_lm_ids, masked_lm_positions, masked_lm_weights = [],[],[]

		masked_lm_prob = .15
		max_predictions_per_seq = 20
		vocab_words = list(tokenization.load_vocab(self.config['vocab_file']).keys())
		rng = random.Random(self.config['rs'])
		max_sentence_length = self.max_segment_len

		for i, sentence in enumerate(sentences):

			sent_output_tokens, sent_masked_lm_positions, sent_masked_lm_labels = create_pretraining_data.create_masked_lm_predictions(
				sentence,
				masked_lm_prob,
				max_predictions_per_seq,
				vocab_words,
				rng)
			sent_masked_lm_ids = self.tokenizer.convert_tokens_to_ids(sent_masked_lm_labels)
			sent_masked_lm_weights = [1.0] * len(sent_masked_lm_ids)
			while len(sent_masked_lm_positions) < max_predictions_per_seq:
				sent_masked_lm_positions.append(0)
				sent_masked_lm_ids.append(0)
				sent_masked_lm_weights.append(0.0)

			masked_lm_ids.append(sent_masked_lm_ids)
			masked_lm_positions.append(sent_masked_lm_positions)
			masked_lm_weights.append(sent_masked_lm_weights)


			sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_output_tokens)
			sent_input_mask = [1] * len(sent_input_ids)
			# sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
			while len(sent_input_ids) < max_sentence_length:
				sent_input_ids.append(0)
				sent_input_mask.append(0)
				# sent_speaker_ids.append(0)
			input_ids.append(sent_input_ids)
			input_mask.append(sent_input_mask)

		input_ids = np.array(input_ids)  # [:stopping_sentence]
		input_mask = np.array(input_mask)
		masked_lm_ids = np.array(masked_lm_ids)
		masked_lm_positions = np.array(masked_lm_positions)
		masked_lm_weights = np.array(masked_lm_weights)

		example_tensors = (input_ids, input_mask, is_training,
						   masked_lm_positions, masked_lm_ids, masked_lm_weights)
		if is_training and len(sentences) > self.config["max_training_sentences"]:
			if self.config['single_example']:
				return self.truncate_example(*example_tensors)
			else:
				offsets = range(self.config['max_training_sentences'], len(sentences),
								self.config['max_training_sentences'])
				tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
				return tensor_list
		else:
			return example_tensors

	def truncate_example(self, input_ids, input_mask, is_training,
							 masked_lm_positions, masked_lm_ids, masked_lm_weights, sentence_offset=None):
		max_training_sentences = self.config["max_training_sentences"]
		num_sentences = input_ids.shape[0]
		assert num_sentences > max_training_sentences

		sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset

		input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
		input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
		masked_lm_positions = masked_lm_positions[sentence_offset:sentence_offset + max_training_sentences, :]
		masked_lm_ids = masked_lm_ids[sentence_offset:sentence_offset + max_training_sentences, :]
		masked_lm_weights = masked_lm_weights[sentence_offset:sentence_offset + max_training_sentences, :]

		return input_ids, input_mask, is_training, masked_lm_positions, masked_lm_ids, masked_lm_weights

	def setup_evaluation(self):
		self.mlm_evaluator = metrics.MLMEvaluator()
		return self.mlm_evaluator
	
        def evaluate(self, predictions, example):
		self.mlm_evaluator.update(predictions['loss'])#self.get_loss(predictions).item())

