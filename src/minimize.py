from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import tempfile
import subprocess
import collections
from string import punctuation
import conll
from bert import tokenization
import shlex
from collections import defaultdict
import copy
import itertools
import debug_util
import glob
import json
import pandas as pd
from mlm_tuning.load_contacts import get_contact_id_to_struct_feat_map
import pyhocon

CONCEPT_CODE_DICT = {'problem':1,
                     'person':2,
                     'test':3,
                     'treatment':4,
                     'pronoun':5}

class DocumentState(object):
  def __init__(self, key, concept_dict_str, umls_dict_str,
               contact_id_to_struct_feat=None, amr_concept_dict={}):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.clusters = collections.defaultdict(list)
    self.concept_clusters = collections.defaultdict(list)
    self.coref_stacks = collections.defaultdict(list)
    self.concept_stacks = collections.defaultdict(list)

    self.speakers = []
    self.segment_info = []
    self.token_ids =[]
    self.concept_dict_str = concept_dict_str
    self.umls_dict_str = umls_dict_str
    self.amr_concept_dict = amr_concept_dict
    self.umls_clusters = {}

    self.contact_id_to_struct_feat = contact_id_to_struct_feat
    self.dhs = not (contact_id_to_struct_feat == None)



  def get_contact_id(self):
    if not self.dhs:
      return None
    else:
      return int(self.doc_key.split('_')[1][:-1])

  def finalize(self):
    pprefix = '[finalize]'
    subtoken_idx = 0
    self.struct_feat = []

    for segment in self.segment_info:
      speakers = []
      token_ids = []

      for i, tok_info in enumerate(segment):
        speakers.append('[SPL]')
        token_ids.append(0)

      self.speakers += [speakers]
      self.token_ids += [token_ids]

    subtoken_text = []
    first_subtoken_index = -1
    for seg_idx, segment in enumerate(self.segment_info):
      speakers = []

      for i, tok_info in enumerate(segment):
        first_subtoken_index += 1

        coref = tok_info[11] if tok_info is not None else '-'
        concept_id = tok_info[-2] if tok_info is not None else '-'
        if concept_id not in ['Person', 'Treatment', 'Test', 'Problem']:
          concept_id = 0
        conll_text = tok_info[1] if tok_info is not None else '-'


        if coref != "-":
          last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
          for part in filter(bool,coref.split("|")):
            if len(part) > 0 and part[0] == "(":
              if part[-1] == ")":
                cluster_id = int(part[1:-1])

                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
                self.concept_clusters[concept_id].append((first_subtoken_index, last_subtoken_index))

              else:
                cluster_id = int(part[1:])
                self.coref_stacks[cluster_id].append(first_subtoken_index)
            else:
              cluster_id = int(part[:-1])
              start = self.coref_stacks[cluster_id].pop()
              self.clusters[cluster_id].append((start, last_subtoken_index))
              self.concept_clusters[concept_id].append((start, last_subtoken_index))


    subtoken_text = list(itertools.chain.from_iterable(self.segments))
    concept_clusters_text = defaultdict(list)
    self.umls_clusters[self.doc_key] = defaultdict(list)
    
    # merge clusters
    merged_clusters = []
    for c1 in self.clusters.values():
      existing = None
      for m in c1:
        for c2 in merged_clusters:
          if m in c2:
            existing = c2
            break
        if existing is not None:
          break
      if existing is not None:
        print("Merging clusters (shouldn't happen very often.)")
        existing.update(c1)
      else:
        merged_clusters.append(set(c1))
    merged_clusters = [list(c) for c in merged_clusters]
    def flatten(l):
        return [item for sublist in l for item in sublist]


    all_mentions = flatten(merged_clusters)
    sentence_map =  get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = flatten(self.segment_subtoken_map)
    assert len(all_mentions) == len(set(all_mentions))
    num_words =  len(flatten(self.segments))
    assert num_words == len(flatten(self.speakers))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      "doc_key": self.doc_key,
      "sentences": self.segments,
      "speakers": self.speakers,
      "constituents": [],
      "ner": [],
      "clusters": merged_clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
      'pronouns': self.pronouns,
      'token_ids':self.token_ids,
      "concept_clusters":self.concept_clusters,
      "umls_clusters":self.umls_clusters[self.doc_key],
      "struct_feat":self.struct_feat
    }



# we have a file of acronym names for internal DHS names. We use these acronyms to preprocess text
def get_acronyms():
  df = pd.read_csv("/path/to/acronyms.txt", header=None, dtype={"acro": str, "long": str})

  a_to_v = {}
  for i, row in df.iterrows():

    if str(row[0]) not in a_to_v:
      a_to_v[str(row[0])] = [row[1]]
      if str(row[0]).lower() != 'her':
        a_to_v[str(row[0]).lower()] = [row[1]]
        a_to_v[str(row[0]).upper()] = [row[1]]

    else:
      a_to_v[str(row[0])].append(row[1])
      if str(row[0]).lower() != 'her':
        a_to_v[str(row[0]).lower()].append(row[1])
        a_to_v[str(row[0]).upper()].append(row[1])
  return a_to_v


   
def normalize_word(word, language):
  acro_dict = get_acronyms()
  
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  elif word in acro_dict:
    return acro_dict[word][0]
  else:
    return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
  current = 0
  previous_token = 0
  while current < len(document_state.subtokens):
    end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current : end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current : end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  # assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map

def get_document_onto(document_lines, tokenizer, language, segment_len):
  document_state = DocumentState(document_lines[0],{}, {})
  word_idx = -1
  for line in document_lines[1]:
    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      # assert len(row) >= 12
      # assert len(row) >= 6
      word_idx += 1
      #TODO
      word = normalize_word(row[3], language)
      # word = normalize_word(row[0], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document


def get_document_dhs(document_lines, tokenizer, language, segment_len, dhs_dict_str)
  document_state = DocumentState(document_lines[0],dhs_dict_str, {})
  word_idx = -1
  for line in document_lines[1]:

    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      word_idx += 1
      word = normalize_word(row[3], language)
      subtokens = [word] if language == 'notokenization' else tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  if language != 'notokenization':
    split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document



def get_document_i2b2(document_lines, tokenizer, language, segment_len, concept_dict_str, umls_dict_str):
  document_state = DocumentState(document_lines[0], concept_dict_str, umls_dict_str)
  word_idx = -1
  for line in document_lines[1]:
    row = line.split()
    sentence_end = len(row) == 0
    if not sentence_end:
      word_idx += 1
      if language == 'normalized-i2b2':
        word = normalize_word(row[1], language)
      else:
        word = normalize_word(row[0], language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else (row + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    else:
      document_state.sentence_end[-1] = True
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document

def skip(doc_key):
  return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir,
                       alignment_dir=''):
  input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
  output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
  amr_concept_dict = None
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document_onto(document_lines, tokenizer, language, seg_len, amr_concept_dict)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))

def get_dhs_dict_str(directory):
  pprefix = '[get_dhs_dict_str]'
  concept_dict = {}
  docs = glob.glob(directory + '/*.json', recursive=True)

  for doc in docs:
    doc_id = '({})_0'.format(os.path.basename(doc)[:-len('.json')])
    concept_dict[doc_id] = {}

    with open(doc, 'r') as f:
      jsonlines = json.load(f)['_referenced_fss']
    text = jsonlines["1"]["sofaString"]

    for span_item in jsonlines.keys():
      if span_item == "1":
        continue
      if 'referenceType' not in jsonlines[span_item]:
        continue
      if 'begin' in jsonlines[span_item]:
        begin_idx = jsonlines[span_item]['begin']
      else:
        begin_idx = 0
      end_idx = jsonlines[span_item]['end']
      span_text = text[begin_idx:end_idx].lower().strip().replace(" ", "")
      concept = jsonlines[span_item]['referenceType']
      concept_dict[doc_id][span_text] = concept

  return concept_dict

def minimize_partition_dhs(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir,
                           alignment_dir):
  input_path = "{}/{}.{}".format(input_dir, name, extension)
  output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX_I2B2_ALT, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), 0)
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)
  dhs_dict_str = get_dhs_dict_str('../../data/concepts/')
  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document_dhs(document_lines, tokenizer, language, seg_len, dhs_dict_str)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))




def minimize_partition_i2b2(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
  input_path = "{}/{}.{}".format(input_dir, name, extension)
  output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX_I2B2_ALT, line)
      if begin_document_match:
        doc_key = conll.get_doc_key(begin_document_match.group(1), 0)
        documents.append((doc_key, []))
      elif line.startswith("#end document"):
        continue
      else:
        documents[-1][1].append(line)

  with open(output_path, "w") as output_file:
    for document_lines in documents:
      if skip(document_lines[0]):
        continue
      document = get_document_i2b2(document_lines, tokenizer, language, seg_len, {}, {})
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))





def minimize_language(language, labels, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case,
                      concept_dir, alignment_dir):
  # do_lower_case = True if 'chinese' in vocab_file else False
  tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)

  minimize_partition_dhs('train', language, 'dhs.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, alignment_dir)
  minimize_partition_dhs('dev', language, 'dhs.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, alignment_dir)
  minimize_partition_dhs('test', language, 'dhs.conll', labels, stats, tokenizer, seg_len, input_dir, output_dir, alignment_dir)
  
  minimize_partition_i2b2('test.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition_i2b2('dev.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition_i2b2('train.i2b2', language, 'conll', labels, stats, tokenizer, seg_len, input_dir, output_dir)
  
  minimize_partition("dev", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("train", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)



if __name__ == "__main__":

  try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
  except KeyError:
    user_paths = []


  config = pyhocon.ConfigFactory.parse_file(sys.argv[1])[sys.argv[2]]
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  if not os.path.isdir(config['output_dir']):
    os.mkdir(config['output_dir'])
  for seg_len in [ 512, 256, 128, 384]:
    minimize_language(config['language'], labels, stats, config['vocab_file'], seg_len,
                      config['input_dir'], config['output_dir'],
                      config['do_lower_case'],
                      config['concept_dir'],
                      config['alignments_dir'])
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
