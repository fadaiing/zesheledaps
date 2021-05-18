# coding=utf-8
# The code refer to lajanugen's zeshel code

"""Create training data examples."""

import os
import json
import math
import random
import torch
import numpy as np
from transformers import BertTokenizer
import pickle


# val_fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
test_fields = ["forgotten_realms", "lego", "star_trek", "yugioh"]
documents_file = "./documents"
mentions_file = "./mentions/test.json"
candidates_file = "./tfidf_candidates/test.json"
entity_file = "./out_data/entity_predict"
out_file = "./out_data/mention_el_em_side"


is_training = False
split_by_domain = True

do_lower_case = True
max_seq_length = 256
num_cands = 64

random_seed = 12345

MS = "[ms]"
MM = "[mm]"
ME = "[me]"
ENT = "[ent]"
ENTM = "[entm]"


class TrainingInstance(object):
	"""A single set of features of data."""

	def __init__(self,
				tokens,
				input_ids,
				input_mask,
				segment_ids,
				label_id,
				mention_ids,
				mention_guid,
				cand_guids):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.mention_ids = mention_ids
		self.mention_guid = mention_guid
		self.cand_guids = cand_guids

	def __str__(self):
		s = ""
		s += "input_ids: %s\n" % (" ".join([str(x) for x in self.input_ids[:max_seq_length]]))
		s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids[:max_seq_length]]))
		s += "input_mask: %s\n" % (" ".join([str(x) for x in self.input_mask[:max_seq_length]]))
		s += "mention_id: %s\n" % (" ".join([str(x) for x in self.mention_id[:max_seq_length]]))
		s += "label_id: %d\n" % self.label_id
		s += "\n"
		
		return s

def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
	start_pos = start_index - max_tokens
	if start_pos < 0:
		start_pos = 0
	prefix = ' '.join(context_tokens[start_pos: start_index])
	suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
	prefix = tokenizer.tokenize(prefix)
	suffix = tokenizer.tokenize(suffix)
	mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index+1]))

	assert len(mention) < max_tokens

	remaining_tokens = max_tokens - len(mention)
	half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

	mention_context = []

	if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
		prefix_len = half_remaining_tokens
	elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
		prefix_len = remaining_tokens - len(suffix)
	elif len(prefix) < half_remaining_tokens:
		prefix_len = len(prefix)

	if prefix_len > len(prefix):
		prefix_len = len(prefix)

	prefix = prefix[-prefix_len:]

	mention_context = prefix + mention + suffix
	mention_start = len(prefix)
	mention_end = mention_start + len(mention) - 1
	mention_context = mention_context[:max_tokens]

	assert mention_start <= max_tokens
	assert mention_end <= max_tokens

	return mention_context, mention_start, mention_end

def pad_sequence(tokens, max_len):
	assert len(tokens) <= max_len
	return tokens + [0] * (max_len - len(tokens))

def create_instances_from_mention_link(
	mention, all_documents, tfidf_candidates, tokenizer, add_map, max_seq_length,
	rng, is_training=True):

	"""Creates TrainingInstances for a mention link"""

	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 3

	mention_length = int(max_num_tokens/2)        
	cand_entity_length = max_num_tokens - mention_length

	context_document_id = mention['context_document_id']
	label_document_id = mention['label_document_id']
	start_index = mention['start_index']
	end_index = mention['end_index']

	context_document = all_documents[context_document_id]['text']
	context_tokens = context_document.split()
	extracted_mention = context_tokens[start_index: end_index+1]
	extracted_mention = ' '.join(extracted_mention)
	context_tokens.insert(start_index, MS)
	context_tokens.insert(end_index + 2, ME)
	start_index += 1
	end_index += 1
	assert extracted_mention == mention['text']
	# mention_text_tokenized = tokenizer.tokenize(mention['text'])

	mention_context, mention_start, mention_end = get_context_tokens(
		context_tokens, start_index, end_index, mention_length-2, tokenizer)

	mention_id = mention['mention_id']
	assert mention_id in tfidf_candidates

	cand_document_ids = tfidf_candidates[mention_id]
	if not cand_document_ids:
		return None

	if not is_training:
		cand_document_ids = cand_document_ids[:num_cands]

	if not is_training and label_document_id not in cand_document_ids:
		return None

	cand_document_ids = [cand for cand in cand_document_ids if cand != label_document_id]
	assert label_document_id not in cand_document_ids

	while len(cand_document_ids) < num_cands:
		cand_document_ids.extend(cand_document_ids)

	cand_document_ids.insert(0, label_document_id)

	cand_document_ids = cand_document_ids[:num_cands]
	assert len(cand_document_ids) == num_cands

	label_id = None
	for i, document in enumerate(cand_document_ids):
		if document == label_document_id:
			assert label_id == None
			label_id = i

	assert label_id == 0	        

	instance_tokens = []
	instance_input_ids = []
	instance_segment_ids = []
	instance_input_mask = []
	instance_mention_ids = []

	for cand_document_id in cand_document_ids:
		tokens_a = ['[CLS]'] + mention_context + ['[SEP]']
		tokens_a.insert(1, MM)

		mention_input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
		mention_input_ids.insert(1, add_map[mention_id])
		# mention_input_ids.insert(mention_start+1, add_map[mention_id])
		tokens_a.insert(1, mention['text'].lower())
		# tokens_a.insert(mention_start+1, mention['text'].lower())
	
		cand_document_title = all_documents[cand_document_id]['title']
		cand_document_text = all_documents[cand_document_id]['text'][len(cand_document_title):].strip()
		cand_document = cand_document_title + ' ' + ENT + cand_document_text
		
		cand_document_truncate = ' '.join(cand_document.split()[:cand_entity_length])
		cand_document = tokenizer.tokenize(cand_document_truncate)
		tokens_b = cand_document[:cand_entity_length-2] + [ENTM] + ['[SEP]']

		# cand_document_text = cand_document_text[len(cand_document_title):].strip()
		# cand_document_text_tokens = tokenizer.tokenize(cand_document_text)
		# cand_document_title_tokens = tokenizer.tokenize(cand_document_title)
		# tokens_b = cand_document_title_tokens + [ENT] + [ENTM]+ cand_document_text_tokens[:cand_entity_length-3-len(cand_document_title_tokens)] + ['[SEP]']
	
		cand_document_input_ids = tokenizer.convert_tokens_to_ids(tokens_b)
		cand_document_input_ids.insert(-1, add_map[cand_document_id])
		
		tokens_b.insert(-1, cand_document_title.lower())

		tokens = tokens_a + tokens_b
		input_ids =  mention_input_ids + cand_document_input_ids

		# print(tokens)
		# print(len(tokens))
		# print(input_ids)
		# print(len(input_ids))

		# exit()

		segment_ids = [0]*(len(mention_input_ids)) + [1]*(len(cand_document_input_ids))
		input_mask = [1]*len(input_ids)
		mention_ids = [0]*len(input_ids)

		# Update these indices to take [CLS] into account
		new_mention_start = mention_start + 1
		new_mention_end = mention_end + 1

		# print(tokens[new_mention_start: new_mention_end+1])
		# print([mention['text'].lower()])
		# assert tokens[new_mention_start] == mention['text'].lower()
		for t in range(new_mention_start, new_mention_end+1):
			mention_ids[t] = 1

		assert len(input_ids) <= max_seq_length

		tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
		instance_tokens.extend(tokens)
		instance_input_ids.extend(pad_sequence(input_ids, max_seq_length))
		instance_segment_ids.extend(pad_sequence(segment_ids, max_seq_length))
		instance_input_mask.extend(pad_sequence(input_mask, max_seq_length))
		instance_mention_ids.extend(pad_sequence(mention_ids, max_seq_length))

		# print(instance_tokens)
		# print(len(instance_tokens))
		# print(instance_input_ids)
		# print(len(instance_input_ids))

		# exit()

	instance = TrainingInstance(
		tokens=instance_tokens,
		input_ids=instance_input_ids,
		input_mask=instance_input_mask,
		segment_ids=instance_segment_ids,
		label_id=label_id,
		mention_ids=instance_mention_ids,
		mention_guid=mention['mention_id'],
		cand_guids=cand_document_ids)

	return instance

def create_trianing_instances(documents_file_field, mentions_file_field, field, tokenizer, em_map, max_seq_length,
								rng, is_training=True):

	"""Create `TrainingInstance`s from raw text"""

	print(documents_file_field)
	field_documents = {}
	with open(documents_file_field, "r") as reader:
		while True:
			line = reader.readline().strip()
			if not line:
				break

			line = json.loads(line)
			field_documents[line['document_id']] = line

	print(mentions_file_field)
	field_mentions = []
	with open(mentions_file_field, "r") as reader:
		while True:
			line = reader.readline().strip()			
			if not line:
				break

			line = json.loads(line)
			if line["corpus"] == field:
				field_mentions.append(line)
			
	tfidf_candidates = {}
	with open(candidates_file, "r") as reader:
		while True:
			line = reader.readline().strip()

			if not line:
				break
			d = json.loads(line)
			tfidf_candidates[d['mention_id']] = d['tfidf_candidates']

	instances = [] 
	print("--- Total {} mention links ---".format(len(field_mentions))) 
	for i, mention in enumerate(field_mentions):
		# print("mention {} : {} ".format(i, mention["text"]))
		# start_time = time.time()
		instance = create_instances_from_mention_link(
				mention, field_documents, tfidf_candidates, tokenizer, em_map, max_seq_length,
				rng, is_training=is_training)
		# end_time = time.time()
		# print('epoch time cost', end_time-start_time,'s')

		if instance:
			instances.append(instance)
		# else:
		# 	print(i)

		if i > 0 and i % 100 == 0:
			print("Instance: %d" % i)

	rng.shuffle(instances)

	return instances			


def create_data(field):

	print("=== Prepare tokenizer for {} ===".format(filed))
	field_tokenizer_file = entity_file + "/" + "%s_tokenizer/" % field
	filed_tokenizer = BertTokenizer.from_pretrained(field_tokenizer_file)

	em_map_file = "%s/em_map.json" % (field_tokenizer_file)
	with open(em_map_file, 'r') as f:
		filed_em_map = json.load(f)

	vocab_size = len(filed_tokenizer) + len(filed_em_map)

	print("Adding {} entity tokens into vocab".format(len(filed_em_map)))
	print("Vocabulary size {}".format(vocab_size))

	filed_documents_file = documents_file + "/%s.json" % field
	filed_mentions_file = mentions_file

	print("=== Reading from input files ===")
	print(filed_documents_file)
	print(filed_mentions_file)

	print("=== Create training data from mention links ===")
	rng = random.Random(random_seed)
	instances = create_trianing_instances(
		filed_documents_file, filed_mentions_file, field, filed_tokenizer, filed_em_map, max_seq_length,
      rng, is_training=is_training
	)

	print("=== Writing training data to files ===")
	output_file = "%s/%s.pkl" % (out_file, field)
	with open(output_file, 'wb') as f:
		pickle.dump(instances, f)
	print("--- Write Success ---")


if __name__ == "__main__":
	for filed in test_fields:
		create_data(filed)
