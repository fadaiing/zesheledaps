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


field = "test"

documents_file = "./documents"
mentions_file = "./mentions/%s.json" % field
candidates_file = "./tfidf_candidates/%s.json" % field
out_file = "./out_data/mention_mse"

is_training = False
split_by_domain = True


do_lower_case = True
max_seq_length = 256
num_cands = 64

random_seed = 12345

MS = "[ms]"
ME = "[me]"
ENT = "[ent]"


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
	return tokens + [0]*(max_len - len(tokens))


def create_instances_from_mention_link(
	mention, all_documents, tfidf_candidates, tokenizer, max_seq_length,
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
	mention_text_tokenized = tokenizer.tokenize(mention['text'])

	mention_context, mention_start, mention_end = get_context_tokens(
		context_tokens, start_index, end_index, mention_length, tokenizer)

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
	instance_mention_id = []

	for cand_document_id in cand_document_ids:
		tokens_a = mention_context
		cand_document_title = all_documents[cand_document_id]['title']
		cand_document_text = all_documents[cand_document_id]['text'][len(cand_document_title):].strip()
		cand_document = cand_document_title + ' ' + ENT + " " + cand_document_text 
		# cand_document = cand_document_title +  ' '  + cand_document_text 
		cand_document_truncate = ' '.join(cand_document.split()[:cand_entity_length])
		cand_document = tokenizer.tokenize(cand_document_truncate)
		tokens_b = cand_document[:cand_entity_length]

		tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		segment_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
		input_mask = [1]*len(input_ids)
		mention_id = [0]*len(input_ids)

		# Update these indices to take [CLS] into account
		new_mention_start = mention_start + 1
		new_mention_end = mention_end + 1

		assert tokens[new_mention_start: new_mention_end+1] == mention_text_tokenized
		for t in range(new_mention_start, new_mention_end+1):
			mention_id[t] = 1

		assert len(input_ids) <= max_seq_length

		tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
		instance_tokens.extend(tokens)
		instance_input_ids.extend(pad_sequence(input_ids, max_seq_length))
		instance_segment_ids.extend(pad_sequence(segment_ids, max_seq_length))
		instance_input_mask.extend(pad_sequence(input_mask, max_seq_length))
		instance_mention_id.extend(pad_sequence(mention_id, max_seq_length))


	instance = TrainingInstance(
		tokens=instance_tokens,
		input_ids=instance_input_ids,
		input_mask=instance_input_mask,
		segment_ids=instance_segment_ids,
		label_id=label_id,
		mention_ids=instance_mention_id,
		mention_guid=mention['mention_id'],
		cand_guids=cand_document_ids)

	return instance

def create_trianing_instances(document_files, mentions_files, tokenizer, max_seq_length,
								rng, is_training=True):

	"""Create `TrainingInstance`s from raw text"""
	documents = {}
	for input_file in document_files:
		with open(documents_file + "/"+ input_file, "r") as reader:
			while True:
				line = reader.readline().strip()
				if not line:
					break
				line = json.loads(line)
				documents[line['document_id']] = line

	mentions = []
	for input_file in mentions_files:
		with open(input_file, "r") as reader:
			while True:
				line = reader.readline().strip()
				
				if not line:
					break
				line = json.loads(line)
				mentions.append(line)


	tfidf_candidates = {}
	with open(candidates_file, "r") as reader:
		while True:
			line = reader.readline().strip()

			if not line:
				break
			d = json.loads(line)
			tfidf_candidates[d['mention_id']] = d['tfidf_candidates']

	if split_by_domain:
		instances = {}
	else:
		instances = [] 

	print("--- Total {} mention links ---".format(len(mentions))) 
	for i, mention in enumerate(mentions):
		# print("mention {} ".format(i))
		instance = create_instances_from_mention_link(
				mention, documents, tfidf_candidates, tokenizer, max_seq_length,
				rng, is_training=is_training)

		if instance:
			if split_by_domain:
				corpus = mention['corpus']
				if corpus not in instances:
					instances[corpus] = []
				instances[corpus].append(instance)
			else:
				instances.append(instance)
		# else:
		# 	print(i)

		if i > 0 and i % 1000 == 0:
			print("Instance: %d" % i)

	if is_training:
		if split_by_domain:
			for corpus in instances:
				rng.shuffle(instances[corpus])
		else:
			rng.shuffle(instances)

	return instances			


def main():

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tokenizer.add_tokens(list([MS, ME, ENT]))

	documents_files = os.listdir(documents_file)
	# mentions_files = os.listdir(mentions_file)
	mentions_files = [mentions_file]

	print("=== Reading from input files ===")

	print("list docment files")
	for file in documents_files:
		print(file)
	
	print("list mention files")
	for file in mentions_files:
		print(file)

	print("=== Create training data from {} mention links ===".format(field))
	rng = random.Random(random_seed)
	instances = create_trianing_instances(
		documents_files, mentions_files, tokenizer, max_seq_length,
      rng, is_training=is_training
	)


	print("=== Writing training data to files ===")

	if split_by_domain:
		for corpus in instances:
			output_file = "%s/%s.pkl" % (out_file, corpus)
			with open(output_file, 'wb') as f:
				pickle.dump(instances[corpus], f)		
	else:
		output_file = "%s/%s.pkl" % (out_file, field)
		with open(output_file, 'wb') as f:
			pickle.dump(instances, f)


if __name__ == "__main__":
	main()
