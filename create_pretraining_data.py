# coding=utf-8
# The code refers to lajanugen's zeshel code

"""Create pretraining data examples."""

import os
import json
import random
import torch
import numpy as np
from transformers import BertTokenizer
import pickle




documents_file = "./documents"
out_file = "./out_data/document"

do_lower_case = True

max_seq_length = 256

random_seed = 12345


class TrainingInstance(object):
	"""A single training instance (sentence pair)."""

	def __init__(self, tokens, input_ids, segment_ids, word_ids):
		self.tokens = tokens
		self.input_ids = input_ids
		self.segment_ids = segment_ids

		self.word_ids = word_ids

	def __str__(self):
		s = ""
		s += "input_ids: %s\n" % (" ".join([str(x) for x in self.input_ids]))
		s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))

		s += "word_ids: %s\n" % (" ".join([str(x) for x in self.word_ids]))
		s += "\n"
		return s

	def __repr__(self):
		return self.__str__()

def pad_sequence(tokens, max_len):
	assert len(tokens) <= max_len
	return tokens + [0]*(max_len - len(tokens))

def create_instances_from_text(text, tokenizer, max_seq_length, rng):
	"""Creates `TrainingInstance`s from raw text."""

	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 3

	mention_length = int(max_num_tokens/2)

	text_instances = []
	for i in range(int(len(text)/(0.5*max_num_tokens))-1):

		start_pos = int(0.5*max_num_tokens*i)

		tokens = text[start_pos: start_pos+max_num_tokens]

		tokens_a = tokens[:mention_length]
		tokens_b = tokens[mention_length:]

		if not tokens_a or not tokens_b:
			continue

		tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		segment_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)
		# mention_id = [0]*len(tokens)
		word_ids = [0] + [1]*len(tokens_a) + [0] + [1]*len(tokens_b) + [0]

		assert len(input_ids) == max_seq_length

		instance = TrainingInstance(
			tokens = tokens,
			input_ids = input_ids,

			segment_ids=segment_ids,
			word_ids=word_ids)
		text_instances.append(instance)

	return text_instances


def create_pretraining_instances(input_file, field, tokenizer, max_seq_length, rng):
	"""Create `TrainingInstance`s from a single document."""
	all_text = []

	with open(documents_file + "/" + input_file, "r") as reader:
		
		while True:
			line = reader.readline()
			if not line:
				break
			line = line.strip()

			# # Empty lines are used as document delimiters
			# if not line:
			# 	all_documents.append([])

			line = json.loads(line)['text']
			tokens = tokenizer.tokenize(line)

			if tokens:
				all_text.append(tokens)

	# Remove empty documents
	# all_documents = [x for x in all_documents if x]
	rng.shuffle(all_text)

	print("--- Document {} have {} texts ---".format(field, len(all_text)))
	file_instances = []
	for text in all_text:
		file_instances.extend(create_instances_from_text(text, tokenizer, max_seq_length, rng))

	rng.shuffle(file_instances)
	return file_instances



def main():

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# documents_files = ["coronation_street.json", "elder_scrolls.json" , "muppets.json", "ice_hockey.json"]

	documents_files = ["forgotten_realms.json", "lego.json", "star_trek.json", "yugioh.json"]

	print("*** Reading from input files ***")

	print("--- List docment files ---")
	for file in documents_files:
		print(file)

	rng = random.Random(random_seed)

	for document in documents_files:

		field = document.split(".")[0]
		print("*** Generate instances for pretraining from document {}  ***".format(field))
		instances = create_pretraining_instances(document, field, tokenizer, max_seq_length, rng)
		print("*** Generate {} instances for pretraining from document {} ***".format(len(instances), field))

		output_file = "%s/%s.pkl" % (out_file, field)
		with open(output_file, "wb") as f:
	            pickle.dump(instances, f)
		

if __name__ == "__main__":
	main()
