# coding=utf-8
# The code refer to lajanugen's zeshel code

import os
import json
import pickle
import random
import math
from transformers import BertTokenizer


# val_fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
test_fields = ["forgotten_realms", "lego", "star_trek", "yugioh"]
documents_file = "./documents"
mentions_file = "./mentions/test.json"
out_file = "./out_data/entity_predict"

do_lower_case = True

max_seq_length = 256

random_seed = 12345

MS = "[ms]"
MM = "[mm]"
ME = "[me]"
ENT = "[ent]"
ENTM = "[entm]"

class MEinstance(object):
	"""A single mention entity training instance"""

	def __init__(self, tokens, input_ids, input_mask, segment_ids, em_labels):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.em_labels = em_labels


def pad_sequence(tokens, max_len):
	assert len(tokens) <= max_len
	return tokens + [0]*(max_len - len(tokens))

def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
	
	start_pos = start_index - max_tokens
	if start_pos < 0:
		start_pos = 0

	prefix = ' '.join(context_tokens[start_pos: start_index])
	suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])

	prefix = tokenizer.tokenize(prefix)
	suffix = tokenizer.tokenize(suffix)

	# mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index+1]))
	# assert len(mention) < max_tokens

	remaining_tokens = max_tokens
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

	mention_context = prefix + suffix
	mention_start = len(prefix)
	mention_end = mention_start
	mention_context = mention_context[:max_tokens]

	assert mention_start <= max_tokens
	assert mention_end <= max_tokens

	return mention_context, mention_start, mention_end


def create_instance_from_mention_link(mention, all_documents, tokenizer, add_map, max_seq_length, rng):

	# Account for [CLS] [SEP] and mention
	mention_length = max_seq_length - 3

	mention_id = mention["mention_id"]
	context_document_id = mention['context_document_id']
	label_document_id = mention['label_document_id']
	start_index = mention['start_index']
	end_index = mention['end_index']

	context_document = all_documents[context_document_id]['text']
	context_tokens = context_document.split()
	extracted_mention = context_tokens[start_index: end_index+1]
	extracted_mention = ' '.join(extracted_mention)
	# context_tokens.insert(start_index, MS)
	# context_tokens.insert(end_index + 2, ME)
	# start_index += 1
	# end_index += 1
	assert extracted_mention == mention['text']

	# mention_text_tokenized = tokenizer.tokenize(mention['text'])
	
	mention_context, mention_start, mention_end = get_context_tokens(
		context_tokens, start_index, end_index, mention_length, tokenizer)


	instance_tokens = ['[CLS]'] + mention_context + ['[SEP]']
	# 加上 [CLS] 之后实体向后移动一个位置 
	instance_input_ids = tokenizer.convert_tokens_to_ids(instance_tokens)
	instance_input_ids.insert(mention_start + 1, add_map[mention_id])
	instance_tokens.insert(mention_start + 1, mention['text'])
	
	instance_input_mask = [1] * len(instance_input_ids)
	instance_segment_ids = [0] * len(instance_input_ids)
	
	instance_input_ids = pad_sequence(instance_input_ids, max_seq_length)
	instance_input_mask = pad_sequence(instance_input_mask, max_seq_length)
	instance_segment_ids = pad_sequence(instance_segment_ids, max_seq_length)

	instance_labels = [-100] * len(instance_input_ids)
	instance_labels[mention_start+1] = instance_input_ids[mention_start+1]
	instance_input_ids[mention_start+1] = tokenizer.mask_token_id

	assert len(instance_input_ids) == max_seq_length


	instance = MEinstance(
		tokens=instance_tokens,
		input_ids=instance_input_ids,
		input_mask=instance_input_mask,
		segment_ids=instance_segment_ids,
		em_labels = instance_labels
	)

	return instance


def create_instance_from_entity_description(document_id, title, text, tokenizer, add_map, max_seq_length):
	
	# Account for [CLS] E [SEP] 
	entity_length = max_seq_length - 3

	text = text[len(title):].strip()

	text_tokens = tokenizer.tokenize(text)

	instance_tokens = ['[CLS]'] + text_tokens[:entity_length] + ['[SEP]']

	instance_input_ids = tokenizer.convert_tokens_to_ids(instance_tokens)
	instance_input_ids.insert(1, add_map[document_id]) 
	instance_tokens.insert(1, title)

	instance_input_mask = [1] * len(instance_input_ids)
	instance_segment_ids = [0] * len(instance_input_ids)
	

	instance_input_ids = pad_sequence(instance_input_ids, max_seq_length)
	instance_input_mask = pad_sequence(instance_input_mask, max_seq_length)
	instance_segment_ids = pad_sequence(instance_segment_ids, max_seq_length)

	instance_labels = [-100] * len(instance_input_ids)
	instance_labels[1] = instance_input_ids[1]

	instance_input_ids[1] = tokenizer.mask_token_id

	assert len(instance_input_ids) == max_seq_length
	assert len(instance_labels) == max_seq_length


	instance = MEinstance(
		tokens=instance_tokens,
		input_ids=instance_input_ids,
		input_mask=instance_input_mask,
		segment_ids=instance_segment_ids,
		em_labels = instance_labels

	)

	return instance


def create_data(field):

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tokenizer.add_tokens(list([MS, ME, ENT, MM, ENTM]))

	vocab_size = len(tokenizer)

	print("Vocabulary size {}".format(vocab_size))

	print("=== Reading from input files ===")
	filed_documents_file = documents_file + "/%s.json" % field
	print(filed_documents_file)
	field_documents = {}
	with open(filed_documents_file, "r") as reader:
		while True:
			line = reader.readline().strip()
			if not line:
				break

			line = json.loads(line)
			field_documents[line['document_id']] = line
	print(mentions_file)
	field_mentions =  []
	with open(mentions_file, "r") as reader:
		while True:
			line = reader.readline().strip()
			if not line:
				break
			
			line = json.loads(line)
			if line["corpus"] == field:
				field_mentions.append(line)

	em_map = {}
	for i, mention in enumerate(field_mentions):
		em_map[mention["mention_id"]] = vocab_size + i
	
	vocab_size += len(field_mentions) 

	for j, doc in enumerate(field_documents):
		em_map[field_documents[doc]["document_id"]] = vocab_size + j

	vocab_size += len(field_documents)
	
	assert len(em_map) == len(field_mentions) + len(field_documents)
	assert vocab_size == len(tokenizer) + len(em_map)

	print("Adding {} entity tokens into vocab".format(len(em_map)))
	print("Vocabulary size {}".format(vocab_size))

	print("=== Generate training instance for entity tokens ===")
	rng = random.Random(random_seed)
	instances = []

	print("--- Total {} mention links ---".format(len(field_mentions)))
	for i, mention in enumerate(field_mentions):
		instance = create_instance_from_mention_link(
				mention, field_documents, tokenizer, em_map, max_seq_length, rng
				)

		if instance:
			instances.append(instance)

		if i > 0 and i % 100 == 0:
			print("Instance: %d" % i)

	print("--- Total {} entity description ---".format(len(field_documents)))
	for i, doc_id in enumerate(field_documents):
		instance = create_instance_from_entity_description(
			doc_id, field_documents[doc_id]["title"], field_documents[doc_id]["text"], tokenizer, em_map, max_seq_length
			)

		if instance:
			instances.append(instance)

		if i > 0 and i % 1000 == 0:
			print("Instance: %d" % i)

	print("=== Write traing instances and tokenizer to files  ===")
	output_file = "%s/%s_me.pkl" % (out_file, field)
	with open(output_file, 'wb') as f:
		pickle.dump(instances, f)
	out_tokenizer_file = "%s/%s_tokenizer" % (out_file, filed)
	tokenizer.save_pretrained(out_tokenizer_file)
	out_em_map_file = "%s/em_map.json" % (out_tokenizer_file)
	with open(out_em_map_file, "w") as f:
		json.dump(em_map, f, indent=4)
	print("Save success")
	
	
if __name__ == "__main__":
	for filed in test_fields:
		create_data(filed)

