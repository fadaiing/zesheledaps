# coding=utf-8
# The code refer to lajanugen's zeshel code

"""Training the linking model"""

import pickle
import torch
import argparse
import torch.nn as nn
import numpy as np
import random
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from create_training_data import TrainingInstance

import time


fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
mention_train_file = "./out_data/mention_mse"
mention_valid_file = "./out_data/mention_mse"
out_file = "./out_data/mention_mse"
model_file = "./out_data/model"


epochs = 10
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 2048
num_cands = 64
max_seq_length = 256

lr_rate = 2e-5
weight_decay = 0.0

random_seed = 12345 

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device GPU or CPU')

parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")

args = parser.parse_args()

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(random_seed)

def get_batch_data(train_data, batch_index, batch_instance_num):

	batch_instances = train_data[batch_instance_num * batch_index : batch_instance_num * (batch_index + 1)]

	batch_token_ids = []
	batch_token_masks = []
	batch_segment_ids = []
	batch_labels = []
		

	for instance in batch_instances:
		instance_token_ids = np.array(instance.input_ids).reshape(-1, max_seq_length)
		assert instance_token_ids.shape[0] == num_cands
		batch_token_ids.extend(instance_token_ids)

		instance_token_masks = np.array(instance.input_mask).reshape(-1, max_seq_length)
		assert instance_token_masks.shape[0] == num_cands
		batch_token_masks.extend(instance_token_masks)

		instance_segment_ids = np.array(instance.segment_ids).reshape(-1, max_seq_length)
		assert instance_segment_ids.shape[0] == num_cands
		batch_segment_ids.extend(instance_segment_ids)

		assert instance.label_id == 0
		# instance_labels = [1] + [0] * 63
		batch_labels.append(instance.label_id)


	return torch.LongTensor(batch_token_ids), torch.FloatTensor(batch_token_masks), \
	 		torch.LongTensor(batch_segment_ids), torch.LongTensor(batch_labels)

		
def train(field, num_epoch):

	print("=======      predict for  {}       =========".format(field))

	print("=== Prepare eval datasets from {}===".format(mention_valid_file))
	with open("%s/%s.pkl" % (mention_valid_file, field),"rb") as f:
		field_data = pickle.load(f)

	print("eval instance nums : {}".format(len(field_data)))

	print("=== Prepare model and optimizer ===")
	device = torch.device(args.device)
	ckpt_file = model_file + "/" + "pretrain_train_el_%s_%s.ckpt" % (field, num_epoch)
	print("=== Load model from {} ===".format(ckpt_file))
	model = BertForSequenceClassification.from_pretrained(ckpt_file, num_labels=1, return_dict=True)
	print("Vocabulary size {}".format(model.config.vocab_size))

	model = torch.nn.DataParallel(model)
	model.to(device)

	print("Model has %s parameters" % sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	eval(model, field, field_data)


def eval(model, field, eval_data):

	print("eval instance nums : {}".format(len(eval_data)))

	batch_instance_num = BATCH_SIZE_TEST // num_cands
	print("eval batch instance num : ", batch_instance_num)

	# device = torch.device(args.device)
	# model = torch.nn.DataParallel(model)
	# model.to(device)

	instance_predict = []
	model.eval( )
	with torch.no_grad():
		print("--- Eval field {} ---".format(field))

		total = 0
		correct = 0
		for i in range(len(eval_data) // batch_instance_num + 1):

			batch_instances = eval_data[batch_instance_num * i : batch_instance_num * (i + 1)]

			eval_batch_token_ids, eval_batch_token_masks, \
				eval_batch_segment_ids, eval_batch_labels = get_batch_data(eval_data, i, batch_instance_num)

			# eval_batch_token_ids = eval_batch_token_ids.to(device)
			# eval_batch_token_masks = eval_batch_token_masks.to(device)
			# eval_batch_segment_ids = eval_batch_segment_ids.to(device)
			# # eval_batch_labels =  eval_batch_labels.to(device)

			outputs = model(input_ids=eval_batch_token_ids, attention_mask=eval_batch_token_masks, \
								token_type_ids=eval_batch_segment_ids)

			logits = outputs.logits
			logits = logits.reshape(-1, num_cands)

			probabilities = nn.Softmax(dim=1)(logits)		

			# predicted = logit[:, 1]
			# predicted = predicted.reshape(-1, num_cands)

			label_predicted = torch.argmax(probabilities, dim=1)

			for i ,label_pre in enumerate(label_predicted.tolist()):
				batch_instances[i].label_id = label_pre

			cor = (label_predicted == 0)

			total += len(cor)
			correct += cor.sum().item()

			instance_predict.extend(batch_instances)

		print("Eval on {}  instance_num: {}  acc: {}".format(field, total, correct/total))

		print("Predict instances num  :  ", len(instance_predict))

		# print("=== Writing predict data to files ===")
		# output_file = "%s/pretrain_%s_predict.pkl" % (out_file, field)
		# with open(output_file, 'wb') as f:
		# 	pickle.dump(instance_predict, f)
		# print("--- Write Success ---")

def main():
	field_num_epoch = {
						"coronation_street": 7,
						"muppets": 7,
						"elder_scrolls": 8,
						"ice_hockey": 6
						}

	for field in fields:
		train(field, field_num_epoch[field])


if __name__ == "__main__":
	main()


	

