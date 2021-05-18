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

mention_train_file = "./out_data/mention_mse"
mention_valid_file = "./out_data/mention_mse"
model_file = "./out_data/model"



epochs = 10
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 2048
num_cands = 64
max_seq_length = 256

lr_rate = 2e-5
weight_decay = 0.0

random_seed = 12345 

MS = "[MS]"
ME = "[ME]"
ENT = "[ENT]"

parser = argparse.ArgumentParser()

parser.add_argument("--num_cands", type=int, default=64, help="num of candidate entity")
parser.add_argument("--max_seq_length", type=int, default=256, help="max length of the sequence")


parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device GPU or CPU')

parser.add_argument("--max_norm", type=float, default=0.25, help="Clipping gradient norm")
parser.add_argument("--random_seed", type=int, default=12345, help="random_seed")

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

		
def train():

	print("=== Prepare train datasets from {} ===".format(mention_train_file))
	with open("%s/train.pkl" % (mention_train_file), "rb") as f:
		train_data = pickle.load(f)
	
	train_instance_num = len(train_data)
	print("train_instance_num:", train_instance_num)

	batch_instance_num = BATCH_SIZE_TRAIN // num_cands

	print("=== Prepare eval datasets from {} ===".format(mention_valid_file))
	with open("%s/coronation_street.pkl" % (mention_valid_file),"rb") as f:
		coronation_street = pickle.load(f)
	with open("%s/elder_scrolls.pkl" % (mention_valid_file),"rb") as f:
		elder_scrolls = pickle.load(f)
	with open("%s/ice_hockey.pkl" % (mention_valid_file),"rb") as f:
		ice_hockey = pickle.load(f)
	with open("%s/muppets.pkl" % (mention_valid_file),"rb") as f:
		muppets = pickle.load(f)


	print("=== Prepare model and optimizer ===")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tokenizer.add_tokens(list([MS, ME, ENT]))

	device = torch.device(args.device)
	# ckpt_file = model_file + "/" + "train_el_5.ckpt"
	# print("=== Load model from {} ===".format(ckpt_file))   
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, return_dict=True)
	model.resize_token_embeddings(len(tokenizer))

	print("Vocabulary size {}".format(model.config.vocab_size))
	
	model = torch.nn.DataParallel(model)
	model.to(device)

	print("Model has %s parameters" % sum(p.numel() for p in model.parameters() if p.requires_grad))
	optimizer = Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
	
	# eval_data = {"coronation_street" : coronation_street,
	# 			"elder_scrolls" : elder_scrolls, 
	# 			"ice_hockey" : ice_hockey, 
	# 			"muppets" : muppets
	# 		}
	# eval_instance_num = {d : len(eval_data[d]) for d in eval_data}
	# print("eval instance nums : {}, {}, {}, {}".format(*list(eval_instance_num.values())))
	
	# eval(model, coronation_street, elder_scrolls, ice_hockey, muppets)

	# exit()

	rng = random.Random(random_seed)
	set_seed(random_seed)    	# Added here for reproductibility
	for epoch in range(epochs):
		print("---   epoch{}   ---".format(epoch))
		
		start_time = time.time() 
		rng.shuffle(train_data)
		epoch_train_data = rng.sample(train_data, train_instance_num // 10)
		epoch_train_instance_num = len(epoch_train_data)
		print("epoch {} sample train_instance_num:".format(epoch), epoch_train_instance_num)

		model.train()
		total_loss = 0.0
		for i in range(epoch_train_instance_num // batch_instance_num + 1):

			train_batch_token_ids, train_batch_token_masks, \
			train_batch_segment_ids, train_batch_labels = get_batch_data(epoch_train_data, i, batch_instance_num)

			# train_batch_token_ids = train_batch_token_ids.to(device)
			# train_batch_token_masks = train_batch_token_masks.to(device)
			# train_batch_segment_ids = train_batch_segment_ids.to(device)
			train_batch_labels =  train_batch_labels.to(device)

			outputs = model(input_ids=train_batch_token_ids, attention_mask=train_batch_token_masks, \
								token_type_ids=train_batch_segment_ids, labels=None)

			logits = outputs.logits
			logits = logits.reshape(-1, num_cands)

			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, train_batch_labels)

			total_loss += loss.item()
			
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			optimizer.step()
			optimizer.zero_grad()
		
		end_time = time.time()
		print('Epoch time cost', end_time-start_time,'s')

		epoch_ave_loss = total_loss 

		print("--- Epoch {} loss: {} ---".format(epoch, epoch_ave_loss))


		eval(model, coronation_street, elder_scrolls, ice_hockey, muppets)

	print("=== Save training model after {} training ===".format(epochs))
	mode_save_file = model_file + "/train_el_{}.ckpt".format(epochs)
	model.module.save_pretrained(mode_save_file)
	
	return model.module


def eval(model, coronation_street, elder_scrolls, ice_hockey, muppets):

	eval_data = {"coronation_street" : coronation_street,
					"elder_scrolls" : elder_scrolls, 
					"ice_hockey" : ice_hockey, 
					"muppets" : muppets
				}
	eval_instance_num = {d : len(eval_data[d]) for d in eval_data}

	print("eval instance nums : {}, {}, {}, {}".format(*list(eval_instance_num.values())))

	batch_instance_num = BATCH_SIZE_TEST // num_cands
	print("batch_instance_num: ", batch_instance_num)

	# device = torch.device(args.device)
	# model = torch.nn.DataParallel(model)
	# model.to(device)

	model.eval( )
	with torch.no_grad():
		for data_filed in eval_data:
			print("--- Eval field {} ---".format(data_filed))
			data = eval_data[data_filed]

			total = 0
			correct = 0
			for i in range(len(data) // batch_instance_num):

				eval_batch_token_ids, eval_batch_token_masks, \
					eval_batch_segment_ids, eval_batch_labels = get_batch_data(data, i, batch_instance_num)

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

				cor = (label_predicted == 0)

				total += len(cor)
				correct += cor.sum().item()

			print("Eval on {}  instance_num: {}  acc: {}".format(data_filed, total, correct/total))


if __name__ == "__main__":
	train()

	

  
















