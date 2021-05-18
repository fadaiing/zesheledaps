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
from create_field_training_data_side import TrainingInstance

import time


fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
mention_train_file = "./out_data/mention_mse"
mention_valid_file = "./out_data/mention_el_em_side"
model_file = "./out_data/model"


epochs = 10
BATCH_SIZE_TRAIN = 256
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

		
def train(field, train_data):
	
	train_instance_num = len(train_data)
	print("train_instance_num:", train_instance_num)

	batch_instance_num = BATCH_SIZE_TRAIN // num_cands

	print("=== Prepare eval datasets from {}===".format(mention_valid_file))
	with open("%s/%s.pkl" % (mention_valid_file, field),"rb") as f:
		field_data = pickle.load(f)

	print("eval instance nums : {}".format(len(field_data)))

	print("=== Prepare model and optimizer ===")
	device = torch.device(args.device)
	ckpt_file = "./out_data/model/lmpretrain_em_predict_pretrain_%s_21.ckpt" % field
	print("--- Load model from {} ---".format(ckpt_file))
	model = BertForSequenceClassification.from_pretrained(ckpt_file, num_labels=1, return_dict=True)
	# model.resize_token_embeddings(50525)
	print("Vocabulary size {}".format(model.config.vocab_size))

	# ckpt_file = model_file + "/" + "train_el_5.ckpt"
	# print("=== Load model from {} ===".format(ckpt_file))   
	# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, return_dict=True)
	# model.resize_token_embeddings(len(tokenizer))
	model = torch.nn.DataParallel(model)
	model.to(device)

	print("Model has %s parameters" % sum(p.numel() for p in model.parameters() if p.requires_grad))
	optimizer = Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
	

	rng = random.Random(random_seed)
	set_seed(random_seed)    # Added here for reproductibility
	for epoch in range(epochs):
		print("---epoch{}---".format(epoch))
		
		start_time = time.time() 
		# rng.shuffle(train_data)
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
			train_batch_labels = train_batch_labels.to(device)

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
		# / (train_instance_num // batch_instance_num)
		print("--- Epoch {} loss: {} ---".format(epoch, epoch_ave_loss))

		eval(model, field, field_data)
	# print("=== Save training model after {} training ===".format(epochs))
	# mode_save_file = model_file + "/train_el_{}.ckpt".format(epochs)
	# model.module.save_pretrained(mode_save_file)
	
	# return model.module


def eval(model, field, eval_data):

	print("eval instance nums : {}".format(len(eval_data)))

	batch_instance_num = BATCH_SIZE_TEST // num_cands
	print("batch_instance_num: ", batch_instance_num)

	# device = torch.device(args.device)
	# model = torch.nn.DataParallel(model)
	# model.to(device)

	model.eval( )
	with torch.no_grad():
		print("--- Eval field {} ---".format(field))

		total = 0
		correct = 0
		for i in range(len(eval_data) // batch_instance_num):

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

			cor = (label_predicted == 0)

			total += len(cor)
			correct += cor.sum().item()

		print("Eval on {}  instance_num: {}  acc: {}".format(field, total, correct/total))


def main():

	print("=== Prepare train datasets from {} ===".format(mention_train_file))
	with open("%s/train.pkl" % (mention_train_file), "rb") as f:
		train_data = pickle.load(f)

	for field in fields:
		train(field, train_data)

if __name__ == "__main__":
	main()
	

