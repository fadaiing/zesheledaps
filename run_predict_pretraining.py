# coding=utf-8
# The code refer to lajanugen's zeshel code

"""Training the pretraining model"""

import pickle
# from numpy.core.arrayprint import printoptions
import torch
import argparse
import numpy as np
from torch import tensor
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import Adam
import random
import json

from create_entity_predict_pretraining_data import MEinstance


# fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
fields = ["forgotten_realms", "lego", "star_trek", "yugioh"]
entity_file = "./out_data/entity_predict"
model_file = "./out_data/model"

# pre_train_filed = "elder_scrolls"
# document_filed = {
# 	"train" : ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling", 
# 				"starwars", "world_of_warcraft"],
# 	"val" : ["coronation_street", "muppets", "ice_hockey", "elder_scrolls"],
# 	"test" : ["forgotten_realms", "lego", "star_trek", "yugioh"]
# }

epochs = 20

max_seq_length = 256

lr_rate = 2e-5
weight_decay = 0.0

random_seed = 12345

MS = "[MS]"
MM = "[MM]"
ME = "[ME]"
ENT = "[ENT]"
ENTM = "[ENTM]"


parser = argparse.ArgumentParser()

parser.add_argument("--train_batch_size", type=int, default=42, help="Batch size for training")

parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device GPU or CPU')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
args = parser.parse_args()


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(random_seed)
# Initialize distributed training if needed
args.distributed = (args.local_rank != -1)
if args.distributed:
	torch.cuda.set_device(args.local_rank)
	args.device = torch.device("cuda", args.local_rank)
	torch.distributed.init_process_group(backend='nccl', init_method='env://')


class pretraining_dataset(Dataset):

	def __init__(self, train_data):
		train_data_tensor = []
		for ins in train_data:
			train_data_tensor.append([torch.tensor(ins.input_ids), torch.tensor(ins.input_mask),
										torch.tensor(ins.segment_ids),torch.tensor(ins.em_labels)])
		self.data = train_data_tensor

	def __getitem__(self, index):
		return self.data[index]
 
	def __len__(self):
		return len(self.data)


def get_data_loader(args, train_data):

	training_dataset = pretraining_dataset(train_data)

	train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset) if args.distributed else None

	train_loader = DataLoader(training_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))

	return train_loader, train_sampler


def train(field):

	print("=== Pretraining for {} ===".format(field))

	if dist.get_rank() not in [-1, 0]:
		dist.barrier()  # 先让主进程(rank==0)先执行，进行数据处理，预训模型参数下载等操作，然后保存cache
	print("=== Prepare train datasets ===")
	rng = random.Random(random_seed)

	with open("%s/%s_me.pkl" % (entity_file, field), "rb") as f:
		print("--- pretraing for {} ---".format(field))
		train_data = pickle.load(f)
	
	rng.shuffle(train_data)

	train_instance_num = len(train_data)
	print("Total {} training instances".format(train_instance_num))
	
	train_loader, train_sampler = get_data_loader(args, train_data)	

	print("=== Prepare model and optimizer ===")
	field_tokenizer_file = entity_file + "/" + "%s_tokenizer/" % field
	filed_tokenizer = BertTokenizer.from_pretrained(field_tokenizer_file)

	em_map_file = "%s/em_map.json" % (field_tokenizer_file)
	with open(em_map_file, 'r') as f:
		filed_em_map = json.load(f)

	vocab_size = len(filed_tokenizer) + len(filed_em_map)

	print("Adding {} entity tokens into vocab".format(len(filed_em_map)))
	print("Vocabulary size {}".format(vocab_size))

	pretrain_ckpt_file = "out_data/model/pretrain_tgt_%s_40.ckpt" % field
	#pretrain_ckpt_file = "bert-base-uncased"
	print("--- load model from {} ---".format(pretrain_ckpt_file))
	model = BertForMaskedLM.from_pretrained(pretrain_ckpt_file, return_dict=True).to(args.device)
	model.resize_token_embeddings(vocab_size)
	print("Model has %s parameters" % sum(p.numel() for p in model.parameters() if p.requires_grad))

	if dist.get_rank() == 0:
		dist.barrier() # 主进程执行完后，其余进程开始读取cache
	
	# Prepare model for distributed training if needed
	if args.distributed:
		model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
	optimizer = Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

	
	for epoch in range(epochs):
		print("---epoch{}---".format(epoch))
		## 保证每次的sampler是一样的
		train_loader.sampler.set_epoch(epoch)

		model.train()
		total_loss = 0.0

		for data in train_loader:

			train_batch_token_ids, train_batch_input_masks, train_batch_segment_ids, train_batch_labels = data

			train_batch_token_ids = train_batch_token_ids.to(args.device)
			train_batch_input_masks = train_batch_input_masks.to(args.device)
			train_batch_segment_ids = train_batch_segment_ids.to(args.device)
			train_batch_labels =  train_batch_labels.to(args.device)
			## print("input_shape{}".format(args.local_rank), train_batch_token_ids.shape, train_batch_labels.shape)
			
			outputs = model(input_ids=train_batch_token_ids, attention_mask=train_batch_input_masks, \
								token_type_ids=train_batch_segment_ids, labels=train_batch_labels)
			
			loss = outputs.loss
			
			total_loss += loss.item()
	
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		epoch_ave_loss = total_loss 
		print("---epoch {} loss: {} ---".format(epoch, epoch_ave_loss))
	
		if torch.distributed.get_rank() == 0 and epoch % 5 == 0:
			print("=== Save pretraining model after {} training ===".format(epoch+1))
			model_save_file = model_file + "/lmpretrain_em_predict_pretrain_{}_{}.ckpt".format(field, epoch+1)
			model_to_save = model.module if hasattr(model, "module") else model
			model_to_save.save_pretrained(model_save_file)



if __name__ == "__main__":
	
	for field in fields[3:]:
		train(field)















































