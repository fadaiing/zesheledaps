# coding=utf-8
# The code refer to lajanugen's zeshel code

"""Training the pretraining model"""

import pickle
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.optim import Adam
import random

from create_pretraining_data import TrainingInstance



document_file = "./out_data/document"
model_file = "./out_data/model"

pre_train_filed = "test"
document_filed = {
	"train" : ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling", 
				"starwars", "world_of_warcraft"],
	"val" : ["coronation_street", "muppets", "ice_hockey", "elder_scrolls"],
	"test" : ["forgotten_realms", "lego", "star_trek", "yugioh"]
}

epochs = 50

max_seq_length = 256
mlm_probability = 0.15

lr_rate = 2e-5
weight_decay = 0.0

random_seed = 12345

parser = argparse.ArgumentParser()

parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training")

parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device GPU or CPU')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

args = parser.parse_args()

# Initialize distributed training if needed
args.distributed = (args.local_rank != -1)
if args.distributed:
	torch.cuda.set_device(args.local_rank)
	args.device = torch.device("cuda", args.local_rank)
	torch.distributed.init_process_group(backend='nccl', init_method='env://')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(random_seed)

# def get_batch_data(batch_instances):

# 	batch_token_ids = []
# 	batch_segment_ids = []


# 	for instance in batch_instances:
# 		instance_token_ids = instance["input_ids"]
# 		assert len(instance_token_ids) == max_seq_length
# 		batch_token_ids.append(instance_token_ids)

# 		instance_segment_ids = instance["segment_ids"]
# 		assert len(instance_segment_ids) == max_seq_length
# 		batch_segment_ids.append(instance_segment_ids)

# 	return torch.LongTensor(batch_token_ids), torch.LongTensor(batch_segment_ids)


class pretraining_dataset(Dataset):

	def __init__(self, train_data):
		train_data_tensor = []
		for ins in train_data:
			train_data_tensor.append([torch.tensor(ins.input_ids), torch.tensor(ins.segment_ids)])
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

	if dist.get_rank() not in [-1, 0]:
		dist.barrier()  # 先让主进程(rank==0)先执行，进行数据处理，预训模型参数下载等操作，然后保存cache
	
	print("=== Prepare pretraining datasets for {}===".format(field))
	rng = random.Random(random_seed)

	with open("%s/%s.pkl" % (document_file, field), "rb") as f:
		train_data = pickle.load(f)

	rng.shuffle(train_data)

	train_instance_num = len(train_data)
	print("Total {} training instances".format(train_instance_num))
	
	train_loader, train_sampler = get_data_loader(args, train_data)	

	print("===Prepare model and optimizer===")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# pretrain_ckpt_file = model_file + "/" + "pretrain_tgt_{}_20.ckpt".format(field)
	# print("--- load model from {} ---".format(pretrain_ckpt_file))
	#configuration = BertConfig()
	#print(configuration)
	model = BertForMaskedLM.from_pretrained("bert-base-uncased", return_dict=True).to(args.device)
	#model = BertForMaskedLM(configuration, return_dict=True).to(args.device)
	print("Model has %s parameters" % sum(p.numel() for p in model.parameters() if p.requires_grad))

	if dist.get_rank() == 0:
		dist.barrier() # 主进程执行完后，其余进程开始读取cache

	
	
	# Prepare model for distributed training if needed
	if args.distributed:
		model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
	optimizer = Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

	def mask_tokens(inputs):
		"""
		Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
		"""
		labels = inputs.clone()

		# sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
		probability_matrix = torch.full(labels.shape, mlm_probability)
		special_tokens_mask = [
			tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
			]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		# if tokenizer._pad_token is not None:
		# 	padding_mask = labels.eq(tokenizer.pad_token_id)
		# 	probability_matrix.masked_fill_(padding_mask, value=0.0)
		masked_indices = torch.bernoulli(probability_matrix).bool()
		labels[~masked_indices] = -100  # only compute loss on masked tokens

		# 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

		# 10% of the time, replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged
		return inputs, labels

	
	for epoch in range(epochs):
		print("---epoch{}---".format(epoch))
		## 保证每次的sampler是一样的
		train_loader.sampler.set_epoch(epoch)

		model.train()
		total_loss = 0.0

		for data in train_loader:
	
			train_batch_token_ids, train_batch_segment_ids = data
			train_batch_token_ids, train_batch_labels = mask_tokens(train_batch_token_ids)

			# print(train_batch_token_ids[0])
			# print(train_batch_labels[0])

			train_batch_token_ids = train_batch_token_ids.to(args.device)
			train_batch_segment_ids = train_batch_segment_ids.to(args.device)
			# batch_word_ids = batch_word_ids.to(device)
			train_batch_labels =  train_batch_labels.to(args.device)
			## print("input_shape{}".format(args.local_rank), train_batch_token_ids.shape, train_batch_labels.shape)
			
			outputs = model(input_ids=train_batch_token_ids,  \
								token_type_ids=train_batch_segment_ids, labels=train_batch_labels)
			
			loss = outputs.loss
			total_loss += loss.item()
			
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		epoch_ave_loss = total_loss 
		print("---epoch {} loss: {} ---".format(epoch, epoch_ave_loss))

		if (epoch+1) % 5 == 0 and torch.distributed.get_rank() == 0:
			print("=== Save pretraining model after {} training ===".format(epoch+1))
			model_save_file = model_file + "/pretrain_tgt_{}_{}.ckpt".format(field, epoch+1+20)
			model_to_save = model.module if hasattr(model, "module") else model
			model_to_save.save_pretrained(model_save_file)
	

if __name__ == "__main__":
	
	for field in document_filed[pre_train_filed]:
		train(field)













































