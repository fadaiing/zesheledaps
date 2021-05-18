import pickle 
import json
from create_training_data import TrainingInstance



# fields = ["coronation_street", "muppets", "elder_scrolls", "ice_hockey"]
fields = ["forgotten_realms", "lego", "star_trek", "yugioh"]
mention_label_file = "./out_data/mention_mse"
mention_em_unlabel_file = "./out_data/mention_mse"



for field in fields:
	# with open(mention_label_file + "/" + "%s_predict.pkl" % field, "rb") as f1:
	# 	label_data = pickle.load(f1)

	with open(mention_label_file + "/" + "%s_pre.json" % field, "r") as f1:
		label_dict = json.load(f1)


	# print(len(label_dict))

	with open(mention_em_unlabel_file + "/" + "%s.pkl" % field, "rb") as f2:
		unlabel_data = pickle.load(f2)

	print(len(unlabel_data))

	# label_dict = {}

	# for i in label_data:
	# 	label_dict[i.mention_guid] = i.label_id

	for i in unlabel_data:
		i.label_id = label_dict[i.mention_guid]

	cor = 0
	for i in unlabel_data:
		if i.label_id.index(max(i.label_id)) == 0:
			cor += 1

	print("acc : ", cor/len(unlabel_data))

	print("=== Writing predict data to files ===")
	output_file = "%s/pretrain_%s_predict.pkl" % (mention_em_unlabel_file, field)
	with open(output_file, 'wb') as f:
		pickle.dump(unlabel_data, f)
	print("--- Write Success ---")

	
