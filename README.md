# # Zero-shot Entity Linking

The data is available [here](https://drive.google.com/file/d/1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO/view?usp=sharing). 


**1. Create EL sentence pair for source domain or target domain** 

python create_training_data.py

**2 .Fine-tune the original pretrained BERT on source label data**

python run_training.py

**3. Perform DAP for pretrained BERT**

python create_pretraining_data.py
python run_pretraining.py

**4. Fine-tune the DAP pretrained BERT on source label data**

python run_filed_training.py

**5.  generate peseudo label for target domain samples**

python run_field_training_label.py

**6. Perform E-DAP after DAP for pretrained BERT**

python create_entity_predict_pretraining_data.py
python run_predict_pretraining.py

**7. Create entity-enhanced EL sentence pair for target domain**

python create_field_training_data_side.py

**8. Perform self-training for the E-DAP pretrained BERT on target pseudo data 
     Fine-tune the E-DAPS pretrained BERT on source label data**

python run_self_training_hard.py
python run_self_training_soft.py









