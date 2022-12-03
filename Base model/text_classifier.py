# this part for old model, using unknow data

# import ktrain
# def load_predictor(model_path):
#     return ktrain.load_predictor(model_path)

# def text_to_emotion(text):
#     predictor = load_predictor('./predictor')
#     labels = ['happiness', 'sadness', 'fear', 'anger', 'neutral']
#     predictions = predictor.predict(text,return_proba=True)
#     return dict(zip(labels,predictions))


# torch with goemotion
# import matplotlib.pyplot as plt
# import numpy as np
# from transformers import AutoConfig, AutoModel, AutoTokenizer
# import torch


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# class BERTClass(torch.nn.Module):
#     def __init__(self):
#         super(BERTClass, self).__init__()
#         self.roberta = AutoModel.from_pretrained('roberta-base')
# #         self.l2 = torch.nn.Dropout(0.3)
#         self.fc = torch.nn.Linear(768,5)
    
#     def forward(self, ids, mask, token_type_ids):
#         _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
# #         output_2 = self.l2(output_1)
#         output = self.fc(features)
#         return output

# model = BERTClass()
# model.to(device)

# model.load_state_dict(torch.load('./model/bert_model.bin', map_location=device))

# model.eval()

# def predict_text_emotion(text):
#     tokenizer = AutoTokenizer.from_pretrained('roberta-base')



#     inputs = tokenizer.encode_plus(
#         text,
#         truncation=True,
#         add_special_tokens=True,
#         max_length=200,
#         padding='max_length',
#         return_token_type_ids=True
#     )


#     pred_id = inputs['input_ids']

#     pred_mask = inputs['attention_mask']

#     pred_token_type_id =inputs["token_type_ids"]

#     pred_id= torch.tensor(pred_id, dtype=torch.long)
#     pred_mask= torch.tensor(pred_mask, dtype=torch.long)
#     pred_token_type_id= torch.tensor(pred_token_type_id, dtype=torch.long)

#     pred_mask=torch.unsqueeze(pred_mask, 0)
#     pred_id=torch.unsqueeze(pred_id, 0)
#     pred_token_type_id=torch.unsqueeze(pred_token_type_id, 0)

#     pred_id=pred_id.to(device, dtype = torch.long)
#     pred_mask=pred_mask.to(device, dtype = torch.long)
#     pred_token_type_id =pred_token_type_id.to(device, dtype = torch.long)

#     prediction = model(pred_id, pred_mask, pred_token_type_id)

#     return prediction
# while(True):
#     text = input('input text: ')
#     import time
#     start = time.time()
#     print(predict_text_emotion(text))
#     print(time.time()-start)

# fast bert

from fast_bert.prediction import BertClassificationPredictor

# ekman_label.csv renamed to labels.csv

MODEL_PATH = './model/model_out'
LABEL_PATH = './model/'
predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=False,
				# model_type='xlnet',
				do_lower_case=False,
				device='cpu') # set custom torch.device, defaults to cuda if available

# Single prediction
single_prediction = predictor.predict("just get me result for this text")

# Batch predictions
texts = [
	"this is the first text",
	"this is the second text"
	]

multiple_predictions = predictor.predict_batch(texts)

import time
start = time.time()
print(predictor.predict("just get me result for this text"))
print(time.time()-start)