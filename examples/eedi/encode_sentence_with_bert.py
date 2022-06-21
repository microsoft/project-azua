import sys

import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

sys.path.insert(0, "research_experiments/eedi/")
from sentence_to_embedding import transform_sentence_csv_to_list

sentences = transform_sentence_csv_to_list()

model_id = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertModel.from_pretrained(model_id)

encoded = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
outputs = model(**encoded)
bert_output = outputs[0]

bert_output = list()
for sentence in tqdm(sentences):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    print(last_hidden_states.shape)
    bert_output.append(last_hidden_states)
