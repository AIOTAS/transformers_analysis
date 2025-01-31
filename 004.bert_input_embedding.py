from transformers import BertTokenizer, BertModel
from transformers.models.bert import BertModel
import torch


model_name = "models/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


test_sentence = "this is a test sentence"

model_input = tokenizer(test_sentence, return_tensors="pt")

model.eval()


with torch.no_grad():
    outputs = model(**model_input)
