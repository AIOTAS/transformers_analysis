from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def tokenizer_demo():
    test_sentences = ["today is not that bad", "today is so bad"]
    tokenizer(test_sentences[0], padding=True, return_tensors="pt")


if __name__ == "__main__":
    tokenizer_demo()
