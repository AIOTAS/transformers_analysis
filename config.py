import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_NAME = "models/distilbert-base-uncased-finetuned-sst-2-english"
