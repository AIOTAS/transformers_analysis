import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/bert-base-chinese")

sentences = ["你好，世界！！！", "欢迎来北京.", "你是谁？"]

inputs = tokenizer(sentences, padding=True, return_tensors="pt")

decoder_x = inputs["input_ids"]


print(decoder_x)
print(decoder_x.shape)  # (3 , 10)

print(tokenizer.special_tokens_map)

PAD_ID = tokenizer.convert_tokens_to_ids("[PAD]")

decoder_first_attn_mask = (
    (decoder_x == PAD_ID)
    .unsqueeze(1)
    .expand(decoder_x.shape[0], decoder_x.shape[1], decoder_x.shape[1])
)  # (3 , 10) => (3 , 1 , 10)


print(decoder_first_attn_mask)

decoder_first_attn_mask = decoder_first_attn_mask | torch.triu(
    torch.ones(decoder_x.shape[1], decoder_x.shape[1]), diagonal=1
).unsqueeze(0).bool().expand(
    decoder_x.shape[0], -1, -1
)  # (10 , 10) => (1, 10 , 10) => (3 , 10 , 10)

print(decoder_first_attn_mask)
