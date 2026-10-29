
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

question = "What is TCS? Explain in simple English."

inputs = tokenizer(
    question,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Answer:", answer)
