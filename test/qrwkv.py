from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "recursal/QRWKV7-7B-Instruct" # This model

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # or torch.float16 if needed
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, device_map="auto")

# Prepare input
text = "The quick brown fox jumps over the lazy"
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

# Generate text
generated_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=False, # Set to True for sampling, adjust temperature/top_p
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id, # Or set to a specific pad_token_id if available
)

# Decode output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Input: {text} Generated: {output_text}")