from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GenerationConfig
from textwrap import dedent

model_name = "recursal/QRWKV7-7B-Instruct" # This model

HINT = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call. The tool call must be in the following
    format:
        <tool>{ "context": "context name", "tool": "tool name", "args": (optional, in json format) }</tool>
    You may also choose not to use any tools. All your output other than the tool call
    will be considered as your thinking process.

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following operations:
    - View a window: <tool>{ "context": "window name", "tool": "read" }</tool>
    - Go to a specific segment: <tool>{ "context": "window name", "tool": "goto", "args": { "segment_number": int } }</tool>

    In this stage, you have access to the following windows:
    - text-default-hint: This hint.
    - window_list: List of all the windows.
    
    You also have access to another tools:
    - end: End the task.

    Your task:
    Step 1. Open the "text-default-hint" window to read the hint again.
    Step 2. Use the end tool to end the task.
    """
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # or torch.float16 if needed
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# Prepare input
chat = [{"role": "user", "content": HINT}]
text = tokenizer.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
print(text)
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

# Generate text
generated_ids = model.generate(
    input_ids,
    attention_mask=torch.ones_like(input_ids).to(model.device),
    max_new_tokens=1024,
    do_sample=True, # Set to True for sampling, adjust temperature/top_p
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id, # Or set to a specific pad_token_id if available
    generation_config=GenerationConfig(
        temperature=0.0,
    )
)

# Decode output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Input: {text} \nGenerated: {output_text}")