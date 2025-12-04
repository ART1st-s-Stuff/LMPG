from transformers import AutoModelForCausalLM, AutoTokenizer
import os

WORK_DIR = "./agent"
MODEL_DIR = os.path.join(WORK_DIR, 'model')

#MODEL = AutoModelForCausalLM.from_pretrained('fla-hub/rwkv7-1.5B-g1', trust_remote_code=True).cuda()
#TOKENIZER = AutoTokenizer.from_pretrained('fla-hub/rwkv7-1.5B-g1', trust_remote_code=True)
TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)
#MODEL = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True).cuda()
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.3,
    "repetition_penalty": 1.2
}
AUTO_SFT_CONFIG = {
    "learning_rate": 3e-4
}
# MODEL_SPECIFIC_SFT_CONFIG = {
#     "eos_token": "\n\n"
# }
MODEL_SPECIFIC_SFT_CONFIG = [
    #f"--model.config {MODEL_DIR}/config.json",
    "--model.config fla-hub/rwkv7-1.5B-g1",
    "--model.tokenizer_path fla-hub/rwkv7-1.5B-g1",
    "--checkpoint.load_step 0",
]

# Text
TEXT_WINDOW_SEGMENT_LENGTH = 1000

# Agent
INVALID_TOOL_CALL_PENALTY = -100.0
OUTPUT_LENGTH_PENALTY = -100.0
MAX_OUTPUT_LENGTH = 500
STRUCTURIZED_TOOL_INPUT_FORMAT = "json"
TELL_REWARD_AFTER_EACH_ROUND = True
ENABLE_THINKING = True