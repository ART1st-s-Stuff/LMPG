import unittest

from trl import SFTTrainer, SFTConfig
from models.rwkv.rwkv import SelfSFT_FLA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
#WORK_DIR = "./agent"
#MODEL_DIR = os.path.join(WORK_DIR, 'model')

class TestSFTTrainer(unittest.TestCase):
    def test_sft_trainer(self):
        # model = AutoModelForCausalLM.from_pretrained('fla-hub/rwkv7-1.5B-g1', trust_remote_code=True).cuda()
        # tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-1.5B-g1', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained('fla-hub/rwkv6-1.6B-finch', trust_remote_code=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv6-1.6B-finch', trust_remote_code=True)
        print("Model loaded")
        trainer = SFTTrainer(model, SFTConfig(), train_dataset=Dataset.from_list([{"text": "Hello, world!"}]), processing_class=tokenizer,
            peft_config=LoraConfig(init_lora_weights="pissa", r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        print("Trainer created")
        trainer.train()
        print("Training completed")
        # trainer = SelfSFT_FLA(model, utils.settings.MODEL_SPECIFIC_SFT_CONFIG)
        # trainer.train([{"text": "Hello, world!"}], {"learning_rate": 3e-4})

        
if __name__ == "__main__":
    unittest.main()