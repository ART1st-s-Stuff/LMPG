import subprocess
import json
import os
from typing import List
import logging

from transformers import AutoModelForCausalLM

from utils import settings
from environment.internal_tools.self_sft import SelfSFT, SelectedSFTConfig
from utils.scoring import Scoreboard

DEFAULT_CONFIG = [
    "--job.config_file flame/flame/models/fla.toml",
    "--model.config configs/rwkv-6-340M.json",
    "--model.tokenizer_path RWKV/rwkv-6-world-3b-v2.1",
    "--optimizer.name AdamW",
    "--optimizer.eps 1e-15",
    "--optimizer.lr 3e-4",
    "--lr_scheduler.warmup_steps 1024",
    "--lr_scheduler.lr_min 0.1",
    "--lr_scheduler.decay_type cosine",
    "--training.batch_size 1",
    "--training.seq_len 2048",
    "--training.gradient_accumulation_steps 1",
    "--training.steps 20480",
    "--training.max_norm 1.0",
    "--training.skip_nan_inf",
    "--training.dataset_split train",
    "--training.num_workers 1",
    "--training.prefetch_factor 2",
    "--training.seed 42",
    "--training.compile",
    "--training.tensor_parallel_degree 1",
    "--training.disable_loss_parallel",
    "--checkpoint.interval 2048",
    "--checkpoint.load_step -1",
    "--metrics.log_freq 1"
]

class SelfSFT_FLA(SelfSFT):
    def __init__(self, model: AutoModelForCausalLM, config: List[str]):
        super().__init__(model)
        self.config = config

    @staticmethod
    def merge_args(config: List[str], *overrides: List[List[str]]) -> List[str]:
        args = {}
        for line in config:
            arg = line.split(" ", 2)
            if len(arg) < 2:
                args[arg[0]] = None
            else:
                args[arg[0]] = arg[1]
        for override in overrides:
            for line in override:
                arg = line.split(" ", 2)
                if len(arg) < 2:
                    args[arg[0]] = None
                else:
                    args[arg[0]] = arg[1]
        ret = []
        for k, v in args.items():
            ret.append(k)
            if v is not None:
                ret.append(v)
        return ret

    def train(self, dataset: dict, config: SelectedSFTConfig) -> None:
        with open(os.path.join(settings.WORK_DIR, "self-sft-data.json"), "w", encoding="utf-8") as f:
            json.dump(dataset, f)
        
        # Destroy current model to free memory
        del self.model

        # Train
        override_config = [f"--optimizer.lr {config['learning_rate']}"]
        config = self.merge_args(DEFAULT_CONFIG, self.config, override_config)
        try:
            subprocess.run([
                "flame/train.sh",
                "--training.data_files", os.path.join(settings.WORK_DIR, "self-sft-data.json"),
                "--training.dataset", "json",
                "--job.dump_folder", settings.MODEL_DIR,
                *config
            ], check=True)
        except subprocess.CalledProcessError as e:
            logging.error("Encountered error during self SFT. Output:")
            logging.error(e.stdout)
            logging.error("Error:")
            logging.error(e.stderr)
            raise e

        # Load new model
        self.model = AutoModelForCausalLM.from_pretrained(settings.MODEL_DIR, trust_remote_code=True).cuda()