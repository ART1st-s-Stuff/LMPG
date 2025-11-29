import os
os.environ["RWKV_V7_ON"] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import subprocess
import shlex
import json
import logging
import re

from utils.agent import StateManagerMixin, OutputLengthPenaltyMixin, SFTAgent
from utils.environment import Environment
from environment.internal_tools.self_sft import SelfSFT, SelectedSFTConfig
from utils import settings
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from utils.tool import InvalidToolCallJSONException

logging.basicConfig(level=logging.DEBUG)

RWKV_CHAT_SEPARATOR = "\n\n"

class PIPELINE_MODIFIED(PIPELINE):
    # TODO: calculate perplexity
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        for i in range(token_count):

            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
                
            for n in args.token_ban:
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= args.alpha_decay
            
            ttt = self.decode([token])
            www = 1
            if ttt in ' \t0123456789':
                www = 0
            # elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
            #     www = 0.5
            if token not in occurrence:
                occurrence[token] = www
            else:
                occurrence[token] += www
            # print(occurrence) # debug
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1

            if out_str.endswith(RWKV_CHAT_SEPARATOR):
                break
        return out_str, state

class RWKVMixin(StateManagerMixin):
    model: RWKV
    pipeline: PIPELINE
    history_state: Optional[list]
    history_chat: str
    rwkv_config: 'Config'
    
    @dataclass
    class Config:
        TEMPERATURE: float = 1.0
        TOP_P: float = 0.3
        TOP_K: int = 0
        ALPHA_FREQUENCY: float = 0.1
        ALPHA_PRESENCE: float = 0.1
        ALPHA_DECAY: float = 0.996
        TOKEN_BAN: List = field(default_factory=list)
        TOKEN_STOP: List = field(default_factory=lambda: [0])
        CHUNK_LEN: int = 256
        MAX_TOKENS: int = 1024
        ENABLE_THINK: bool = False
    
    def __init__(self, model: RWKV, rwkv_config: Config, **kwargs):
        """
        初始化 RWKV 模型
        
        Args:
            model_path: RWKV 模型文件路径
            config: 配置对象
        """
        super().__init__(**kwargs)
        self.rwkv_config = rwkv_config
        self.history_str = ""
        self.history_state = None
        self.pipeline_args = PIPELINE_ARGS(
            temperature=self.rwkv_config.TEMPERATURE,
            top_p=self.rwkv_config.TOP_P,
            top_k=self.rwkv_config.TOP_K,
            alpha_frequency=self.rwkv_config.ALPHA_FREQUENCY,
            alpha_presence=self.rwkv_config.ALPHA_PRESENCE,
            alpha_decay=self.rwkv_config.ALPHA_DECAY,
            token_ban=self.rwkv_config.TOKEN_BAN,
            token_stop=self.rwkv_config.TOKEN_STOP,
            chunk_len=self.rwkv_config.CHUNK_LEN
        )
        # 初始化 RWKV 模型
        self.update_model(model)

        
    def _to_chat_format(self, input: str | Dict[str, str]) -> str:
        if isinstance(input, dict):
            # 如果是字典格式，包含 ai, environment, reward 等字段
            output1 = None
            output2 = None
            if "environment" in input:
                environment = input["environment"].replace(RWKV_CHAT_SEPARATOR, "\n")
                output1 = f"User: {environment}"
            if "reward" in input:
                reward = input["reward"].replace(RWKV_CHAT_SEPARATOR, "\n")
                output2 = f"User: {reward}"
            if output1 is not None and output2 is not None:
                ret = f"{output1}{RWKV_CHAT_SEPARATOR}{output2}"
            elif output1 is not None:
                ret = output1
            elif output2 is not None:
                ret = output2
            else:
                ret = ""
        else:
            # 如果是纯字符串，作为用户输入
            input = input.replace(RWKV_CHAT_SEPARATOR, "\n")
            ret = f"User: {input}"
        if self.rwkv_config.ENABLE_THINK:
            ret += f"{RWKV_CHAT_SEPARATOR}Assistant: <think"
        else:
            ret += f"{RWKV_CHAT_SEPARATOR}Assistant: "
        return ret
    
    def _forward(self, input: str | Dict[str, str]) -> str:
        """
        执行前向传播生成文本
        
        Args:
            input: 输入文本或字典
            
        Returns:
            生成的文本
        """
        # 转换输入格式
        formatted_input = self._to_chat_format(input)
        logging.debug(f"==========================================")
        logging.debug(f"Stepping with input:\n{formatted_input}")
        
        # 如果有历史状态，使用历史状态
        state = self.history_state
        
        # 生成文本
        output_text, state = self.pipeline.generate(
            formatted_input,
            token_count=self.rwkv_config.MAX_TOKENS,
            args=self.pipeline_args,
            callback=None,
            state=state
        )
        
        # 更新历史状态
        self.history_state = state
        self.history_str += formatted_input + output_text
        
        logging.debug(f"==========================================")
        logging.debug(f"Generated output:\n{output_text}")
        return output_text
    
    def clear_state(self):
        """
        清除模型的历史状态
        """
        self.history_state = None
        logging.debug("Cleared RWKV history state")
        
    def update_model(self, model: RWKV):
        self.model = model
        self.pipeline = PIPELINE_MODIFIED(self.model, "rwkv_vocab_v20230424")

class RWKVSelfSFT(SelfSFT[RWKV]):
    @dataclass
    class Config:
        DATA_PATH: str = "agent/rwkv/self-sft-data.json"
        N_LAYER: int = 24
        N_EMBD: int = 2048
        CTX_LEN: int = 8192
        MICRO_BSZ: int = 8
        EPOCH_STEPS: int = 200
        EPOCH_SAVE: int = 1
        EPOCH_COUNT: int = 10
        ACCELERATOR: str = "gpu"
        PRECISION: str = "fp16"
        DEVICES: int = 1
        STRATEGY: str = "deepspeed_stage_1"
        GRAD_CP: int = 1
        PEFT_CONFIG: str = '{"r":8}'

    def __init__(self, model_dir: str, model_strategy: str, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.model_dir = model_dir
        self.model_strategy = model_strategy
        self.model = RWKV(model=self.get_latest_weight_pth(self.model_dir), strategy=self.model_strategy)

    @staticmethod
    def get_latest_weight_pth(dir: str) -> str:
        # Find the latest weight filename
        weight_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".pth")]
        return max(weight_files, key=os.path.getctime).rsplit(".", 1)[0]

    def train(self, dataset: list[dict], config: SelectedSFTConfig) -> None:
        with open(self.config.DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f)

        del self.model

        # Train
        try:
            version = "x070" if os.environ["RWKV_V7_ON"] else "x060"
            subprocess.run(shlex.split(f"""python train.py --load_model {os.path.abspath(self.get_latest_weight_pth(self.model_dir))} \
                --proj_dir {os.path.abspath(self.model_dir)} --data_file {os.path.abspath(self.config.DATA_PATH)} \
                --vocab_size 65536 \
                --data_type binidx \
                --n_layer {self.config.N_LAYER} --n_embd {self.config.N_EMBD} \
                --ctx_len {self.config.CTX_LEN} --micro_bsz {self.config.MICRO_BSZ} \
                --epoch_steps {self.config.EPOCH_STEPS} --epoch_count {self.config.EPOCH_COUNT} --epoch_save {self.config.EPOCH_SAVE} \
                --lr_init {config["learning_rate"]} --lr_final {config["learning_rate"]} \
                --accelerator {self.config.ACCELERATOR} --precision {self.config.PRECISION} \
                --devices {self.config.DEVICES} --strategy {self.config.STRATEGY} --grad_cp {self.config.GRAD_CP} \
                --my_testing {version} \
                --peft_config {self.config.PEFT_CONFIG}
            """), cwd="models/rwkv/RWKV-PEFT", check=True)
        except subprocess.CalledProcessError as e:
            logging.error("Encountered error during self SFT. Output:")
            logging.error(e.stdout)
            logging.error("Error:")
            logging.error(e.stderr)
            raise e

        self.model = RWKV(model=self.get_latest_weight_pth(self.model_dir), strategy=self.model_strategy)

    def get_model(self) -> RWKV:
        return self.model

    def _parse_llm_output(self, output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
        regex = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
        extracted = regex.findall(output)
        if len(extracted) == 0:
            return None, None, None
        try:
            tool_call_json = json.loads(extracted[0])
            assert tool_call_json["name"] == "tool", "Invalid tool call JSON. Expected tool call name to be 'tool'."
            return tool_call_json["parameters"]["context"], tool_call_json["parameters"]["tool"], tool_call_json["parameters"].get("args", {})
        except Exception as e:
            raise InvalidToolCallJSONException()

class RWKVSFTAgent(RWKVMixin, OutputLengthPenaltyMixin, SFTAgent[RWKV]):
    @dataclass
    class Config(SFTAgent.Config):
        MODEL_PATH: str = "agent/rwkv"
        MODEL_STRATEGY: str = "cuda fp16"
        RWKV_CONFIG: RWKVMixin.Config = field(default_factory=RWKVMixin.Config)
        SFT_CONFIG: RWKVSelfSFT.Config = field(default_factory=RWKVSelfSFT.Config)
    
    def __init__(self, environment: Environment, config: Config):
        sft_trainer = RWKVSelfSFT(config.MODEL_PATH, config.MODEL_STRATEGY, config.SFT_CONFIG)
        super().__init__(
            model=sft_trainer.get_model(),
            rwkv_config=config.RWKV_CONFIG,
            environment=environment,
            sft_trainer=sft_trainer,
            config=config,
            output_length_penalty=settings.OUTPUT_LENGTH_PENALTY,
            max_output_length=settings.MAX_OUTPUT_LENGTH,
        )

    def _calculate_output_length(self, model_output: str) -> int:
        return len(self.pipeline.encode(model_output))