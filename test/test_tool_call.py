from models.rwkv.rwkv_official import RWKVSFTAgent, RWKVMixin, RWKVSelfSFT
from tasks.tool.tool_training import build_one_task
from utils import settings
import json

if __name__ == "__main__":
    tasks = [build_one_task() for _ in range(100)]
    for task in tasks:
        agent = RWKVSFTAgent(task, RWKVSFTAgent.Config(
            TELL_REWARD_AFTER_EACH_ROUND=True,
            AUTO_SFT=False,
            AUTO_SFT_CONFIG={
                "learning_rate": 1e-6
            },
            MODEL_PATH="./agent/rwkv",
            MODEL_STRATEGY="cuda bf16",
            RWKV_CONFIG=RWKVMixin.Config(
                ENABLE_THINK=settings.ENABLE_THINKING
            ),
            SFT_CONFIG=RWKVSelfSFT.Config(
                DATA_PATH="./agent/rwkv/self-sft-data.json"
            )
        ))
        #agent.add_post_step_hook(lambda model_output: f"You are now in tool call context {context}. You can use the following tools: {tool}. The input is {tool_input}.")
        task.run(agent)
        if agent.scoreboard_manager.get_current_score() > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print("Agent successfully completed the task. Training the agent...")
            agent.sft_trainer.train([{"text": agent.history_chat}], {"learning_rate": 3e-5})
        else:
            print("Agent failed to complete the task. Skipping training.")