from models.rwkv.rwkv_official import RWKVSFTAgent, RWKVMixin, RWKVSelfSFT
from tasks.retrieval.retrieval import build_tasks
from utils import settings

def pre_tool_call_hook(agent: RWKVSFTAgent, model_output: str, context: str, tool: str, tool_input: str):
    tool_input = tool_input.replace("'", '"')
    return (context, tool, tool_input)

if __name__ == "__main__":
    tasks = build_tasks(20)
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
                ENABLE_THINK=settings.ENABLE_THINKING,
                TEMPERATURE=1.0,
                TOP_P=0.3,
                ALPHA_FREQUENCY=0.5,
                ALPHA_PRESENCE=0.5,
                ALPHA_DECAY=0.996,
                TOKEN_STOP=[0]
            ),
            SFT_CONFIG=RWKVSelfSFT.Config(
                DATA_PATH="./agent/rwkv/self-sft-data.json"
            )
        ))
        #agent.add_post_step_hook(lambda model_output: f"You are now in tool call context {context}. You can use the following tools: {tool}. The input is {tool_input}.")
        agent.add_pre_tool_call_hook(pre_tool_call_hook)
        task.run(agent)