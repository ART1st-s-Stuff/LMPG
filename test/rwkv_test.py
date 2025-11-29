import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textwrap import dedent
from models.rwkv.rwkv_official import RWKVMixin, RWKV

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

def main():
    # 配置模型路径
    model_path = "agent/rwkv/rwkv7-g1a4-2.9b-20251118-ctx8192"  # 根据实际情况修改
    model_strategy = "cuda fp16"  # 或 "cuda fp32", "cpu fp32" 等
    
    print("正在加载 RWKV 模型...")
    
    # 创建 RWKV 模型实例
    model = RWKV(model=model_path, strategy=model_strategy)
    
    # 创建配置
    rwkv_config = RWKVMixin.Config(
        TEMPERATURE=1.0,
        TOP_P=0.3,
        ALPHA_FREQUENCY=0.5,
        ALPHA_PRESENCE=0.5,
        ALPHA_DECAY=0.996,
        TOKEN_BAN=[],
        TOKEN_STOP=[0],
        CHUNK_LEN=256,
        MAX_TOKENS=1024,
        ENABLE_THINK=False
    )
    
    # 创建 RWKVMixin 实例用于测试
    class TestRWKV(RWKVMixin):
        def __init__(self, model: RWKV, rwkv_config: RWKVMixin.Config):
            super().__init__(model=model, rwkv_config=rwkv_config)
    
    test_agent = TestRWKV(model=model, rwkv_config=rwkv_config)
    
    print("模型加载完成！")
    print("=" * 60)
    print("输入提示:")
    print(HINT)
    print("=" * 60)
    
    # 生成文本
    print("\n正在生成回复...")
    output = test_agent._forward(HINT)
    
    print("=" * 60)
    print("生成的回复:")
    print(output)
    print("=" * 60)
    
    # 测试对话历史
    print("\n测试多轮对话...")
    second_input = "Please proceed with step 1."
    print(f"第二轮输入: {second_input}")
    output2 = test_agent._forward(second_input)
    print("=" * 60)
    print("第二轮回复:")
    print(output2)
    print("=" * 60)
    
    # 清除状态测试
    print("\n清除历史状态...")
    test_agent.clear_state()
    print("状态已清除")
    
    # 测试清除状态后的生成
    print("\n清除状态后重新生成...")
    output3 = test_agent._forward("Hello, who are you?")
    print("=" * 60)
    print("清除状态后的回复:")
    print(output3)
    print("=" * 60)

if __name__ == "__main__":
    main()

