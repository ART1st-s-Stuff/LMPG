from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, TypedDict, List

import tiktoken

from .tool import Toolset

ContextType = TypeVar('ContextType')
class AbstractContextManager(ABC, Generic[ContextType]):
    """This class is for managing context."""
    @abstractmethod
    def add(self, item: ContextType):
        ...
    
    @abstractmethod
    def get(self) -> List[ContextType]:
        ...
        
class DefaultContextType(TypedDict):
    role: str
    content: str
    
class DefaultContextManager(AbstractContextManager[DefaultContextType], List[DefaultContextType]): 
    def add(self, item: DefaultContextType):
        self.append(item)
    
# 树状 topic 的路径连接符，用于展示
TOPIC_PATH_SEP = " > "

class SlidingWindowContextManager(DefaultContextManager):
    """Use a sliding window for context. Topic is stored as a tree path (list of strings)."""
    class SetTopicTool(Toolset):
        def __init__(self, context_manager: 'SlidingWindowContextManager'):
            self.context_manager = context_manager

        @Toolset.structurized_tool()
        def set_topic(self, topic: str):
            """设置当前主题（替换整条主题路径）。会清空之前的层级，只保留这一层。
            适合在切换大阶段时使用。

            Args:
                topic, str: 要设置的主题，简洁即可。
            """
            self.context_manager.set_topic(topic)
            return "Topic is set."

        @Toolset.structurized_tool()
        def push_topic(self, subtopic: str):
            """在当前主题路径下压入一层子主题（树状进入下一层）。
            例如当前为「任务A」时调用 push_topic("步骤2")，则路径变为「任务A > 步骤2」。

            Args:
                subtopic, str: 子主题名称，简洁即可。
            """
            self.context_manager.push_topic(subtopic)
            return "Subtopic pushed."

        @Toolset.structurized_tool()
        def pop_topic(self):
            """从当前主题路径弹出一层，回到上一层主题（树状回退）。
            若当前只有一层则不会改变。
            """
            self.context_manager.pop_topic()
            return "Topic popped to parent."

    def __init__(self, topic: DefaultContextType, max_context_length: int):
        self.max_context_length = max_context_length
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        # 树状路径：从根到当前层的字符串列表
        raw = (topic.get("content") or "").strip()
        self._topic_path: List[str] = [raw] if raw else []
        # 保证 self[0] 存在且为 system topic；内容由 _sync_topic_content 统一维护
        self.append({"role": topic.get("role", "system"), "content": TOPIC_PATH_SEP.join(self._topic_path)})
        self.topic_tool = self.SetTopicTool(self)

    @property
    def _current_context_length(self) -> int:
        return sum(len(self.tokenizer.encode(item['content'])) for item in self)
    
    def add(self, item: DefaultContextType):
        if self._current_context_length + len(self.tokenizer.encode(item['content'])) > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Topic length exceeds the context length of {self.max_context_length}.")
            self.pop(1)
        self.append(item)

    def _sync_topic_content(self):
        """把 _topic_path 写回 self[0]['content']。"""
        self[0]["content"] = TOPIC_PATH_SEP.join(self._topic_path)

    def set_topic(self, topic: str | DefaultContextType):
        """设置整条主题路径。若传入 str 则路径变为 [topic]；若传入 DefaultContextType 则兼容旧接口，取 content 为单层路径。"""
        if isinstance(topic, str):
            self._topic_path = [topic] if topic.strip() else []
        else:
            raw = (topic.get("content") or "").strip()
            self._topic_path = [raw] if raw else []
        self._sync_topic_content()
        if self._current_context_length > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Topic length exceeds the context length of {self.max_context_length}.")
            self.pop(1)

    def push_topic(self, subtopic: str):
        """在当前路径下压入一层子主题。"""
        if subtopic.strip():
            self._topic_path.append(subtopic.strip())
            self._sync_topic_content()
        if self._current_context_length > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Topic length exceeds the context length of {self.max_context_length}.")
            self.pop(1)

    def pop_topic(self):
        """弹出一层主题，回到上一层。若只有一层则不变。"""
        if len(self._topic_path) > 1:
            self._topic_path.pop()
            self._sync_topic_content()

    def clear(self):
        topic = self[0]
        super().clear()
        self.append(topic)