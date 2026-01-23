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
    
class SlidingWindowContextManager(DefaultContextManager):
    """Use a sliding window for context"""
    class SetTopicTool(Toolset):
        def __init__(self, context_manager: 'SlidingWindowContextManager'):
            self.context_manager = context_manager

        @Toolset.structurized_tool()
        def set_topic(self, topic: str):
            """Set the topic of the context. This will be presented in the beginning of your context.
            You can update the topic when needed, for example during different stages of the task.
            The topic should be brief and concise.

            Args:
                topic, str: The topic to set.
            """
            self.context_manager.set_topic(topic)
            return "Topic is set."

    def __init__(self, topic: DefaultContextType, max_context_length: int):
        self.max_context_length = max_context_length
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        self.append(topic)
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

    def set_topic(self, topic: DefaultContextType):
        self[0] = topic
        if self._current_context_length > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Topic length exceeds the context length of {self.max_context_length}.")
            self.pop(1)

    def clear(self):
        topic = self[0]
        super().clear()
        self.append(topic)