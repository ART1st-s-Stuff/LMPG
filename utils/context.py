from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, TypedDict, List

import tiktoken

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
    def __init__(self, initial_prompt: DefaultContextType, max_context_length: int):
        self.max_context_length = max_context_length
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        self.append(initial_prompt)

    @property
    def _current_context_length(self) -> int:
        return sum(len(self.tokenizer.encode(item['content'])) for item in self)
    
    def add(self, item: DefaultContextType):
        if self._current_context_length + len(self.tokenizer.encode(item['content'])) > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Initial prompt length exceeds the context length of {self.max_context_length}.")
            self.pop(1)
        self.append(item)

    def set_initial_prompt(self, initial_prompt: DefaultContextType):
        self[0] = initial_prompt
        if self._current_context_length > self.max_context_length:
            if len(self) <= 1:
                raise ValueError(f"Initial prompt length exceeds the context length of {self.max_context_length}.")
            self.pop(1)

    def clear(self):
        initial_prompt = self[0]
        super().clear()
        self.append(initial_prompt)