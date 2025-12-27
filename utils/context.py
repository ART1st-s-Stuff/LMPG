from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, TypedDict, List

ContextType = TypeVar('ContextType')
class AbstractContextManager(ABC, Generic[ContextType]):
    """This class is for managing context."""
    @abstractmethod
    def add(self, item: ContextType):
        ...
    
    @abstractmethod
    def get(self) -> ContextType:
        ...
        
class DefaultContextType(TypedDict):
    role: str
    content: str
    
class DefaultContextManager(AbstractContextManager[DefaultContextType]):
    def __init__(self):
        self.history = []
        
    def add(self, item: DefaultContextType):
        self.history = 
    
    def get(self, item: )
    
class SlidingWindowContextManager(DefaultContextManager):
    """Use a sliding window for context"""
    history : List[DefaultContextType]
    
    def __init__(self, initial_prompt: DefaultContextType, max_context_length: int):
        self.initial_prompt = initial_prompt
        self.max_context_length = max_context_length
        self.history = []
    
    def add(self, item: DefaultContextType):
        self.history