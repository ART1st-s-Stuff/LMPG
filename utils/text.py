from typing import Callable, Generic, Sequence, TypeVar

import torch

class Text:
    SEGMENT_LENGTH = 80      # 80 tokens per window

    def __init__(self, tokenizer: Callable[[str], torch.Tensor], text: str, internal: bool = False):
        self.tokenizer = tokenizer
        self.text = self.tokenizer(text)
        self.internal = internal
        self.reset()
        
    def interface(self):
        return [
            self.read,
            self.reset,
            self.go_to_segment
        ]
    
    def read(self) -> torch.Tensor:
        ret = self.text[self.index:self.index + self.SEGMENT_LENGTH]
        self.index += self.SEGMENT_LENGTH
        current_segment = self.index // self.SEGMENT_LENGTH + 1
        if current_segment != self.segment:
            self.segment = current_segment
            ret = torch.cat([self.tokenizer(f'<|SEGMENT {self.segment}|>'), ret, self.tokenizer(f'</|SEGMENT {self.segment}|>')])
        return ret
    
    def reset(self):
        self.index = 0
        self.segment = 1

    def go_to_segment(self, segment_number: int) -> torch.Tensor:
        self.index = segment_number * self.SEGMENT_LENGTH
        return self.read()

    def embed_roles(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.internal:
            dim_role = torch.zeros(tokens.size(0))
        else:
            dim_role = torch.ones(tokens.size(0))
        return torch.stack([tokens, dim_role], dim=-1)
    
def text_window(text: str) -> Text[str]:
    ...