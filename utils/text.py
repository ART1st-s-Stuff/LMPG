from typing import Callable, Generic, Sequence, TypeVar
        
T = TypeVar('T')

class Text(Generic[T]):
    WINDOW_LENGTH = 80      # 80 tokens per window
    WINDOW_PER_PAGE = 10    # 10 windows per page

    def __init__(self, tokenizer: Callable[[str], Sequence[T]], text: str):
        self.tokenizer = tokenizer
        self.text = self.tokenizer(text)
        self.reset()
        
    def interface(self):
        return [
            self.read,
            self.reset,
            self.go_to_page
        ]
    
    def read(self) -> Sequence[T]:
        ret = self.text[self.index:self.index + self.WINDOW_LENGTH]
        self.index += self.WINDOW_LENGTH
        current_page = self.index // (self.WINDOW_LENGTH * self.WINDOW_PER_PAGE) + 1
        if current_page != self.page:
            self.page = current_page
            ret = self.tokenizer('<Page %d>' % self.page) + ret
        return ret
    
    def reset(self):
        self.index = 0
        self.page = 1

    def go_to_page(self, page_number: int) -> Sequence[T]:
        self.index = page_number * self.WINDOW_LENGTH * self.WINDOW_PER_PAGE
        return self.read()
    
def text_window(text: str) -> Text[str]:
    ...