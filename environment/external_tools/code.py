from typing import List
import os

from environment.tool import Tool, text_window, Text

class VirtualIDE(Tool):
    def __init__(self, scoreboard, working_dir: str):
        super().__init__(scoreboard)
        self.working_dir = working_dir

    def interface(self):
        return [
            self.create_file,
            self.read_file,
            self.apply_patch,
            self.delete_file,
            self.list_files
        ]

    def create_file(self, filename: str):
        """Creates an empty file with the given filename."""
        filepath = os.path.join(self.working_dir, filename)
        with open(filepath, 'w') as f:
            f.write('')
        
    def read_file(self, filename: str) -> Text:
        """Read file content in a window."""
        filepath = os.path.join(self.working_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        return text_window(content)

    def apply_patch(self, filename: str, patch: str):
        """Modify the file by applying a patch."""
        filepath = os.path.join(self.working_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        
        
        
        with open(filepath, 'w') as f:
            f.write(patched_content)

    def delete_file(self, filename: str):
        filepath = os.path.join(self.working_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
    def list_files(self) -> List[str]:
        return os.listdir(self.working_dir)

    def