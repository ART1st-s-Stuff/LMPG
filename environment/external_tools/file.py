from typing import List, Dict, Optional
import difflib
import os

from utils.tool import Toolset
from utils.text import TextWindow, text_window
from utils.scoring import Scoreboard

class FileIO(Toolset):
    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir
        self.opened_files : Dict[str, TextWindow] = {}

    @Toolset.structurized_tool()
    def create_file(self, filename: str, _scoreboard: Scoreboard) -> str:
        """Creates an empty file with the given filename.
        
        Args:
            filename, str: The name of the file to create.
        """
        filepath = os.path.join(self.working_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('')
                return f"Created file {filename}."
        except Exception as e:
            return f"Failed to create file {filename} due to {e}"
    
    @Toolset.structurized_tool()
    def open_file(self, filename: str, _scoreboard: Scoreboard) -> TextWindow | str:
        """Open a window of the file.
        
        Args:
            filename, str: The name of the file to view.
        """
        if filename in self.opened_files:
            return f"File {filename} is already opened."
        filepath = os.path.join(self.working_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return text_window(content, filename, 'file')
        except Exception as e:
            return f"Failed to open file {filename} due to {e}"

    @Toolset.structurized_tool()
    def close_file(self, filename: str, _scoreboard: Scoreboard) -> str:
        """Close the opened file window.
        
        Args:
            filename, str: The name of the file to close.
        """
        if filename not in self.opened_files:
            return f"File {filename} is not opened."
        del self.opened_files[filename]
        return f"Closed file {filename}."

    def apply_patch(self, filename: str, patch: str, _scoreboard: Scoreboard) -> str:
        """Modify the file by applying a patch. The file will be saved immediately.
        
        Args:
            filename, str: The name of the file to modify.
            patch, str: The patch to apply to the file.
        """
        filepath = os.path.join(self.working_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()
                diff_lines = patch.split('\n')
                patched = difflib.restore(content, diff_lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(patched)
            return f"Saved patch to {filename}."
        except Exception as e:
            return f"Failed to apply patch to {filename} due to {e}"

    def delete_file(self, filename: str, _scoreboard: Scoreboard) -> str:
        """Delete a file. Will close the file window if it is opened.
        
        Args:
            filename, str: The name of the file to delete.
        """
        try:
            if filename in self.opened_files:
                self.close_file(filename, scoreboard)
            filepath = os.path.join(self.working_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                return f"Deleted file {filename}."
            else:
                return f"File {filename} does not exist."
        except Exception as e:
            return f"Failed to delete file {filename} due to {e}"

    def list_files(self, dir: str, _scoreboard: Scoreboard) -> TextWindow:
        """List all files in the working directory.

        Args:
            dir, str: The directory to list files from.
        """
        return text_window(os.listdir(os.path.join(self.working_dir, dir)), 'files', 'file')