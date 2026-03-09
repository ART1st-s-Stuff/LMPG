import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from debug.trace import start_server

if __name__ == "__main__":
    trace_thread = start_server()
    trace_thread.join()