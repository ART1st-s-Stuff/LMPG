import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from utils.environment import Stop

stop = Stop()

stop.invoke("ctx", "stop", {}, None)