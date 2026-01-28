import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from utils.shell import LocalShellEnvironment

environment = LocalShellEnvironment(cwd="tasks/business/matrix/data")
print(environment.execute("cat 1cb3d727-edf7-49ef-88a2-d994a767b979.xml", cwd=""))