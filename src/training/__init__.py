import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent.resolve().as_posix()
sys.path.append(root_path)
