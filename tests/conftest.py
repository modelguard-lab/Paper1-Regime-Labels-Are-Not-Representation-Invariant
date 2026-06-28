import sys
from pathlib import Path

# Project root on sys.path so tests can `from src.core.x import ...`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
