"""Repository entry point. Never add the legacy project to sys.path."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from optical_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
