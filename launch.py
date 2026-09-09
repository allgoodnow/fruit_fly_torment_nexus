"""Development and frozen application entry point."""
from multiprocessing import freeze_support
from pathlib import Path
import sys

if __name__ == "__main__":
    freeze_support()
    if not getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from nexus.app import main
    raise SystemExit(main())
