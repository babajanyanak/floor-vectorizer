"""Thin launcher so the script can be called as `python vectorize_floor.py`."""
from vectorize_floor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
