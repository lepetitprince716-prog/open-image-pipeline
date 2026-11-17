#!/usr/bin/env python3
"""Official Danbooru TagDB CLI (print-first). Lookup, validate, fetch+pin."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loramake.tags import main

if __name__ == "__main__":
    raise SystemExit(main())
