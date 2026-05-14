"""Reproducibility helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def md5_of_paths(paths: Iterable[Path]) -> str:
    h = hashlib.md5()
    for p in sorted(Path(x) for x in paths):
        h.update(p.read_bytes())
    return h.hexdigest()
