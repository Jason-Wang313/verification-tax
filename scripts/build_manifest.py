"""Scan models/ for GGUF files and emit a MANIFEST.csv that the sweep driver reads."""

from __future__ import annotations

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST = MODELS_DIR / "MANIFEST.csv"


def slug(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-(q|iq)\d+(_\w+)*$", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def main() -> int:
    rows: list[tuple[str, str]] = []
    for path in sorted(MODELS_DIR.glob("*.gguf")):
        rel = path.relative_to(PROJECT_ROOT).as_posix()  # forward slashes for bash/sh
        rows.append((slug(path.stem), rel))
    # Explicit LF-only line endings so bash read-loops don't carry \r
    # on Windows (CRLF would break `-f` file tests).
    with MANIFEST.open("wb") as fh:
        fh.write(("".join([f"{slug_},{path}\n" for slug_, path in rows])).encode("utf-8"))
    print(f"wrote {len(rows)} entries to {MANIFEST}")
    for slug_, path in rows:
        print(f"  {slug_:<50s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
