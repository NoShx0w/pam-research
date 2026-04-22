from __future__ import annotations

import shutil
from pathlib import Path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mirror_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Cannot mirror missing file: {src}")
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)


def mirror_optional_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)
    return True
