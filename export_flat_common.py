from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIBLING_OMNI_SEARCH_ROOT = PROJECT_ROOT.parent / "omni_search"
MODELS_ROOT = (
    SIBLING_OMNI_SEARCH_ROOT / "models"
    if SIBLING_OMNI_SEARCH_ROOT.is_dir()
    else PROJECT_ROOT / "models"
)


def prepare_output_dir(output_dir: Path, force: bool) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"output already exists: {output_dir}\n"
                "rerun with --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def copy_required_file(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def copy_optional_files(source_dir: Path, output_dir: Path, filenames: list[str]) -> None:
    for name in filenames:
        source = source_dir / name
        if source.is_file():
            copy_required_file(source, output_dir / name)
