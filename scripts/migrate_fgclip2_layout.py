from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import project_layout


KNOWN_SOURCE_MODELS = [
    "fg-clip2-base",
    "chinese-clip-vit-base-patch16",
    "qwen3-vl-embedding-2b",
]
LEGACY_FGCLIP2_ROOT = PROJECT_ROOT / ".onnx-wrapper-test"


@dataclass(frozen=True)
class MoveOp:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or apply migration from legacy model/output paths to the new "
            "models/source and artifacts/fgclip2 layout."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Perform moves instead of printing the plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ops = planned_moves()
    if not ops:
        print("No legacy files need migration.")
        return

    print("Planned moves:")
    for op in ops:
        print(f"  {op.source} -> {op.target}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to move files.")
        return

    for op in ops:
        op.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(op.source), str(op.target))
    remove_empty_dirs(project_layout.LEGACY_FGCLIP2_ROOT)
    for model_name in KNOWN_SOURCE_MODELS:
        remove_empty_dirs(project_layout.LEGACY_SOURCE_MODELS_ROOT / model_name)
    print(f"Moved {len(ops)} files.")


def planned_moves() -> list[MoveOp]:
    ops: list[MoveOp] = []
    ops.extend(source_model_moves())
    ops.extend(legacy_fgclip2_moves())
    return ops


def source_model_moves() -> list[MoveOp]:
    ops: list[MoveOp] = []
    for model_name in KNOWN_SOURCE_MODELS:
        legacy_dir = project_layout.LEGACY_SOURCE_MODELS_ROOT / model_name
        preferred_dir = project_layout.PREFERRED_SOURCE_MODELS_ROOT / model_name
        if not legacy_dir.exists() or preferred_dir.exists():
            continue
        for source in iter_files(legacy_dir):
            target = preferred_dir / source.relative_to(legacy_dir)
            if target.exists():
                continue
            ops.append(MoveOp(source, target))
    return ops


def legacy_fgclip2_moves() -> list[MoveOp]:
    layout = project_layout.FGCLIP2_LAYOUT
    if not LEGACY_FGCLIP2_ROOT.exists():
        return []

    ops: list[MoveOp] = []
    for source in iter_files(LEGACY_FGCLIP2_ROOT):
        rel = source.relative_to(LEGACY_FGCLIP2_ROOT)
        target = map_legacy_fgclip2_file(rel, layout)
        if target.exists():
            continue
        ops.append(MoveOp(source, target))
    return ops


def map_legacy_fgclip2_file(rel: Path, layout: project_layout.FgClip2Layout) -> Path:
    parts = rel.parts
    if not parts:
        raise ValueError("empty relative path")

    head = parts[0]
    if head == "assets":
        return layout.runtime_assets_dir.joinpath(*parts[1:])
    if head == "split":
        if rel.name == "split_text_embedding_report.json":
            return layout.split_report
        return layout.runtime_split_dir.joinpath(*parts[1:])
    if head == "quantized":
        if rel.name == "dynamic_int8_report.json":
            return layout.quant_report
        return layout.runtime_quantized_dir.joinpath(*parts[1:])
    if head == "fixtures":
        return layout.fixtures_root.joinpath(*parts[1:])
    if len(parts) == 1 and rel.suffix == ".onnx":
        return layout.runtime_root / rel.name
    return layout.diagnostics_root.joinpath(*parts)


def iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue


if __name__ == "__main__":
    main()
