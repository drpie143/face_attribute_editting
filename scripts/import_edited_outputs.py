#!/usr/bin/env python
"""
Import edited outputs for one model from a zip/folder into the local run tree.

Examples:
    python scripts/import_edited_outputs.py --source sdxl_edited_outputs.zip --model-id sdxl
    python scripts/import_edited_outputs.py --source exported_run/edited/sdxl --model-id sdxl
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, TASKS


def find_model_edited_dir(source_root: Path, model_id: str) -> Path:
    """Find a folder that contains task subfolders for the requested model."""
    source_root = source_root.resolve()
    required_any = set(TASKS)

    candidates = [
        source_root,
        source_root / model_id,
        source_root / "edited" / model_id,
        source_root / "runs" / PATHS["run_dir"].name / "edited" / model_id,
        source_root / "face_attr_edit_v2_full_inpaint" / "edited" / model_id,
        source_root / "face-attr-edit" / "runs" / PATHS["run_dir"].name / "edited" / model_id,
    ]

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        child_names = {p.name for p in candidate.iterdir() if p.is_dir()}
        if child_names & required_any:
            return candidate

    matches = []
    for path in source_root.rglob(model_id):
        if not path.is_dir():
            continue
        child_names = {p.name for p in path.iterdir() if p.is_dir()}
        if child_names & required_any:
            matches.append(path)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = "\n".join(f"  - {p}" for p in matches[:10])
        raise RuntimeError(f"Found multiple possible {model_id} folders:\n{joined}")

    raise FileNotFoundError(
        f"Could not find an edited/{model_id} folder under {source_root}."
    )


def copy_tree_contents(src: Path, dst: Path, replace: bool = False) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            if target.exists() and not replace:
                continue
            shutil.copy2(item, target)


def validate_import(model_dir: Path) -> None:
    print(f"\nImported folder: {model_dir}")
    total_meta = 0
    for task in TASKS:
        task_dir = model_dir / task
        meta_count = len(list(task_dir.glob("*_meta.json"))) if task_dir.exists() else 0
        best_count = len(list(task_dir.glob("*_best.*"))) if task_dir.exists() else 0
        total_meta += meta_count
        status = "OK" if meta_count > 0 else "MISSING"
        print(f"  {task:15s} meta={meta_count:3d} best={best_count:3d}  {status}")

    if total_meta == 0:
        raise RuntimeError("Import finished, but no *_meta.json files were found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import edited model outputs into runs/<run_tag>/edited/<model_id>."
    )
    parser.add_argument("--source", required=True, help="Zip file or folder to import.")
    parser.add_argument("--model-id", default="sdxl", help="Model id, usually sdxl.")
    parser.add_argument("--run-dir", default=None, help="Target run directory.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing files with the same name.",
    )
    parser.add_argument(
        "--replace-model-dir",
        action="store_true",
        help="Move the existing target model folder to a timestamped backup before import.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    run_dir = Path(args.run_dir) if args.run_dir else PATHS["run_dir"]
    target_model_dir = run_dir / "edited" / args.model_id

    with tempfile.TemporaryDirectory(prefix="face_attr_import_") as tmp:
        if source.is_file() and source.suffix.lower() == ".zip":
            extract_dir = Path(tmp) / "zip"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {source} ...")
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(extract_dir)
            import_root = extract_dir
        else:
            import_root = source

        source_model_dir = find_model_edited_dir(import_root, args.model_id)
        print(f"Found source model folder: {source_model_dir}")
        print(f"Copying to: {target_model_dir}")
        if source_model_dir.resolve() == target_model_dir.resolve():
            print("Source is already the target folder; validating in place.")
        else:
            if args.replace_model_dir and target_model_dir.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_root = run_dir / "edited_backups"
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_dir = backup_root / f"{target_model_dir.name}_backup_{stamp}"
                print(f"Backing up existing target to: {backup_dir}")
                shutil.move(str(target_model_dir), str(backup_dir))
            copy_tree_contents(source_model_dir, target_model_dir, replace=args.replace)

    validate_import(target_model_dir)
    print("\nNext:")
    print("  python scripts/run_evaluation.py")


if __name__ == "__main__":
    main()
