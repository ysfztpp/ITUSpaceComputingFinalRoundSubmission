from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "c03": {
        "checkpoint": ROOT / "checkpoints" / "c03_full_data_model.pt",
        "config": ROOT / "configs" / "submission_c03.json",
    },
    "tsvit": {
        "checkpoint": ROOT / "checkpoints" / "c29_tsvit_model.pt",
        "config": ROOT / "configs" / "submission_tsvit.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the GitLab submission repo using one of the organized finalists.")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--submission-repo", type=Path, required=True)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = CANDIDATES[args.candidate]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_submission.py"),
        "--trained-model",
        str(candidate["checkpoint"]),
        "--submission-config",
        str(candidate["config"]),
        "--submission-repo",
        str(args.submission_repo),
    ]
    if args.skip_validate:
        command.append("--skip-validate")
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
