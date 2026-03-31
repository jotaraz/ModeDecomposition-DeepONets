"""
Pull large data directories from a HuggingFace dataset repository.

Usage:
    python hf_pull.py <hf_repo_id> [--token <hf_token>] [--subset nets|datasets|sb_data]

Examples:
    # Pull everything
    python hf_pull.py username/deeponet-data

    # Pull only the PDE datasets (fast, ~280 MB total)
    python hf_pull.py username/deeponet-data --subset datasets sb_data

    # Pull only trained nets (slow, ~20 GB)
    python hf_pull.py username/deeponet-data --subset nets

Files are placed under data/ in the repository root, matching the original layout:
    data/nets/
    data/datasets/
    data/sb_data/
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

AVAILABLE_SUBSETS = ["nets", "datasets", "sb_data"]


def pull(repo_id: str, token: str | None, subsets: list[str]) -> None:
    # Build glob patterns that select only the requested subdirectories.
    allow_patterns = [f"data/{s}/**" for s in subsets]
    print(f"Downloading from: {repo_id}")
    print(f"Subsets:  {subsets}")
    print(f"Patterns: {allow_patterns}")
    print(f"Target:   {REPO_ROOT}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=allow_patterns,
            local_dir=REPO_ROOT,
            local_dir_use_symlinks=False,  # copy files, don't symlink
        )
    except HfHubHTTPError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify
    print("\nVerifying downloaded directories:")
    for s in subsets:
        target = os.path.join(REPO_ROOT, "data", s)
        if os.path.isdir(target):
            n = sum(len(files) for _, _, files in os.walk(target))
            print(f"  data/{s}/  — {n} files")
        else:
            print(f"  data/{s}/  — NOT FOUND (check repo contents)")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull large data directories from a HuggingFace dataset repo."
    )
    parser.add_argument(
        "repo_id",
        help="HuggingFace repo ID in the form username/repo-name",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (defaults to $HF_TOKEN env var)",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=AVAILABLE_SUBSETS,
        default=AVAILABLE_SUBSETS,
        metavar="SUBSET",
        help=(
            f"Which subsets to download. Choices: {AVAILABLE_SUBSETS}. "
            "Defaults to all three."
        ),
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "Warning: no HF token provided. "
            "Set HF_TOKEN or pass --token. "
            "Private repos will be inaccessible.",
            file=sys.stderr,
        )

    pull(args.repo_id, args.token, args.subset)


if __name__ == "__main__":
    main()
