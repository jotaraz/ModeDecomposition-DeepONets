"""
Push large data directories to a HuggingFace dataset repository.

Usage:
    python hf_push.py <hf_repo_id> [--token <hf_token>]

Example:
    python hf_push.py username/deeponet-data
    HF_TOKEN=hf_xxx python hf_push.py username/deeponet-data

The HF_TOKEN environment variable is used if --token is not provided.
The repo is created automatically if it does not exist.

Directories uploaded (relative to repo root):
    data/nets/      -> data/nets/      (~20 GB, model checkpoints)
    data/datasets/  -> data/datasets/  (~229 MB, PDE solution data)
    data/sb_data/   -> data/sb_data/   (~49 MB, spectral bias data)
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# (local path relative to repo root, path inside HF repo)
DIRS_TO_UPLOAD = [
    ("data/datasets", "data/datasets"),
    ("data/sb_data",  "data/sb_data"),
    ("data/nets",     "data/nets"),     # largest — uploaded last
]


def push(repo_id: str, token: str | None) -> None:
    api = HfApi(token=token)

    print(f"Creating/verifying repo: {repo_id}")
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    except HfHubHTTPError as e:
        print(f"Error creating repo: {e}", file=sys.stderr)
        sys.exit(1)

    for local_rel, repo_path in DIRS_TO_UPLOAD:
        local_abs = os.path.join(REPO_ROOT, local_rel)
        if not os.path.isdir(local_abs):
            print(f"  Skipping {local_rel!r} — directory not found.")
            continue

        n_files = sum(len(files) for _, _, files in os.walk(local_abs))
        print(f"\nUploading {local_rel!r}  ({n_files} files) -> {repo_path!r}")

        # upload_large_folder handles resumption and parallel uploads well for
        # directories with many files; falls back gracefully on older hub versions.
        try:
            api.upload_large_folder(
                folder_path=local_abs,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
        except AttributeError:
            # huggingface_hub < 0.22 — fall back to upload_folder with multi-commit
            print("  (upload_large_folder unavailable, falling back to upload_folder)")
            api.upload_folder(
                folder_path=local_abs,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                multi_commits=True,
                multi_commits_verbose=True,
            )

        print(f"  Done: {local_rel!r}")

    print("\nAll uploads complete.")
    print(f"Repo URL: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push large data directories to a HuggingFace dataset repo."
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
    args = parser.parse_args()

    if not args.token:
        print(
            "Warning: no HF token provided. "
            "Set HF_TOKEN or pass --token. "
            "This will fail for private repos or write operations.",
            file=sys.stderr,
        )

    push(args.repo_id, args.token)


if __name__ == "__main__":
    main()
