#!/usr/bin/env python
"""Preload or verify fine-tuned subject models locally.

Usage examples:
  python scripts/preload_models.py --dry-run
  python scripts/preload_models.py --download
  python scripts/preload_models.py --download --model Fawl/is469_project_physics

Options:
  --dry-run     Only show status & estimated size (default if no flag given)
  --download    Actually trigger download for all (or selected) models
  --model       Restrict to a single model id (can repeat)
  --token       Explicit HF token (else uses HUGGINGFACE_HUB_TOKEN env)

The script uses snapshot_download from huggingface_hub which populates
~/.cache/huggingface/hub (or HF_HOME if set). Repeated runs won't re-download
already cached artifacts unless revision changes.
"""
import os
import argparse
from typing import List
from huggingface_hub import HfApi, snapshot_download

DEFAULT_MODELS = [
    "Fawl/is469_project_physics",
    "Fawl/is469_project_chem",
    "Fawl/is469_project_bio",
]

def get_token(explicit: str | None) -> str | None:
    return explicit or os.getenv("HUGGINGFACE_HUB_TOKEN")

def model_status(api: HfApi, model_id: str, token: str | None):
    try:
        info = api.model_info(model_id, token=token)
        size = 0
        # Sum LFS file sizes if present
        for sibling in info.siblings:
            if sibling.size is not None:
                size += sibling.size
        return {
            "model_id": model_id,
            "exists_remote": True,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "estimated_size_mb": round(size / (1024 * 1024), 2) if size else None,
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "exists_remote": False,
            "error": str(e)
        }

def ensure_download(model_id: str, token: str | None) -> str:
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            token=token,
            local_files_only=False,
            tqdm_class=None,
        )
        return local_dir
    except Exception as e:
        raise RuntimeError(f"Download failed for {model_id}: {e}")


def already_cached(model_id: str) -> bool:
    # Heuristic: check for directory fragments in HF cache
    hf_home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # Snapshot dirs named like models--<org>--<name>
    prefix = f"models--{model_id.replace('/', '--')}"
    if not os.path.isdir(hf_home):
        return False
    for entry in os.listdir(hf_home):
        if entry.startswith(prefix):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only show status; do not download")
    parser.add_argument("--download", action="store_true", help="Download models (overrides dry-run)")
    parser.add_argument("--model", action="append", help="Specific model id (repeatable)")
    parser.add_argument("--token", help="HF token override", default=None)
    args = parser.parse_args()

    models: List[str] = args.model if args.model else DEFAULT_MODELS
    token = get_token(args.token)
    api = HfApi()

    print("Subject model preload verification")
    print("Token provided:" if token else "No token provided (may still work if public)")

    statuses = [model_status(api, m, token) for m in models]

    for st in statuses:
        print(f"\nModel: {st['model_id']}")
        if not st.get("exists_remote"):
            print(f"  Remote: NOT FOUND ({st.get('error')})")
            continue
        print("  Remote: OK")
        print(f"  pipeline_tag: {st.get('pipeline_tag')}" )
        print(f"  estimated size: {st.get('estimated_size_mb')} MB" )
        cached = already_cached(st['model_id'])
        print(f"  cached locally: {'yes' if cached else 'no'}")

    if args.dry_run and not args.download:
        print("\nDry run only. Use --download to fetch missing models.")
        return

    print("\nStarting downloads (will skip already cached)...")
    for m in models:
        if already_cached(m):
            print(f"  {m}: already cached, skipping")
            continue
        try:
            local_dir = ensure_download(m, token)
            print(f"  {m}: downloaded to {local_dir}")
        except Exception as e:
            print(f"  {m}: FAILED -> {e}")

    print("\nPreload complete.")

if __name__ == "__main__":
    main()
