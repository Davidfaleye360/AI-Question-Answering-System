"""Put the demo on a Hugging Face Space.

    huggingface-cli login                      # once; paste a token with Write permission
    python scripts/deploy_space.py --dry-run   # build the folder and show what would be uploaded
    python scripts/deploy_space.py             # create or update the Space

A Space is a small git repository on Hugging Face that runs `app.py`. It needs the app, the
few `src/` modules the app imports, the config and a requirements file, laid out like this:

    app.py  README.md  requirements.txt  configs/distilbert.yaml  src/...

This script copies exactly those files into a temporary folder and uploads it, so the Space
is always a copy of what is in this repository. The models are not uploaded: the app
downloads them from the Hub when it starts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = "Ingsyling/squad2-qa-demo"

# {file in this repo: where it goes in the Space}
SPACE_FILES = {
    "app/app.py": "app.py",
    "app/README.md": "README.md",                 # starts with the Space's settings (sdk, version, title)
    "app/requirements.txt": "requirements.txt",
    "configs/distilbert.yaml": "configs/distilbert.yaml",
    "src/__init__.py": "src/__init__.py",
    "src/config.py": "src/config.py",
    "src/preprocess.py": "src/preprocess.py",
    "src/postprocess.py": "src/postprocess.py",
    "src/predict.py": "src/predict.py",
    "src/demo.py": "src/demo.py",
}


def build_space_folder(destination: str | Path, root: Path = ROOT) -> list[str]:
    """Copy the Space's files into `destination`. Returns the paths created, relative to it."""
    destination = Path(destination)
    for source, target in SPACE_FILES.items():
        (destination / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, destination / target)
    return sorted(SPACE_FILES.values())


def deploy(space_id: str, folder: str | Path, api=None) -> str:
    """Create the Space if needed and upload the folder. Returns the Space's URL."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()   # uses the token saved by `huggingface-cli login`, or the HF_TOKEN variable
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=space_id, repo_type="space",
                      commit_message="Update the demo from the project repository")
    return f"https://huggingface.co/spaces/{space_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--space", default=DEFAULT_SPACE, help="username/space-name")
    parser.add_argument("--dry-run", action="store_true", help="build the folder but upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        files = build_space_folder(folder)
        print(f"Files for the Space {args.space}:")
        for name in files:
            print(f"  {name:28} {(Path(folder) / name).stat().st_size:>7,} bytes")
        if args.dry_run:
            print("Dry run: nothing was uploaded.")
            return 0
        url = deploy(args.space, folder)
    print(f"Uploaded. The Space builds for a few minutes, then runs at {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
