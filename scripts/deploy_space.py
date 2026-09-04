"""Push the demo to a Hugging Face Space.

    huggingface-cli login
    python scripts/deploy_space.py --repo <your-username>/age-gender-classifier

Assembles a staging directory rather than pushing the repository as-is: the Space
needs app.py at its root, the serving requirements rather than the training ones,
and the saved model, but none of the notebooks, caches or training scripts.
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (source, destination-inside-the-space). Everything else is left behind.
CONTENT = [
    ("space/app.py", "app.py"),
    ("space/README.md", "README.md"),
    ("space/requirements.txt", "requirements.txt"),
    ("agc", "agc"),
    ("scripts/demo.py", "scripts/demo.py"),
    ("scripts/__init__.py", "scripts/__init__.py"),
    ("artifacts/deep.keras", "artifacts/deep.keras"),
]


#: Never ship compiled bytecode: it is stale the moment the Space rebuilds on a
#: different Python, and it is pure noise in the Space's file listing.
JUNK = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def stage(into):
    for src, dst in CONTENT:
        s, d = os.path.join(ROOT, src), os.path.join(into, dst)
        if not os.path.exists(s):
            sys.exit(f"missing: {src}\n(run scripts/prep.py then scripts/train.py first?)")
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copytree(s, d, ignore=JUNK) if os.path.isdir(s) else shutil.copy2(s, d)
    return into


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/age-gender-classifier")
    ap.add_argument("--dry-run", action="store_true", help="stage and list, upload nothing")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        listing = sorted(os.path.relpath(os.path.join(r, f), tmp)
                         for r, _, fs in os.walk(tmp) for f in fs)
        print("\n".join("  " + f for f in listing))
        if args.dry_run:
            print(f"\ndry run: {len(listing)} files staged, nothing uploaded")
            return

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.repo, repo_type="space")
        print(f"\nhttps://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
