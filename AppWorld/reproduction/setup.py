"""Install the pinned upstream AppWorld checkout and the timeout-only patch."""
import argparse
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REVISION = (HERE / "appworld_revision.txt").read_text().strip()


def command(*args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def apply_patch(root):
    patch = HERE / "patches/appworld-http-timeout.patch"
    if subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root,
                      capture_output=True).returncode == 0:
        return
    command("git", "apply", "--check", patch, cwd=root)
    command("git", "apply", patch, cwd=root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    root = args.appworld_root.resolve()
    if not root.exists():
        command("git", "clone", "https://github.com/stonybrooknlp/appworld.git", root)
        command("git", "checkout", "--detach", REVISION, cwd=root)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"Expected AppWorld {REVISION}, found {actual}. Use a new checkout directory.")
    apply_patch(root)
    python = root / ".venv/bin/python"
    if not args.skip_install:
        if not python.exists():
            command(args.python, "-m", "venv", root / ".venv")
        pip_args = ["install", "-c", HERE / "constraints-appworld-py312.txt",
                    "-e", root, "-e", str(root / "experiments") + "[simplified]", "-e", HERE.parent]
        if shutil.which("uv"):
            command("uv", "pip", *pip_args, "--python", python)
        else:
            if subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode:
                command(python, "-m", "ensurepip", "--upgrade")
            command(python, "-m", "pip", *pip_args)
        command(python, "-m", "appworld.cli", "install", "--repo", cwd=root)
    if not args.skip_download and not (root / "data/datasets/dev.txt").exists():
        command(python, "-m", "appworld.cli", "download", "data", "--root", root, cwd=root)
    print(f"Ready: {root}\nPython: {python}")


if __name__ == "__main__":
    main()
