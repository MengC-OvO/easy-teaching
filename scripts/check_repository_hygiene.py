"""Reject private documents and generated files from the Git index."""

from pathlib import PurePosixPath
import argparse
import re
import subprocess


PRIVATE_PREFIXES = (
    "tmp/", "private-backups/", "reports/", "artifacts/", ".test-tmp/", ".test-cache/",
    "data/local/", "data/chroma/", "data/knowledge/index/", ".venv/", ".venv-safety/",
)
PERSONAL_DOCUMENT = re.compile(r"(^|[/_ -])(resume|cv|curriculum.vitae|interview|简历|面试)([/_. -]|$)", re.I)


def forbidden_path(path: str) -> bool:
    normalized = path.lower()
    name = PurePosixPath(normalized).name
    if normalized.startswith(PRIVATE_PREFIXES) or normalized.startswith(".pytest"):
        return True
    if normalized.startswith("data/exports/") and name != ".gitkeep":
        return True
    if (name == ".env" or name.startswith(".env.") or name.endswith(".env")) and name != ".env.example":
        return True
    if PurePosixPath(normalized).suffix in {".pem", ".key", ".db", ".sqlite", ".sqlite3"}:
        return True
    if (PERSONAL_DOCUMENT.search(normalized) or "简历" in normalized or "面试" in normalized) and PurePosixPath(normalized).suffix in {
        ".md", ".txt", ".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".webp"
    }:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also check every commit reachable from HEAD")
    args = parser.parse_args()
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    if args.history:
        for commit in subprocess.check_output(["git", "rev-list", "HEAD"]).decode().splitlines():
            paths.extend(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", commit]).decode("utf-8").split("\0"))
    blocked = sorted({path for path in paths if path and forbidden_path(path)})
    if blocked:
        print("Private or generated files must not be committed:")
        print("\n".join(blocked))
        return 1
    print("Repository file hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
