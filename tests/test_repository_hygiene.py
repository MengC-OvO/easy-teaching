import pytest
from pathlib import Path
import subprocess
import sys

from scripts.check_repository_hygiene import forbidden_path


@pytest.mark.parametrize("path", [
    "tmp/pdfs/resume-review-1.png", "resume.pdf", "docs/interview-notes.md", ".env",
    ".env.production", "private-backups/snapshot.bundle", "data/exports/record.docx",
    "data/local/uploads/metadata.json", ".test-tmp/export.docx", "certificates/server.key",
    "storage.sqlite3", "docs/个人简历.pdf",
])
def test_reject_private_or_generated_files(path):
    assert forbidden_path(path)


@pytest.mark.parametrize("path", [
    ".env.example", "data/exports/.gitkeep", "app/schemas/approval_resume.py",
    "data/knowledge/raw/official/eylf-v2.0.pdf", "docs/deployment.md",
])
def test_allow_source_and_public_knowledge(path):
    assert not forbidden_path(path)


def test_history_check_catches_deleted_private_files(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "resume.pdf").write_bytes(b"synthetic fixture")
    git("add", "resume.pdf")
    git("commit", "-m", "fixture")
    git("rm", "resume.pdf")
    git("commit", "-m", "remove fixture")
    script = str(Path("scripts/check_repository_hygiene.py").resolve())
    assert subprocess.run([sys.executable, script], cwd=tmp_path, capture_output=True).returncode == 0
    result = subprocess.run([sys.executable, script, "--history"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 1
    assert b"resume.pdf" in result.stdout
