import pytest

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
