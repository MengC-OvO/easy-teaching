import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_dependencies as audit


@pytest.mark.parametrize("trusted", [True, False])
def test_cpu_audit_requires_verified_origin_and_checks_upstream(monkeypatch, tmp_path, trusted):
    requirement = next(line.split(" @ ")[1] for line in Path("requirements-linux.in").read_text().splitlines()
                       if line.startswith("torch @ "))
    url, fragment = requirement.split("#")
    metadata = {"url": url if trusted else "https://untrusted.example/torch.whl",
                "archive_info": {"hashes": {"sha256": fragment.split("=")[1]}}}
    monkeypatch.setattr(audit, "distribution", lambda name: SimpleNamespace(
        version="2.15.0+cpu", read_text=lambda filename: json.dumps(metadata)))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert Path(command[command.index("-r") + 1]).read_text() == "torch==2.15.0\n"
        Path(command[-1]).write_text(json.dumps({"dependencies": [{"name": "torch", "version": "2.15.0", "vulns": []}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audit.subprocess, "run", run)
    # pip-audit omits the version field for dependencies it cannot resolve.
    assert audit.audit_official_cpu_torch({"name": "torch", "skip_reason": "Not on PyPI"}, tmp_path) is trusted
    assert len(calls) == int(trusted)
