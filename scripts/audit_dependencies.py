"""Fail on known dependency vulnerabilities outside reviewed embedded-only exceptions."""

import json
from importlib.metadata import distribution
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
from packaging.version import Version

# Chroma is embedded only. Its HTTP server and RBAC provider are never deployed.
# Review these exceptions whenever Chroma or the deployment topology changes.
EMBEDDED_CHROMA_EXCEPTIONS = {
    "PYSEC-2026-311", "PYSEC-2026-3813", "PYSEC-2026-3814", "PYSEC-2026-3815",
}


def audit_official_cpu_torch(dependency, folder):
    """Audit a hash-verified official CPU wheel against its upstream release."""
    if dependency["name"] != "torch":
        return False
    installed = distribution("torch")
    version = installed.version
    if Version(version).local != "cpu" or dependency.get("version", version) != version:
        return False
    expected = next((line.partition(" @ ")[2] for line in
                     (Path(__file__).resolve().parents[1] / "requirements-linux.in").read_text().splitlines()
                     if line.startswith("torch @ ")), "")
    if not expected.startswith("https://download-r2.pytorch.org/whl/cpu/torch-"):
        return False
    provenance = json.loads(installed.read_text("direct_url.json") or "{}")
    url = urlsplit(expected)
    if unquote(provenance.get("url", "")) != unquote(expected.split("#")[0]):
        return False
    sha = url.fragment.removeprefix("sha256=")
    if len(sha) != 64 or provenance.get("archive_info", {}).get("hashes", {}).get("sha256") != sha:
        return False
    upstream = Version(version).public
    requirement = Path(folder) / "cpu-torch.txt"
    report = Path(folder) / "cpu-audit.json"
    requirement.write_text(f"torch=={upstream}\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "pip_audit", "--no-deps", "--disable-pip",
                             "-r", str(requirement), "--format", "json", "--output", str(report)], check=False)
    if result.returncode != 0 or not report.exists():
        return False
    packages = json.loads(report.read_text(encoding="utf-8"))["dependencies"]
    if len(packages) != 1 or packages[0].get("skip_reason") or packages[0].get("vulns"):
        return False
    print(f"Audited official SHA256-verified CPU wheel against torch {upstream} advisories.")
    return True


def main():
    with tempfile.TemporaryDirectory() as folder:
        report = Path(folder) / "audit.json"
        result = subprocess.run([sys.executable, "-m", "pip_audit", "--format", "json", "--output", str(report)], check=False)
        if result.returncode not in {0, 1} or not report.exists():
            return 2
        findings = json.loads(report.read_text(encoding="utf-8"))
        unexpected = []
        for dependency in findings["dependencies"]:
            if dependency.get("skip_reason"):
                if not audit_official_cpu_torch(dependency, folder):
                    print(f"Unaudited: {dependency['name']}: {dependency['skip_reason']}")
                    unexpected.append(f"{dependency['name']}: could not audit")
            for vulnerability in dependency.get("vulns", []):
                reviewed = dependency["name"] == "chromadb" and vulnerability["id"] in EMBEDDED_CHROMA_EXCEPTIONS
                label = f"{dependency['name']} {dependency['version']}: {vulnerability['id']}"
                print(("Reviewed embedded-only exception: " if reviewed else "Unresolved: ") + label)
                if not reviewed:
                    unexpected.append(label)
        return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
