"""Fail on known dependency vulnerabilities outside reviewed embedded-only exceptions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

# Chroma is embedded only. Its HTTP server and RBAC provider are never deployed.
# Review these exceptions whenever Chroma or the deployment topology changes.
EMBEDDED_CHROMA_EXCEPTIONS = {
    "PYSEC-2026-311", "PYSEC-2026-3813", "PYSEC-2026-3814", "PYSEC-2026-3815",
}


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
