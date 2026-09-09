"""Archive an already-built, acceptance-tested Linux bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
bundle = ROOT / "dist" / "FruitFlyNexus-0.6.0"
report_path = ROOT / "runs" / "perturbation_packaged_acceptance" / "acceptance.json"
report = json.loads(report_path.read_text())
if not report.get("success"):
    raise SystemExit("Packaged acceptance must pass before archiving.")
for name in ("README.md", "THIRD_PARTY_NOTICES.md", "environment.json"):
    shutil.copyfile(ROOT / name, bundle / name)
shutil.copytree(ROOT / 'docs', bundle / 'docs', dirs_exist_ok=True)
shutil.copytree(ROOT / 'experiments', bundle / 'experiments', dirs_exist_ok=True)
for name in ('IMPLEMENTATION_PLAN.md', 'PROGRESS.md'):
    shutil.copyfile(ROOT / name, bundle / name)
shutil.copyfile(report_path, bundle / "acceptance.json")
archive = ROOT / "dist" / "fruit-fly-nexus-0.6.0-linux-x86_64.tar.gz"
with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
    tar.add(bundle, arcname="FruitFlyNexus")
with archive.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
archive.with_name("SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum}, indent=2))
