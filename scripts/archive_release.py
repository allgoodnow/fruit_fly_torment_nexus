"""Archive an already-built, acceptance-tested Linux bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
bundle = ROOT / "dist" / "FruitFlyTormentNexus-1.1.1"
report_path = ROOT / "runs" / "branding_111_packaged_acceptance" / "acceptance.json"
report = json.loads(report_path.read_text())
if not report.get("success") or report.get("version") != "1.1.1":
    raise SystemExit("Packaged acceptance must pass before archiving.")
for name in ("THIRD_PARTY_NOTICES.md", "environment.json", "README.md"):
    shutil.copyfile(ROOT / name, bundle / name)
shutil.copyfile(ROOT / 'docs/user-guide.md', bundle / 'USER_GUIDE.md')
shutil.copytree(ROOT / 'docs/assets', bundle / 'docs/assets', dirs_exist_ok=True)
shutil.copyfile(ROOT / 'packaging/START_HERE.txt', bundle / 'START_HERE.txt')
shutil.copytree(ROOT / 'experiments', bundle / 'experiments', dirs_exist_ok=True)
shutil.copyfile(report_path, bundle / "acceptance.json")
(bundle/'BUILD_INFO.json').write_text(json.dumps({
    'version': report['version'],
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'tracked_changes_at_archive': bool(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip()),
}, indent=2)+'\n')
archive = ROOT / "dist" / "fruit-fly-torment-nexus-1.1.1-linux-x86_64.tar.gz"
with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
    tar.add(bundle, arcname="FruitFlyTormentNexus")
with archive.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
archive.with_name("SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum}, indent=2))
