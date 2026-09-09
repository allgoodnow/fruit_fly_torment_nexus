"""Collect reproducibility information and installed distribution licenses."""
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
licenses = ROOT / "licenses"
licenses.mkdir(exist_ok=True)
shutil.copyfile(ROOT / "vendor/flygym/LICENSE", licenses / "flygym-LICENSE")
python_license = Path(sysconfig.get_path("stdlib")) / "LICENSE.txt"
if python_license.exists():
    shutil.copyfile(python_license, licenses / "Python-LICENSE.txt")
elif not (licenses / "Python-LICENSE.txt").exists():
    raise SystemExit("Obtain the Python runtime license before packaging.")
inventory = []
for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
    name = dist.metadata["Name"]
    inventory.append({"name": name, "version": dist.version})
    for path in dist.files or ():
        if ".dist-info/" not in str(path):
            continue
        if any(word in path.name.lower() for word in ("license", "copying", "notice")):
            source = Path(dist.locate_file(path))
            if source.is_file():
                target = licenses / name / str(path).split(".dist-info/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
(ROOT / "environment.json").write_text(json.dumps({
    "python": sys.version, "platform": sys.platform,
    "flygym_commit": "38c8ec61034cd59bc5ba0de20688d4a3c0000d60", "packages": inventory,
}, indent=2))
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
lines = ["./vendor/flygym" if line.startswith("flygym @") else line for line in freeze.splitlines()]
(ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n")
print(f"Collected {len(inventory)} distributions and their available license files.")
