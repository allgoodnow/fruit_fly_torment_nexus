"""Restore a narrowly supported photoreceptor sign on existing MaleCNS edges."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus.datasets.histamine import upgrade_pack


def main():
    manifest, report = upgrade_pack(ROOT / 'data/brain-male-cns-v1.0-lif',
                                    ROOT / 'data/brain-male-cns-v1.0-structural',
                                    ROOT / 'data/raw/male-cns-v1.0')
    for path, data in [(ROOT / 'data/manifests/male-cns-runtime-v5.json', manifest),
                       (ROOT / 'experiments/histamine-evidence-v1.json', report)]:
        path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
