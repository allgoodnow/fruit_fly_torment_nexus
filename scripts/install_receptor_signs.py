"""Install the audited visual receptor policy from original MaleCNS data."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus.datasets.histamine import upgrade_pack


def main():
    manifest, report = upgrade_pack(ROOT / 'data/brain-male-cns-v1.0-lif',
                                    ROOT / 'data/brain-male-cns-v1.0-structural',
                                    ROOT / 'data/raw/male-cns-v1.0', extended=True)
    for path, data in [(ROOT / 'data/manifests/male-cns-runtime-v6.json', manifest),
                       (ROOT / 'experiments/receptor-sign-evidence-v1.json', report)]:
        path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'model': report['model'], 'changed_weights': report['changed_weights'],
                      'rules': [{k: r[k] for k in ['id', 'edges', 'contacts']} for r in report['rules']]}))


if __name__ == '__main__':
    main()
