"""Prepare an unsigned MaleCNS structural pack without changing the live model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus.datasets.male_cns import audit_and_prepare, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw/male-cns-v1.0')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/brain-male-cns-v1.0-structural')
    parser.add_argument('--report', type=Path, default=ROOT / 'experiments/male-cns-v1.0-audit.json')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/manifests/male-cns-v1.0.json').read_text())
    verify_sources(args.raw, manifest, download=args.download)
    report = audit_and_prepare(args.raw, args.output, manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.report)
    print(f"Prepared {report['classified_neurons']:,} cells and {report['retained_connection_rows']:,} connections")
    print(f'Report: {args.report}')
    print('The application still uses FlyWire v630; this structural pack is not a runtime model.')


if __name__ == '__main__':
    main()
