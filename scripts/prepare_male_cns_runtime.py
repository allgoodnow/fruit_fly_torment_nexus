"""Build an explicitly experimental LIF pack from audited MaleCNS contacts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
from nexus.datasets.male_cns_runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structural', type=Path, default=ROOT/'data/brain-male-cns-v1.0-structural')
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/male-cns-v1.0')
    parser.add_argument('--output', type=Path, default=ROOT/'data/brain-male-cns-v1.0-lif')
    args = parser.parse_args()
    result = prepare_runtime(args.structural, args.raw, args.output)
    print(json.dumps(result['model'], indent=2))
    print(f'Experimental runtime pack: {args.output}')


if __name__ == '__main__':
    main()
