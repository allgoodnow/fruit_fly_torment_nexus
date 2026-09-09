"""CB0059 candidate tested in an active network, with matched output ablation."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.circuits import circuit_ids


def main():
    out = ROOT/'runs/aversion_probe'
    out.mkdir(parents=True, exist_ok=True)
    brain = Brain(Connectome.load(ROOT/'data/brain-v630'))
    rows = []
    for seed in [73100, 73101, 73102]:
        for blocked in [False, True]:
            brain.seed = seed
            brain.reset()
            brain.set_heat(40)
            brain.set_circuit_input('aversion_proxy', 200)
            if blocked:
                brain.silence(circuit_ids('aversion_proxy'))
            for _ in range(60):
                brain.advance(.01)
            row = {'seed': seed, 'outputs_blocked': blocked, 'spikes': int(brain.counts.sum()),
                   'candidate_spikes': int(brain.counts[brain.resolve(circuit_ids('aversion_proxy'))].sum()),
                   'mean_voltage_mv': float(brain.v.mean())}
            rows.append(row)
            print(row, flush=True)
        active, blocked = rows[-2:]
        assert active['candidate_spikes'] == blocked['candidate_spikes'] > 0
        assert active['spikes'] != blocked['spikes']
    report = {'format': 'nexus-aversion-candidate-assay-1', 'success': True, 'duration_ms': 600,
              'background': '7 TRN_VP2 cells at 300 Hz; scenario temperature 40°C',
              'candidate': '2 CB0059 cells at 200 Hz, inhibitory edges in v630',
              'interpretation': 'Output-dependent network modulation only; not a validation of pain, nociception, or learning.',
              'trials': rows}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
