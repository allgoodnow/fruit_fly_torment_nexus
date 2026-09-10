"""Ten-second post-release neural observations across three seeds and histories."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.runtime/numba'))
import numpy as np
from nexus.brain.runtime import Brain, Connectome
from nexus.brain.protocol import Protocol
from nexus.brain.scenarios import scenario_protocol


def main():
    directory = ROOT/'data/brain-male-cns-v1.0-lif'
    graph = Connectome.load(directory, allow_experimental=True)
    continuous = json.loads((ROOT/'experiments/male-cns-recovery-sequence.json').read_text())
    trials = []
    for seed in [73100, 73101, 73102]:
        for name in ['aversion', 'seizure', 'continuous']:
            description = copy.deepcopy(continuous) if name == 'continuous' else scenario_protocol(
                name, dataset='male-cns:v1.0', pain_circuit='nociception_proxy',
                baseline_ms=100, stimulus_ms=300, recovery_ms=10000)
            release = max(e['at_ms'] for e in description['events'] if e['action'] == 'release')
            brain = Brain(graph, seed=seed)
            protocol = Protocol(description, brain)
            rates = []
            quiet_ticks = 0
            first_quiet_ms = None
            begin = time.perf_counter()
            while not protocol.completed:
                before = int(brain.counts.sum())
                start = brain.step
                protocol.advance(brain, 100)
                if start/10 >= release:
                    assert not brain.inputs.size and brain.inhibition_gain == 1
                    rate = (int(brain.counts.sum())-before)/.01/len(brain.counts)
                    quiet_ticks = quiet_ticks+100 if rate <= .1 else 0
                    if quiet_ticks >= 5000 and first_quiet_ms is None:
                        first_quiet_ms = brain.step/10-release
                    rates.append(rate)
            bins = np.asarray(rates).reshape(-1, 10).mean(axis=1)
            row = {'scenario': name, 'seed': seed, 'release_ms': release,
                   'observed_after_release_ms': len(rates)*10, 'total_spikes': int(brain.counts.sum()),
                   'first_500ms_neural_quiet_after_release_ms': first_quiet_ms,
                   'final_neural_quiet_duration_ms': quiet_ticks/10,
                   'last_second_hz_per_neuron': float(np.mean(rates[-100:])),
                   'post_release_100ms_mean_hz_per_neuron': bins.tolist(),
                   'protocol': description, 'wall_seconds': time.perf_counter()-begin,
                   'end_counts_sha256': hashlib.sha256(brain.counts.astype('<i8').tobytes()).hexdigest()}
            trials.append(row)
            print({k: v for k, v in row.items() if k not in ('protocol', 'post_release_100ms_mean_hz_per_neuron')}, flush=True)
    result = {'format': 'nexus-neural-recovery-assay-1', 'success': True,
              'dataset': graph.snapshot, 'model': graph.model,
              'pack_manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
              'quiet_rule': 'At most 0.1 Hz per neuron in every 10 ms interval for 500 consecutive ms.',
              'trials': trials,
              'limitations': ['Neural observations only; no body simulation in this assay.',
                              'Operational quiet is not biological recovery or subjective relief.',
                              'Absence of quiet within ten seconds does not establish permanent instability.']}
    (ROOT/'experiments/recovery-neural-v1-results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
