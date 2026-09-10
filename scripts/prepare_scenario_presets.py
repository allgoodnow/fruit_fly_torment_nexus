"""Build dataset-bound saved sequences from the controls' actual preset compiler."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nexus.brain.scenarios import scenario_protocol, sequence_protocol


def main():
    output = ROOT/'experiments'
    for dataset, folder, pain in [('male-cns:v1.0', output, 'nociception_proxy'),
                                  ('630', output/'flywire-v630', 'aversion_proxy')]:
        folder.mkdir(exist_ok=True)
        for name in ['defensive', 'aversion', 'heat', 'seizure', 'heat_overload']:
            protocol = scenario_protocol(name, dataset=dataset, pain_circuit=pain,
                                         celsius=100 if name == 'heat_overload' else 40)
            (folder/f'{name}-preset.json').write_text(json.dumps(protocol, indent=2)+'\n')
    stages = [
        {'name': 'defensive', 'baseline_ms': 1750, 'stimulus_ms': 250, 'recovery_ms': 500},
        {'name': 'aversion', 'baseline_ms': 100, 'stimulus_ms': 300, 'recovery_ms': 500},
        {'name': 'heat', 'baseline_ms': 100, 'stimulus_ms': 300, 'recovery_ms': 500},
        {'name': 'seizure', 'baseline_ms': 100, 'stimulus_ms': 300, 'recovery_ms': 1000},
        {'name': 'heat_overload', 'celsius': 100, 'baseline_ms': 100, 'stimulus_ms': 300, 'recovery_ms': 1500},
    ]
    plan = sequence_protocol(stages, dataset='male-cns:v1.0', pain_circuit='nociception_proxy')
    (output/'male-cns-continuous-sequence.json').write_text(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
