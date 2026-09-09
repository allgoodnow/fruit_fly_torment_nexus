import multiprocessing as mp
from pathlib import Path
import time

from nexus.brain.worker import simulate_brain


def test_independent_worker_applies_rejects_and_releases_overlay_while_paused():
    ctx = mp.get_context('spawn')
    commands, frames, events = ctx.Queue(), ctx.Queue(maxsize=2), ctx.Queue()
    pack = Path(__file__).resolve().parents[1] / 'data/brain-v630'
    process = ctx.Process(target=simulate_brain, args=(str(pack), commands, frames, events))
    process.start()

    def receive(gain, step):
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            packet = frames.get(timeout=15)
            if packet['tick'] == step and packet['inhibition_gain'] == gain:
                return packet
        raise AssertionError('Worker did not publish the expected overlay state')

    try:
        assert events.get(timeout=30)['kind'] == 'ready'
        receive(1., 0)
        for item in [{'id': 1, 'kind': 'inhibition_gain', 'value': .25},
                     {'id': 2, 'kind': 'step'}, {'id': 2, 'kind': 'step'},
                     {'id': 3, 'kind': 'inhibition_gain', 'value': -1}]:
            commands.put(item)
        replies = [events.get(timeout=15) for _ in range(3)]
        assert [r['kind'] for r in replies] == ['applied', 'applied', 'rejected']
        packet = receive(.25, 100)
        assert not packet['running']
        commands.put({'id': 4, 'kind': 'release'})
        assert events.get(timeout=15)['kind'] == 'applied'
        restored = receive(1., 100)
        assert restored['total_spikes'] == packet['total_spikes']
        commands.put({'id': 5, 'kind': 'shutdown'})
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for channel in (commands, frames, events):
            channel.cancel_join_thread()
            channel.close()
