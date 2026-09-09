import multiprocessing as mp
from pathlib import Path
import time

from nexus.coupled import simulate_coupled


def test_coupled_worker_deduplicates_steps_rejects_invalid_input_and_preserves_order():
    ctx = mp.get_context('spawn')
    commands, frames, events = ctx.Queue(), ctx.Queue(maxsize=2), ctx.Queue()
    pack = Path(__file__).resolve().parents[1] / 'data/brain-v630'
    process = ctx.Process(target=simulate_coupled, args=(str(pack), commands, frames, events), kwargs={'render': False})
    process.start()
    try:
        assert events.get(timeout=30)['kind'] == 'ready'
        for command in [
            {'id': 1, 'kind': 'brain', 'value': {'kind': 'step'}},
            {'id': 1, 'kind': 'brain', 'value': {'kind': 'step'}},
            {'id': 2, 'kind': 'brain', 'value': None},
            {'id': 3, 'kind': 'reset'},
            {'id': 4, 'kind': 'drive', 'value': .5},
            {'id': 5, 'kind': 'step'},
            {'id': 6, 'kind': 'brain', 'value': {'kind': 'release'}},
        ]:
            commands.put(command)
        replies = [events.get(timeout=10) for _ in range(6)]
        assert [r['id'] for r in replies] == [1, 2, 3, 4, 5, 6]
        assert replies[1]['kind'] == 'rejected'
        assert [r['sim_time'] for r in replies if r['kind'] == 'applied'] == [.01, 0, 0, .01, .01]
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            p = frames.get(timeout=10)
            if p['telemetry']['generation'] == 1 and p['telemetry']['sim_time'] == .01:
                break
        assert p['brain']['sim_time'] == p['telemetry']['sim_time'] == .01
        assert p['telemetry']['drive'] == .5
        assert not p['brain']['running']
        commands.put({'id': 7, 'kind': 'shutdown'})
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for q in (commands, frames, events):
            q.cancel_join_thread()
            q.close()
