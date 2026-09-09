import multiprocessing as mp
import time
from queue import Empty

from nexus.worker import simulate


def test_worker_orders_commands_deduplicates_steps_and_shuts_down():
    ctx = mp.get_context("spawn")
    commands, frames, events = ctx.Queue(), ctx.Queue(maxsize=2), ctx.Queue()
    process = ctx.Process(target=simulate, args=(commands, frames, events), kwargs={"render": False})
    process.start()
    try:
        assert events.get(timeout=30)["kind"] == "ready"
        for command in (
            {"id": 1, "kind": "step"},
            {"id": 1, "kind": "step"},
            {"id": 2, "kind": "reset"},
            {"id": 3, "kind": "drive", "value": 0.5},
            {"id": 4, "kind": "step"},
            {"id": 5, "kind": "release"},
        ):
            commands.put(command)
        applied = [events.get(timeout=10) for _ in range(5)]
        assert [event["id"] for event in applied] == [1, 2, 3, 4, 5]
        assert [event["sim_time"] for event in applied] == [0.01, 0, 0, 0.01, 0.01]
        assert applied[-1]["generation"] == 1
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            t = frames.get(timeout=10)["telemetry"]
            if t["generation"] == 1 and t["sim_time"] == 0.01 and t["drive"] == 1:
                break
        else:
            raise AssertionError("Final worker state not received")
        assert not t["running"] and t["wander"]
        assert t["turn"] == 0
        commands.put({"id": 6, "kind": "shutdown"})
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        for queue in (commands, frames, events):
            queue.cancel_join_thread()
            queue.close()
