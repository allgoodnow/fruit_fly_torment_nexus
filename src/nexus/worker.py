"""Process boundary for native UI, physics, and ordered commands."""
from __future__ import annotations

from collections import deque
from queue import Empty, Full
import time
import traceback


def put_latest(channel, value):
    """Display data may be dropped; command acknowledgements never use this queue."""
    try:
        channel.put_nowait(value)
    except Full:
        try:
            channel.get_nowait()
        except Empty:
            pass
        try:
            channel.put_nowait(value)
        except Full:
            pass


def simulate(commands, frames, events, *, render=True):
    body = None
    try:
        from nexus.body import FlyBody

        body = FlyBody(render=render)
        running, wander = False, True
        drive, turn = 1.0, 0.0
        generation = 0
        seen, recent = set(), deque()
        last_render = 0.0
        last_time = body.time
        last_wall = time.perf_counter()
        dirty = True
        events.put({"kind": "ready", "telemetry": body.telemetry()})
        while True:
            stepped = False
            for _ in range(64):
                try:
                    command = commands.get_nowait()
                except Empty:
                    break
                cid, kind = command["id"], command["kind"]
                if cid in seen:
                    continue
                seen.add(cid)
                recent.append(cid)
                if len(recent) > 4096:
                    seen.remove(recent.popleft())
                value = command.get("value")
                if kind == "shutdown":
                    return
                if kind == "running":
                    running = bool(value)
                elif kind == "step":
                    running = False
                    # Execute here so motor changes and reset retain queue order.
                    body.advance(0.01, drive=drive, turn=turn, wander=wander)
                    stepped = True
                elif kind == "reset":
                    running = False
                    body.reset()
                    generation += 1
                    last_time, last_wall = body.time, time.perf_counter()
                elif kind == "drive":
                    drive = min(1.3, max(0.0, float(value)))
                elif kind == "turn":
                    turn = min(0.6, max(-0.6, float(value)))
                elif kind == "wander":
                    wander = bool(value)
                elif kind == "release":
                    drive, turn, wander = 1.0, 0.0, True
                elif kind == "camera":
                    body.orbit(**value)
                elif kind == "camera_reset":
                    body.reset_camera()
                else:
                    events.put({"kind": "rejected", "id": cid, "message": f"Unknown command: {kind}"})
                    continue
                events.put({"kind": "applied", "id": cid, "command": kind, "value": value,
                            "sim_time": body.time, "generation": generation})
                dirty = True

            if running and not stepped:
                body.advance(0.004, drive=drive, turn=turn, wander=wander)
                dirty = True
            now = time.perf_counter()
            if dirty and (now - last_render >= 1 / 30 or not running):
                telemetry = body.telemetry()
                telemetry.update(running=running, wander=wander, drive=drive, turn=turn,
                                 generation=generation,
                                 realtime_factor=(body.time-last_time)/max(now-last_wall, 1e-9))
                pixels = body.render() if render else None
                put_latest(frames, {"telemetry": telemetry, "pixels": pixels})
                last_time, last_wall = body.time, now
                last_render, dirty = now, False
            if not running:
                time.sleep(0.01)
    except BaseException:
        events.put({"kind": "error", "message": traceback.format_exc()})
    finally:
        if body is not None:
            body.close()
        # Don't let a final queued image keep the process alive at shutdown.
        frames.cancel_join_thread()
