"""Independent neural worker; frames may drop, command events are acknowledged."""
from collections import deque
from queue import Empty
import time
import traceback

from nexus.worker import put_latest


def simulate_brain(directory, commands, frames, events):
    try:
        from .runtime import Brain, Connectome
        from .telemetry import NeuralTelemetry
        from .protocol import Protocol
        from .targets import MN9, SUGAR
        brain = Brain(Connectome.load(directory))
        if brain.graph.snapshot != "630":
            raise ValueError("This neural panel requires the v630 reference pack")
        brain.resolve(SUGAR + [MN9])
        # Compile before enabling controls, then restore the initial state.
        brain.advance(.0001)
        brain.reset()
        monitor = NeuralTelemetry(brain)
        running = False
        protocol = None
        generation = 0
        seen, recent = set(), deque()
        last_publish = 0
        dirty = True
        events.put({"kind":"ready", "neurons":len(brain.graph.ids), "edges":len(brain.graph.posts)})
        while True:
            for _ in range(32):
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
                try:
                    if kind == "running":
                        running = bool(value)
                        if protocol and protocol.completed:
                            protocol = None
                    elif kind == "step":
                        running = False
                        if protocol and not protocol.completed:
                            protocol.advance(brain,100)
                        else:
                            protocol = None
                            brain.advance(.01)
                    elif kind == "stimulate":
                        brain.stimulate(value["ids"], value["rate_hz"])
                        protocol = None
                    elif kind == "silence":
                        brain.silence(value["ids"])
                        protocol = None
                    elif kind == 'inhibition_gain':
                        brain.set_inhibition_gain(value)
                        protocol = None
                    elif kind == "release":
                        brain.release()
                        protocol = None
                    elif kind == "protocol":
                        candidate = Protocol(value, brain)
                        protocol = candidate
                        running = True
                    elif kind == "reset":
                        brain.reset()
                        protocol = None
                        running = False
                        generation += 1
                        monitor.reset(brain)
                    else:
                        raise ValueError(f"Unknown command: {kind}")
                except (ValueError, KeyError, TypeError) as error:
                    events.put({"kind":"rejected","id":cid,"message":str(error)})
                    continue
                events.put({"kind":"applied","id":cid,"command":kind,"value":value,
                            "sim_time":brain.time,"generation":generation})
                dirty = True
            if running:
                if protocol:
                    protocol.advance(brain,100)
                    if protocol.completed:
                        running = False
                        events.put({"kind":"protocol_complete","sim_time":brain.time})
                else:
                    brain.advance(.01)
                dirty = True
            now = time.perf_counter()
            if dirty and (now-last_publish >= .05 or not running):
                packet = monitor.snapshot(brain, running, generation, protocol)
                put_latest(frames, packet)
                last_publish, dirty = now, False
            if not running:
                time.sleep(.005)
    except BaseException:
        events.put({"kind":"error","message":traceback.format_exc()})
    finally:
        frames.cancel_join_thread()
