"""Validated timed commands, measured in neural simulation ticks."""
from copy import deepcopy
import math

from .runtime import DT_MS


def ticks(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Times must be finite numbers in milliseconds")
    result = round(value / DT_MS)
    if value < 0 or not math.isclose(result*DT_MS, value, abs_tol=1e-9):
        raise ValueError("Times must be nonnegative multiples of 0.1 ms")
    return result


class Protocol:
    def __init__(self, description, brain):
        if not isinstance(description, dict):
            raise ValueError("Protocol must be a JSON object")
        self.description = deepcopy(description)
        if description.get("format") != "nexus-protocol-1":
            raise ValueError("Unsupported protocol format")
        if description.get('dataset', brain.graph.snapshot) != brain.graph.snapshot:
            raise ValueError('Protocol belongs to another neural dataset')
        self.duration = ticks(description["duration_ms"])
        if not 0 < self.duration <= 6000000:
            raise ValueError("Protocol duration must be between 0.1 ms and 10 minutes")
        events = description.get("events")
        if not isinstance(events, list) or len(events) > 1000:
            raise ValueError("Protocol must contain a list of at most 1000 events")
        self.commands = []
        previous = -1
        for item in events:
            if not isinstance(item, dict):
                raise ValueError("Each event must be a JSON object")
            at = ticks(item["at_ms"])
            if at < previous or at > self.duration:
                raise ValueError("Events must be ordered and inside the protocol duration")
            action = item["action"]
            if action in ("stimulate", "silence"):
                targets = brain.resolve(item["ids"])
                if not 0 < len(targets) <= 256:
                    raise ValueError("Choose between 1 and 256 target neurons")
                if action == "stimulate":
                    rate = float(item["rate_hz"])
                    if not math.isfinite(rate) or not 0 < rate <= 1000:
                        raise ValueError("Input rate must be in (0, 1000] Hz")
            elif action == 'inhibition_gain':
                brain.validate_inhibition_gain(item['gain'])
            elif action == 'circuit':
                brain.validate_circuit(item['name'], item['rate_hz'])
            elif action == 'heat':
                brain.validate_heat(item['celsius'])
            elif action != "release":
                raise ValueError(f"Unknown protocol action: {action}")
            self.commands.append((at, deepcopy(item)))
            previous = at
        self.cursor = 0
        self.origin = brain.step
        self.completed = False

    def advance(self, brain, steps):
        end = min(brain.step + steps, self.origin + self.duration)
        while True:
            while self.cursor < len(self.commands) and self.origin+self.commands[self.cursor][0] <= brain.step:
                _, command = self.commands[self.cursor]
                if command["action"] == "stimulate":
                    brain.stimulate(command["ids"], command["rate_hz"])
                elif command["action"] == "silence":
                    brain.silence(command["ids"])
                elif command['action'] == 'inhibition_gain':
                    brain.set_inhibition_gain(command['gain'])
                elif command['action'] == 'circuit':
                    brain.set_circuit_input(command['name'], command['rate_hz'])
                elif command['action'] == 'heat':
                    brain.set_heat(command['celsius'])
                else:
                    brain.release()
                self.cursor += 1
            if brain.step >= end:
                break
            boundary = self.origin+self.commands[self.cursor][0] if self.cursor < len(self.commands) else end
            next_step = min(end, boundary)
            brain.advance((next_step-brain.step)*DT_MS/1000)
        self.completed = brain.step == self.origin+self.duration


def taste_protocol(ids):
    return {"format":"nexus-protocol-1", "name":"Taste baseline / stimulus / release",
            "duration_ms":1100, "events":[
                {"at_ms":0,"action":"release"},
                {"at_ms":100,"action":"stimulate","ids":list(ids),"rate_hz":200},
                {"at_ms":600,"action":"release"}]}
