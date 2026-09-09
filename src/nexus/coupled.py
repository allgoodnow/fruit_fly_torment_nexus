"""One clock for brain-to-controller steering and articulated body physics."""
import math
import time
import traceback
from collections import deque
from queue import Empty

from .brain.motor import SteeringDecoder
from .brain.motor_effects import MotorEffects
from .brain.protocol import Protocol
from .brain.telemetry import NeuralTelemetry
from .worker import put_latest
from .behavior import GroundBehavior


class CoupledSession:
    def __init__(self, brain, body, *, coupling_ticks=100, environment=None, autonomous=False):
        if not isinstance(coupling_ticks, int) or not 1 <= coupling_ticks <= 100:
            raise ValueError('Coupling interval must be 1–100 neural ticks')
        if brain.step != 0 or body.time != 0:
            raise ValueError('Coupled session must begin with both clocks at zero')
        self.brain, self.body = brain, body
        self.environment = environment
        self.decoder = SteeringDecoder(brain)
        self.motor_effects = MotorEffects(brain)
        self.behavior = GroundBehavior(enabled=autonomous)
        self.resume_after_protocol = autonomous
        self.coupling_ticks = coupling_ticks
        self.baseline = 1.
        self.running = False
        self.protocol = None
        self.completed_protocol = None
        self.generation = 0
        self.monitor = NeuralTelemetry(brain)
        self.sync_environment()

    def sync_environment(self):
        if self.environment is not None:
            self.environment.sample(self.body.ground_contacts(), self.brain)
            self.body.food_patch = {'center_mm': self.environment.center.copy(),
                                    'radius_mm': self.environment.radius, 'present': self.environment.present}

    def command(self, kind, value=None):
        if kind == 'brain':
            action = 'neural_release' if value['kind'] == 'release' else value['kind']
            return self.command(action, value.get('value'))
        if kind == 'running':
            self.running = bool(value)
            if self.protocol and self.protocol.completed:
                self.protocol = None
        elif kind == 'step':
            self.running = False
            if self.protocol and self.protocol.completed:
                self.protocol = None
            self.advance(100)
        elif kind == 'reset':
            self.brain.reset()
            self.body.reset()
            self.decoder.reset()
            self.motor_effects.reset()
            self.behavior.reset()
            self.monitor.reset(self.brain)
            self.running, self.protocol = False, None
            self.completed_protocol = None
            self.generation += 1
            if self.environment is not None:
                self.environment.reset()
                self.sync_environment()
        elif kind == 'stimulate':
            self.brain.stimulate(value['ids'], value['rate_hz'])
            self.protocol = None
        elif kind == 'silence':
            self.brain.silence(value['ids'])
            self.protocol = None
        elif kind == 'inhibition_gain':
            self.brain.set_inhibition_gain(value)
            self.protocol = None
        elif kind == 'circuit':
            self.brain.set_circuit_input(value['name'], value['rate_hz'])
            self.protocol = None
        elif kind == 'heat':
            self.brain.set_heat(value)
            self.protocol = None
        elif kind == 'neural_release':
            self.brain.release()
            self.protocol = None
        elif kind == 'release':
            self.baseline = 1.
        elif kind == 'protocol':
            candidate = Protocol(value, self.brain)
            self.protocol, self.running = candidate, True
        elif kind == 'bridge_enabled':
            self.decoder.enabled = bool(value)
            self.motor_effects.enabled = bool(value)
        elif kind == 'motor_effects_enabled':
            self.motor_effects.enabled = bool(value)
        elif kind == 'autonomous':
            self.behavior.enabled = bool(value)
        elif kind == 'resume_after_protocol':
            self.resume_after_protocol = bool(value)
        elif kind == 'drive':
            drive = float(value)
            if not math.isfinite(drive):
                raise ValueError('Baseline drive must be finite')
            self.baseline = min(1.3, max(0., drive))
        elif kind in ('food_config', 'food_place'):
            if self.environment is None:
                raise ValueError('This session has no food environment')
            if kind == 'food_place':
                value = {'center_mm': self.body.food_position(value), 'present': True}
            self.environment.configure(value, self.brain.time)
            self.sync_environment()
        elif kind == 'food_demo':
            if self.environment is None:
                raise ValueError('This session has no food environment')
            self.environment.configure({'center_mm': [5., 0.], 'radius_mm': 2.5,
                                        'present': True, 'enabled': True, 'rate_hz': 200}, self.brain.time)
            self.baseline = 1.
            self.behavior.enabled = False
            self.command('reset')
            self.body.reset_camera()
            self.body.orbit(zoom=math.log(10/7))
            self.running = True
        elif kind == 'camera':
            self.body.orbit(**value)
        elif kind == 'camera_reset':
            self.body.reset_camera()
        else:
            raise ValueError(f'Unsupported coupled command: {kind}')

    def advance(self, ticks=100):
        end = self.brain.step + ticks
        while self.brain.step < end:
            if self.protocol and self.protocol.completed:
                if self.running and self.resume_after_protocol and self.behavior.enabled:
                    self.protocol = None
                else:
                    self.running = False
                    break
            step = min(self.coupling_ticks, end-self.brain.step)
            before = self.brain.step
            before_counts = self.brain.counts.copy()
            if self.protocol:
                # Stop at intervention boundaries too, so outgoing silencing is
                # applied to the same interval in the brain and motor decoder.
                self.protocol.advance(self.brain, 0)
                if self.protocol.completed:
                    self.completed_protocol = self.protocol
                    if self.running and self.resume_after_protocol and self.behavior.enabled:
                        self.protocol = None
                        continue
                    self.running = False
                    break
                if self.protocol.cursor < len(self.protocol.commands):
                    at = self.protocol.origin+self.protocol.commands[self.protocol.cursor][0]
                    step = min(step, at-before)
                step = min(step, self.protocol.origin+self.protocol.duration-before)
            gains = self.brain.output_gain[self.decoder.indices].copy()
            # The protocol applies endpoint events on the next iteration, after
            # this body's interval has used the correct pre-event output gains.
            self.brain.advance(step*.0001)
            elapsed = (self.brain.step-before)*.0001
            if self.environment is not None and self.environment.active:
                self.environment.active_seconds += elapsed
            counts = self.brain.counts-before_counts
            self.decoder.observe(counts[self.decoder.indices], elapsed, gains)
            self.motor_effects.observe(counts, elapsed, self.brain.output_gain, self.brain.inputs)
            effects = self.motor_effects.output()
            interrupted = (effects['escape'] > .05 or effects['disruption'] > .05 or
                           (self.decoder.enabled and max(self.decoder.rates) > 10.))
            self.behavior.advance(elapsed, interrupted)
            behavior = self.behavior.output(self.baseline)
            output = self.motor_output(behavior)
            extra = {'escape': effects['escape'], 'disruption': effects['disruption']}
            if not any(extra.values()):
                extra = {}  # Preserve the original body adapter interface at baseline.
            if behavior['resting']:
                extra = {'resting': True}
            self.body.advance(elapsed, drive=output['drive'], turn=output['turn'], wander=False, **extra)
            if abs(self.body.time-self.brain.time) > 1e-9:
                raise RuntimeError('Body and brain clocks diverged')
            self.sync_environment()
        if self.protocol:
            self.protocol.advance(self.brain, 0)
            if self.protocol.completed:
                self.completed_protocol = self.protocol
                self.running = self.running and self.resume_after_protocol and self.behavior.enabled

    def motor_output(self, behavior):
        motor = self.decoder.output(behavior['drive'])
        motor['neural_turn'] = motor['turn']
        motor['turn'] += behavior['turn']
        motor['left_drive'] += behavior['turn']
        motor['right_drive'] -= behavior['turn']
        return motor

    def snapshot(self):
        neural = self.monitor.snapshot(self.brain, self.running, self.generation, self.protocol,
                                       coupled=self.decoder.enabled or self.motor_effects.enabled)
        neural['shared_clock'] = True
        food = self.environment.snapshot() if self.environment is not None else None
        neural['environment'] = food
        behavior = self.behavior.output(self.baseline)
        motor = self.motor_output(behavior)
        neural['motor_bridge'] = motor
        neural['motor_effects'] = self.motor_effects.output()
        neural['ground_behavior'] = self.behavior.output(self.baseline)
        neural['resume_after_protocol'] = self.resume_after_protocol
        body = self.body.telemetry()
        body.update(running=self.running, wander=False, drive=self.baseline, turn=motor['turn'],
                    generation=self.generation, realtime_factor=neural['realtime_factor'],
                    mode='experimental DNa02 neural steering / engineered gait',
                    motor_bridge=motor, brain_time=self.brain.time, coupling_ms=self.coupling_ticks*.1,
                    environment=food)
        body['motor_effects'] = neural['motor_effects']
        body['ground_behavior'] = neural['ground_behavior']
        return {'telemetry': body, 'brain': neural}


def simulate_coupled(directory, commands, frames, events, *, render=True, autonomous=True, allow_experimental=False):
    body = None
    try:
        from .body import FlyBody
        from .brain.runtime import Brain, Connectome
        from .brain.targets import readout_ids
        from .environment import FoodEnvironment
        brain = Brain(Connectome.load(directory, allow_experimental=allow_experimental))
        brain.resolve(readout_ids('mn9', brain.graph))
        sugar = readout_ids('sugar', brain.graph) if brain.graph.snapshot == '630' else []
        brain.advance(.0001)
        brain.reset()
        body = FlyBody(render=render)
        session = CoupledSession(brain, body, environment=FoodEnvironment(targets=sugar), autonomous=autonomous)
        seen, recent = set(), deque()
        events.put({'kind': 'ready', 'coupled': True, 'dataset': brain.graph.snapshot})
        dirty, last_publish = True, 0
        completion_logged = None
        while True:
            stepped = False
            for _ in range(32):
                try:
                    command = commands.get_nowait()
                except Empty:
                    break
                cid, kind = command['id'], command['kind']
                if cid in seen:
                    continue
                seen.add(cid)
                recent.append(cid)
                if len(recent) > 4096:
                    seen.remove(recent.popleft())
                if kind == 'shutdown':
                    return
                value = command.get('value')
                try:
                    effective = value['kind'] if kind == 'brain' else kind
                    if kind == 'brain' and effective == 'release':
                        session.command('neural_release')
                    else:
                        session.command(kind, value)
                except (ValueError, KeyError, TypeError) as error:
                    events.put({'kind': 'rejected', 'id': cid, 'message': str(error)})
                    continue
                stepped |= effective == 'step'
                events.put({'kind': 'applied', 'id': cid, 'command': effective, 'value': value,
                            'sim_time': brain.time, 'generation': session.generation, 'coupled': True})
                dirty = True
            if session.running and not stepped:
                session.advance()
                dirty = True
            if session.completed_protocol and completion_logged is not session.completed_protocol:
                completion_logged = session.completed_protocol
                events.put({'kind': 'protocol_complete',
                            'sim_time': (completion_logged.origin+completion_logged.duration)*.0001, 'coupled': True})
            now = time.perf_counter()
            if dirty and (now-last_publish >= .05 or not session.running):
                packet = session.snapshot()
                packet['pixels'] = body.render() if render else None
                put_latest(frames, packet)
                last_publish, dirty = now, False
            if not session.running:
                time.sleep(.005)
    except BaseException:
        events.put({'kind': 'error', 'message': traceback.format_exc()})
    finally:
        if body is not None:
            body.close()
        frames.cancel_join_thread()
