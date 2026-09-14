"""One clock for brain-to-controller steering and articulated body physics."""
import math
import time
import traceback
from collections import deque
from queue import Empty

from .brain.motor import SteeringDecoder
from .brain.motor_effects import MotorEffects
from .brain.walking import WalkingDecoder
from .brain.protocol import Protocol
from .brain.telemetry import NeuralTelemetry
from .worker import put_latest
from .behavior import GroundBehavior
from .recovery import RecoveryMonitor
from .vision import EyeFeedback


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
        self.walking_decoder = WalkingDecoder(brain)
        self.neural_walking = False
        self.behavior = GroundBehavior(enabled=autonomous)
        self.resume_after_protocol = autonomous
        self.coupling_ticks = coupling_ticks
        self.baseline = 1.
        self.running = False
        self.protocol = None
        self.completed_protocol = None
        self.generation = 0
        self.reposition_count = 0
        self.monitor = NeuralTelemetry(brain)
        self.recovery = RecoveryMonitor()
        self.eyes = EyeFeedback(brain, body)
        self.sync_environment()

    def sync_recovery(self):
        active = (bool(self.brain.inputs.size) or self.brain.inhibition_gain != 1
                  or bool((self.brain.output_gain != 1).any()))
        self.recovery.controls(self.brain.step, active)

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
            self.walking_decoder.reset()
            self.behavior.reset()
            self.monitor.reset(self.brain)
            self.recovery.reset()
            self.eyes.reset()
            self.running, self.protocol = False, None
            self.completed_protocol = None
            self.generation += 1
            self.reposition_count = 0
            if self.environment is not None:
                self.environment.reset()
                self.sync_environment()
        elif kind == 'reposition_body':
            self.body.reposition()
            self.running = False
            self.reposition_count += 1
            self.recovery.body_repositioned(self.brain.step)
            self.eyes.next_sample = self.brain.step
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
        elif kind == 'loom':
            self.brain.set_looming(value)
            self.protocol = None
        elif kind == 'neural_release':
            self.eyes.configure(False)
            self.brain.release()
            self.protocol = None
        elif kind == 'release':
            self.baseline = 1.
        elif kind == 'protocol':
            candidate = Protocol(value, self.brain)
            self.eyes.configure(False)
            self.protocol, self.running = candidate, True
        elif kind == 'eye_feedback':
            if value and self.protocol and not self.protocol.completed:
                raise ValueError('Finish or release the prepared sequence before enabling visual input')
            self.eyes.configure(value)
        elif kind == 'vision_video':
            self.eyes.load_video(value)
            self.running = False
        elif kind == 'vision_eyes':
            self.eyes.use_eyes()
        elif kind == 'vision_restart':
            self.eyes.restart_video()
            self.running = False
        elif kind == 'bridge_enabled':
            self.decoder.enabled = bool(value)
            self.motor_effects.enabled = bool(value)
            self.walking_decoder.enabled = bool(value)
        elif kind == 'neural_walking':
            if not isinstance(value, bool):
                raise ValueError('Neural walking must be enabled or disabled')
            if value and not self.walking_decoder.output()['available']:
                raise ValueError('This pack has no mapped bilateral BDN2 walking readout')
            self.neural_walking = value
        elif kind == 'motor_effects_enabled':
            self.motor_effects.enabled = bool(value)
        elif kind == 'descending_enabled':
            self.motor_effects.descending_enabled = bool(value)
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
        self.sync_recovery()

    def advance(self, ticks=100):
        end = self.brain.step + ticks
        while self.brain.step < end:
            if self.protocol and self.protocol.completed:
                if self.running and self.resume_after_protocol and self.free_control_enabled:
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
                    if self.running and self.resume_after_protocol and self.free_control_enabled:
                        self.protocol = None
                        continue
                    self.running = False
                    break
                if self.protocol.cursor < len(self.protocol.commands):
                    at = self.protocol.origin+self.protocol.commands[self.protocol.cursor][0]
                    step = min(step, at-before)
                step = min(step, self.protocol.origin+self.protocol.duration-before)
            if self.brain.looming_input:
                step = min(step, self.brain.looming_input.end - before)
            step = self.eyes.before_step(step)
            self.sync_recovery()
            gains = self.brain.output_gain[self.decoder.indices].copy()
            walking_gains = self.brain.output_gain[self.walking_decoder.indices].copy()
            directly_driven = self.brain.inputs.copy()
            # The protocol applies endpoint events on the next iteration, after
            # this body's interval has used the correct pre-event output gains.
            self.brain.advance(step*.0001)
            elapsed = (self.brain.step-before)*.0001
            if self.environment is not None and self.environment.active:
                self.environment.active_seconds += elapsed
            counts = self.brain.counts-before_counts
            self.decoder.observe(counts[self.decoder.indices], elapsed, gains)
            self.walking_decoder.observe(counts[self.walking_decoder.indices], elapsed, walking_gains)
            self.motor_effects.observe(counts, elapsed, self.brain.output_gain, directly_driven)
            effects = self.motor_effects.output()
            interrupted = (effects['retreat'] > .05 or effects['escape'] > .05 or effects['disruption'] > .05 or
                           (self.decoder.enabled and max(self.decoder.rates) > 10.))
            if not self.neural_walking:
                self.behavior.advance(elapsed, interrupted)
            behavior = self.ground_output()
            output = self.motor_output(behavior)
            extra = {'escape': effects['escape'], 'disruption': effects['disruption']}
            if not any(extra.values()):
                extra = {}  # Preserve the original body adapter interface at baseline.
            if behavior['resting']:
                extra = {'resting': True}
            self.body.advance(elapsed, drive=output['drive'], turn=output['turn'], wander=False, **extra)
            self.eyes.after_step(self.brain.step - before)
            if abs(self.body.time-self.brain.time) > 1e-9:
                raise RuntimeError('Body and brain clocks diverged')
            self.recovery.observe(self.brain.step, int(counts.sum()), len(counts),
                                  escape=effects['escape'], disruption=effects['disruption'],
                                  retreat=effects['retreat'],
                                  walking_drive=self.walking_decoder.output(self.baseline)['drive'] if self.neural_walking else 0.,
                                  steering_hz=float(max(self.decoder.rates)) if self.decoder.enabled else 0.,
                                  upright=self.body.upright() if hasattr(self.body, 'upright') else None)
            self.sync_environment()
        if self.protocol:
            self.protocol.advance(self.brain, 0)
            if self.protocol.completed:
                self.completed_protocol = self.protocol
                self.running = self.running and self.resume_after_protocol and self.free_control_enabled
        self.sync_recovery()

    @property
    def free_control_enabled(self):
        return self.neural_walking or self.behavior.enabled

    def ground_output(self):
        if not self.neural_walking:
            return self.behavior.output(self.baseline)
        forward = self.walking_decoder.output(self.baseline)
        effects = self.motor_effects.output()
        active = max(forward['drive'], effects['retreat'], effects['escape'], effects['disruption']) >= .01
        return {'enabled': self.behavior.enabled, 'state': 'neural response' if active else 'neural idle',
                'drive': forward['drive'] if active else 0., 'turn': 0., 'resting': not active,
                'controller': 'BDN2 / DNa02 / MDN and escape readouts; engineered leg coordination'}

    def motor_output(self, behavior):
        retreat = self.motor_effects.output()['retreat']
        motor = self.decoder.output(behavior['drive'])
        motor['neural_turn'] = motor['turn']
        # MDNs command backward walking. Reverse stepping uses the upstream
        # FlyGym signed CPG interface; the gain is not a fitted biological rate.
        motor['drive'] = (1-retreat)*motor['drive']-.8*retreat
        motor['turn'] = (1-retreat)*(motor['turn']+behavior['turn'])
        motor['left_drive'] = motor['drive']+motor['turn']
        motor['right_drive'] = motor['drive']-motor['turn']
        motor['retreat'] = retreat
        return motor

    def snapshot(self):
        neural = self.monitor.snapshot(self.brain, self.running, self.generation, self.protocol,
                                       coupled=self.decoder.enabled or self.motor_effects.enabled
                                               or (self.neural_walking and self.walking_decoder.enabled))
        neural['shared_clock'] = True
        neural['eye_feedback'] = self.eyes.snapshot()
        food = self.environment.snapshot() if self.environment is not None else None
        neural['environment'] = food
        behavior = self.ground_output()
        motor = self.motor_output(behavior)
        neural['motor_bridge'] = motor
        neural['motor_effects'] = self.motor_effects.output()
        neural['ground_behavior'] = self.ground_output()
        neural['neural_walking'] = self.neural_walking
        neural['walking_decoder'] = self.walking_decoder.output(self.baseline)
        neural['resume_after_protocol'] = self.resume_after_protocol
        neural['recovery'] = self.recovery.snapshot()
        neural['reposition_count'] = self.reposition_count
        neural['body_upright'] = self.body.upright() if hasattr(self.body, 'upright') else None
        body = self.body.telemetry()
        body.update(running=self.running, wander=False, drive=self.baseline, turn=motor['turn'],
                    generation=self.generation, realtime_factor=neural['realtime_factor'],
                    mode='experimental DNa02 neural steering / engineered gait',
                    motor_bridge=motor, brain_time=self.brain.time, coupling_ms=self.coupling_ticks*.1,
                    environment=food)
        body['motor_effects'] = neural['motor_effects']
        body['ground_behavior'] = neural['ground_behavior']
        body['neural_walking'] = self.neural_walking
        body['walking_decoder'] = neural['walking_decoder']
        body['eye_feedback'] = neural['eye_feedback']
        if self.neural_walking:
            body['mode'] = 'BDN2 forward / DNa02 steering / MDN retreat; engineered gait'
        body['reposition_count'] = self.reposition_count
        return {'telemetry': body, 'brain': neural, 'vision_pixels': self.eyes.preview}


def simulate_coupled(directory, commands, frames, events, *, render=True, autonomous=True, allow_experimental=False):
    body = None
    session = None
    try:
        from .body import FlyBody
        from .brain.runtime import Brain, Connectome
        from .brain.targets import readout_ids
        brain = Brain(Connectome.load(directory, allow_experimental=allow_experimental))
        brain.resolve(readout_ids('mn9', brain.graph))
        brain.advance(.0001)
        brain.reset()
        body = FlyBody(render=render)
        session = CoupledSession(brain, body, autonomous=autonomous)
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
        if session is not None:
            session.eyes.close()
        if body is not None:
            body.close()
        frames.cancel_join_thread()
