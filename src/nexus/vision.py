"""Clocked video and eye-camera input, with optional spatial sampling and adaptation."""
import numpy as np

SAMPLE_TICKS = 500  # 50 ms of simulated time, independent of GUI refresh.


def pooled_brightness(frames, mask):
    frames, mask = np.asarray(frames), np.asarray(mask)
    if (frames.ndim != 4 or frames.shape[0] != 2 or frames.shape[-1] != 3
            or frames.dtype != np.uint8 or mask.shape != frames.shape[1:3]
            or mask.dtype != np.bool_ or not mask.any()):
        raise ValueError('Expected two RGB8 eye frames and a nonempty retinal mask')
    # Equal RGB average is an image-intensity proxy, not fly spectral sensitivity.
    return frames[:, mask, :].mean(axis=(1, 2)) / 255.


class EyeFeedback:
    def __init__(self, brain, body):
        self.brain, self.body = brain, body
        self.video = None
        self.preview = None
        self.video_origin = brain.step
        registry = brain.graph.circuits or {}
        from .retina import VisualColumns
        self.spatial = VisualColumns(registry) if 'visual_columns' in registry else None
        from .light_adaptation import LightAdaptation
        self.adaptation = LightAdaptation()
        self.available = (brain.graph.snapshot == 'male-cns:v1.0'
                          and all(registry.get('circuits', {}).get(k, {}).get('ids')
                                  for k in ['eye_left', 'eye_right'])
                          and hasattr(body, 'eye_brightness'))
        self.reset()

    def reset(self):
        if self.video is not None:
            self.video.close()
            self.video = None
        self.preview = None
        self.mapping_mode = 'pooled'
        self.adaptive = False
        self.adaptation.reset()
        self.enabled = False
        self.next_sample = self.brain.step
        self.sample_step = None
        self.brightness = None
        self.samples = 0
        self.error = None
        self.brain.clear_eye_input()

    def configure(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Eye feedback must be enabled or disabled')
        if enabled and not self.available:
            raise ValueError('Eye feedback requires mapped MaleCNS photoreceptors and eye cameras')
        if enabled != self.enabled:
            self.enabled = enabled
            self.next_sample = self.brain.step
        if enabled:
            self.error = None
        if not enabled:
            self.brain.clear_eye_input()

    def load_video(self, path):
        from .video import VideoSource
        if not self.available:
            raise ValueError('Video input requires mapped MaleCNS eye cohorts')
        candidate = VideoSource(path)
        if self.video is not None:
            self.video.close()
        self.configure(False)
        self.video = candidate
        self.preview = candidate.frame
        self.video_origin = self.brain.step
        self.adaptation.reset()
        self.sample_step = None
        self.brightness = None
        self.samples = 0
        self.error = None

    def use_eyes(self):
        self.configure(False)
        self.adaptation.reset()
        if self.video is not None:
            self.video.close()
            self.video = None
        self.preview = None
        self.mapping_mode = 'pooled'
        self.sample_step = None
        self.brightness = None
        self.samples = 0
        self.error = None

    def restart_video(self):
        if self.video is None:
            raise ValueError('Load a video first')
        self.load_video(self.video.path)

    def set_mapping(self, mode):
        if mode not in ('pooled', 'spatial'):
            raise ValueError('Unknown visual input mapping')
        if mode == 'spatial' and (self.video is None or self.spatial is None):
            raise ValueError('Spatial input requires a video and the visual-column pack update')
        self.configure(False)
        self.mapping_mode = mode
        self.brightness = None
        self.adaptation.reset()

    def set_adaptation(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('Light adaptation must be enabled or disabled')
        self.configure(False)
        self.adaptive = enabled
        self.adaptation.reset()
        self.brightness = None

    def after_step(self, ticks):
        # Turning feedback off freezes video position even if the body still runs.
        if self.video is not None and not self.enabled:
            self.video_origin += ticks
        if self.enabled and self.adaptive:
            self.adaptation.advance(ticks)

    def before_step(self, step):
        if not self.enabled:
            return step
        if self.brain.step >= self.next_sample:
            if self.video is None:
                value = self.body.eye_brightness()
                self.preview = getattr(self.body, 'eye_preview', None)
            else:
                try:
                    frame = self.video.frame_at((self.brain.step - self.video_origin) // SAMPLE_TICKS)
                except ValueError as error:
                    self.configure(False)
                    self.error = str(error)
                    return step
                if frame is None:
                    self.configure(False)
                    return step
                self.preview = frame
                value = np.full(2, frame.mean() / 255.)
            if self.video is not None and self.mapping_mode == 'spatial':
                values = self.spatial.sample(frame)
                self.brightness = [float(values[s].mean()) for s in ['L', 'R']]
                driven = self.adaptation.sample(values) if self.adaptive else values
                self.brain.set_spatial_eye_input(driven)
            else:
                self.brightness = [float(x) for x in value]
                values = {'L': np.asarray([value[0]]), 'R': np.asarray([value[1]])}
                driven = self.adaptation.sample(values) if self.adaptive else values
                self.brain.set_eye_input([float(driven[s][0]) for s in ['L', 'R']])
            self.sample_step = self.brain.step
            self.next_sample = self.brain.step + SAMPLE_TICKS
            self.samples += 1
        return min(step, self.next_sample - self.brain.step)

    def snapshot(self):
        return {'available': bool(self.available), 'enabled': self.enabled,
                'sample_ms': None if self.sample_step is None else self.sample_step / 10,
                'sample_interval_ms': SAMPLE_TICKS / 10, 'samples': self.samples,
                'brightness': self.brightness,
                'source': 'video' if self.video is not None else 'eyes',
                'video_name': self.video.path.name if self.video is not None else None,
                'video_frame': self.video.index if self.video is not None else None,
                'video_ended': self.video.ended if self.video is not None else False,
                'error': self.error,
                'rates_hz': {side: float(np.asarray(rate).mean()) for side, (_, rate) in self.brain.eye_inputs.items()},
                'rate_ranges_hz': {side: [float(np.min(rate)), float(np.max(rate))]
                                   for side, (_, rate) in self.brain.eye_inputs.items()},
                'spatial_available': self.spatial is not None,
                'mapping_mode': self.mapping_mode,
                'adaptation': {'selected': self.adaptive, **self.adaptation.snapshot()},
                'mapping': ('spatial-video-columns-v1; inferred columns, uncalibrated projection, 0-100 Hz'
                            if self.mapping_mode == 'spatial' else
                            'pooled-eye-brightness-v1; unfitted 0-100 Hz R1-R6 input')}

    def close(self):
        if self.video is not None:
            self.video.close()
