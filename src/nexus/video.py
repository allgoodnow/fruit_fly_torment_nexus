"""Bounded local-video decoder, resampled to the visual input's 20 Hz clock."""
from pathlib import Path

import numpy as np


class VideoSource:
    def __init__(self, path):
        import imageio_ffmpeg
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise ValueError('Choose an existing local video file')
        self.stream = imageio_ffmpeg.read_frames(
            str(self.path), output_params=['-an', '-vf',
                'fps=20,scale=320:180:force_original_aspect_ratio=decrease'])
        self.index, self.ended = 0, False
        try:
            self.metadata = next(self.stream)
            self.width, self.height = self.metadata['size']
            if not (0 < self.width <= 320 and 0 < self.height <= 180):
                raise ValueError('Unexpected decoded video dimensions')
            self.frame = self._decode(next(self.stream))
        except Exception as error:
            self.close()
            raise ValueError('Cannot decode this video; try an MP4 with an H.264 video track') from error

    def _decode(self, data):
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3).copy()

    def frame_at(self, index):
        if index < self.index:
            raise ValueError('Restart the video before seeking backwards')
        if self.ended:
            return None
        try:
            while self.index < index:
                self.frame = self._decode(next(self.stream))
                self.index += 1
        except StopIteration:
            self.ended = True
            self.close()
            return None
        except Exception as error:
            self.ended = True
            self.close()
            raise ValueError('Video decoding failed during playback') from error
        return self.frame

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
