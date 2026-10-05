import imageio_ffmpeg
import numpy as np

from nexus.photon_input import VideoPhotonInput, relative_intensity


def test_video_exposure_preserves_spatial_input_photons_and_clock_across_chunks(tmp_path):
    registry = {'snapshot': 'male-cns:v1.0',
                'circuits': {'eye_left': {'ids': ['1']}, 'eye_right': {'ids': ['2']}},
                'visual_columns': {'format': 'nexus-visual-columns-1', 'eyes': {
                    'L': {'cells': [{'id': '1', 'uv': [.25, .5]}]},
                    'R': {'cells': [{'id': '2', 'uv': [.75, .5]}]}}}}
    path = tmp_path/'spatial.mp4'
    writer = imageio_ffmpeg.write_frames(str(path), (64, 48), fps=20, codec='libx264rgb',
                                       pix_fmt_out='rgb24', quality=10, macro_block_size=1,
                                       ffmpeg_log_level='error')
    writer.send(None)
    for position in ('dark', 'left', 'right', 'gray', 'dark'):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        if position == 'left': frame[:, :32] = 255
        if position == 'right': frame[:, 32:] = 255
        if position == 'gray': frame[:] = 128
        writer.send(frame)
    writer.close()
    inputs = [VideoPhotonInput(path, registry, ['1', '2'], [88100, 88101],
                              white_rate_hz=30000, transfer='srgb') for _ in range(2)]
    try:
        whole = inputs[0].read(300)
        pieces = [inputs[1].read(ms) for ms in (37, 41, 123, 99)]
        for key in whole:
            np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in pieces]))
        assert inputs[0].time_ms == inputs[1].time_ms == 250
        assert inputs[0].ended and inputs[1].ended
        intensity, photons = whole['intensity'], whole['absorbed_photons']
        assert not photons[:50].any() and not photons[200:].any()
        assert photons[50:100, 0].sum() > 0 and not photons[50:100, 1].any()
        assert photons[100:150, 1].sum() > 0 and not photons[100:150, 0].any()
        np.testing.assert_allclose(intensity[150:200], .2158605, atol=1e-7)
        assert relative_intensity(np.full((1, 1, 3), 128, np.uint8), transfer='linear')[0, 0] > .5
    finally:
        for source in inputs:
            source.close()
