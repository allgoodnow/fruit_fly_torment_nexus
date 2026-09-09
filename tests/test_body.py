import numpy as np
import pytest
from nexus.body import FlyBody


def test_reset_repeats_trajectory_and_chunking_preserves_physics():
    body = FlyBody(render=False)
    try:
        body.advance(0.1)
        expected = body.sim.mj_data.qpos.copy()
        assert body.telemetry()["displacement_mm"] > 0.1
        body.reset()
        assert body.time == 0
        for _ in range(10):
            body.advance(0.01)
        np.testing.assert_allclose(body.sim.mj_data.qpos, expected, atol=1e-10, rtol=0)
        assert body.time == pytest.approx(0.1)
        before = body.sim.mj_data.qpos.copy()
        body.orbit(dx=500, dy=-100, zoom=-20)
        np.testing.assert_array_equal(body.sim.mj_data.qpos, before)
        assert 2.5 <= body.camera.distance <= 35
    finally:
        body.close()
