import json
import multiprocessing as mp
import os
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from nexus.receptor_preparation import preparation_worker
from nexus.receptor_preparation_dialog import ReceptorPreparationDialog
from test_receptor_preparation import CURVE, preparation_pack, movie


@pytest.fixture
def job(preparation_pack, movie, tmp_path):
    app = QApplication.instance() or QApplication([])
    dialog = ReceptorPreparationDialog(movie, preparation_pack, tmp_path)
    settings = {'video': str(movie), 'pack': str(preparation_pack), 'output_dir': str(dialog.output_dir),
                'white_rate_hz': 30000., 'transfer': 'srgb', 'duration_ms': 150,
                'receptor_ids': ['13'], 'release_curve': str(CURVE)}
    ctx = mp.get_context('spawn')
    dialog.cancel = ctx.Event()
    try:
        yield dialog, settings
    finally:
        dialog.shutdown()
        dialog.close()


def run_without_progress_reader(dialog, settings):
    ctx = mp.get_context('spawn')
    dialog.process = ctx.Process(target=preparation_worker, args=(settings, str(dialog.status_path), dialog.cancel))
    dialog.process.start()
    # The producer must finish even when the dialog hasn't consumed any progress.
    dialog.process.join(timeout=30)
    assert not dialog.process.is_alive() and dialog.process.exitcode == 0


def test_completed_worker_leaves_a_loadable_outcome_without_a_progress_reader(job):
    dialog, settings = job
    run_without_progress_reader(dialog, settings)
    outcome = json.loads(dialog.status_path.read_text())
    assert outcome['kind'] == 'complete' and outcome['sequence'] > 1
    assert not dialog.load_button.isEnabled()
    dialog.poll()
    assert dialog.load_button.isEnabled() and dialog.failure is None
    assert dialog.result_files['report_path'] == str(dialog.output_dir/'report.json')
    assert dialog.status.text() == 'Ready · 150 ms · 1 receptors'
    assert not dialog.status_path.with_suffix('.json.tmp').exists()


def test_moved_video_leaves_an_error_and_never_enables_loading(job):
    from pathlib import Path
    dialog, settings = job
    Path(settings['video']).unlink()  # This test's own clip, moved after selection.
    run_without_progress_reader(dialog, settings)
    outcome = json.loads(dialog.status_path.read_text())
    assert outcome['kind'] == 'failed' and 'FileNotFoundError' in outcome['detail']
    dialog.poll()
    assert dialog.failure and dialog.status.text() == outcome['message']
    assert 'FileNotFoundError' in dialog.status.toolTip()
    assert not dialog.load_button.isEnabled() and dialog.result_files is None
    assert not dialog.output_dir.exists()
