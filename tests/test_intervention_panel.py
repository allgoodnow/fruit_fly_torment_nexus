import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QObject,Signal
from PySide6.QtWidgets import QApplication
from nexus.intervention_panel import InterventionPanel


class BrainPanelStub(QObject):
    snapshot=Signal(dict)
    def __init__(self):
        super().__init__()
        self.telemetry={};self.failed=False;self.sent=[]
    def send(self,*args):self.sent.append(args)


def test_pause_resume_and_reposition_controls_follow_acknowledged_state():
    app=QApplication.instance() or QApplication([])
    b=BrainPanelStub();panel=InterventionPanel(b)
    for running in [False,True]:
        b.telemetry={'running':running,'sim_time':.1,'body_upright':-1.}
        b.snapshot.emit(b.telemetry)
        assert panel.pause.text()==('Pause' if running else 'Resume')
        panel.pause.click()
        assert b.sent[-1]==('running',not running)
        assert not panel.reposition.isHidden()
    panel.reposition.click()
    assert b.sent[-1]==('reposition_body',)
    b.telemetry.update(running=False,body_upright=1.)
    b.snapshot.emit(b.telemetry)
    assert panel.reposition.isHidden()
    panel.close()
