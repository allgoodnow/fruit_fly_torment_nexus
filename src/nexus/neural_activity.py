"""A bounded, simulation-time plot of exact population spike counts."""
import pyqtgraph as pg


class NeuralActivityPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, title='Neural activity')
        self.setLabel('bottom', 'Brain time', units='s')
        self.setLabel('left', 'Mean firing rate', units='Hz')
        self.setToolTip('All neurons · spikes per second per neuron · 10 ms bins · last 5 simulated seconds')
        self.curve = self.plot(pen=pg.mkPen('#b91c1c', width=1.5),
                               fillLevel=0, brush=(185, 28, 28, 25))
        self.setXRange(0, 1, padding=0)
        self.setYRange(0, 1, padding=0)

    def update_snapshot(self, packet):
        activity = packet['population_activity']
        times = [step*.0001 for step in activity['end_steps']]
        rates = activity['rates_hz_per_neuron']
        self.curve.setData(times, rates)
        right = max(1., packet['sim_time'])
        left = max(0., right-activity['window_ms']/1000)
        self.setXRange(left, right, padding=0)
        peak = max(rates, default=0.)
        self.setYRange(0, peak*1.1 if peak else 1., padding=0)
