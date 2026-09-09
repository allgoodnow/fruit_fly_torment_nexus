"""Native controls for the visible food patch and sensory feedback."""
from PySide6.QtWidgets import QGroupBox, QFormLayout, QHBoxLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QLabel, QPushButton


class FoodPanel(QGroupBox):
    def __init__(self, send):
        super().__init__('Food and sensory feedback')
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.present = QCheckBox('Food present')
        self.feedback = QCheckBox('Taste feedback')
        self.present.setChecked(True)
        self.feedback.setChecked(True)
        self.present.toggled.connect(lambda value: send('food_config', {'present': value}))
        self.feedback.toggled.connect(lambda value: send('food_config', {'enabled': value}))
        row.addWidget(self.present)
        row.addWidget(self.feedback)
        form.addRow(row)
        self.rate = QSpinBox()
        self.rate.setRange(1, 1000)
        self.rate.setValue(200)
        self.rate.setSuffix(' Hz')
        self.rate.valueChanged.connect(lambda value: send('food_config', {'rate_hz': value}))
        form.addRow('Model taste input', self.rate)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(.25, 20)
        self.radius.setValue(2.5)
        self.radius.setSingleStep(.25)
        self.radius.setSuffix(' mm')
        self.radius.valueChanged.connect(lambda value: send('food_config', {'radius_mm': value}))
        form.addRow('Patch radius', self.radius)
        row = QHBoxLayout()
        ahead = QPushButton('Place ahead')
        under = QPushButton('Place under fly')
        ahead.clicked.connect(lambda: send('food_place', 'ahead'))
        under.clicked.connect(lambda: send('food_place', 'under'))
        row.addWidget(ahead)
        row.addWidget(under)
        form.addRow(row)
        self.location = QLabel('Patch centre: 5.0, 0.0 mm')
        form.addRow(self.location)
        demo = QPushButton('Reset + run food demo')
        demo.clicked.connect(lambda: send('food_demo'))
        form.addRow(demo)
        note = QLabel('Foot contact drives a pooled taste-input proxy.\nManual inputs and food inputs remain separate.')
        note.setWordWrap(True)
        form.addRow(note)

    def receive(self, food):
        for widget, value in ((self.present, food['present']), (self.feedback, food['enabled'])):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        for widget, value in ((self.rate, food['rate_hz']), (self.radius, food['radius_mm'])):
            if not widget.hasFocus():
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        x, y = food['center_mm']
        self.location.setText(f'Patch centre: {x:.1f}, {y:.1f} mm')
