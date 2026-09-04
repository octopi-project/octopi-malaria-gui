"""
kohler.py - minimal 405 nm LED on/off + intensity GUI for Kohler alignment.

No homing, no camera, no autofocus - it only opens the microcontroller and
drives the illumination. Run from the repo root:

    python3 kohler.py
    python3 kohler.py --simulation   # no hardware, prints commands only
"""

import os
import sys

# control/_def.py reads ./config/configuration*.ini relative to the cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QApplication, QCheckBox, QGridLayout, QLabel,
                            QPushButton, QSlider, QSpinBox, QWidget)

from control._def import CONTROLLER_SN, CONTROLLER_VERSION, ILLUMINATION_CODE
import control.microcontroller as microcontroller

ILLUMINATION_SOURCE = ILLUMINATION_CODE.ILLUMINATION_SOURCE_405NM
DEFAULT_INTENSITY = 20
SEND_INTERVAL_MS = 50  # coalesce slider drags into at most one command per 50 ms


class KohlerWidget(QWidget):

    def __init__(self, mcu):
        super().__init__()
        self.mcu = mcu
        self.is_on = False

        self.setWindowTitle('Kohler - 405 nm')

        self.btn_toggle = QPushButton('Turn ON')
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setMinimumHeight(48)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(DEFAULT_INTENSITY)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 100)
        self.spinbox.setValue(DEFAULT_INTENSITY)
        self.spinbox.setSuffix(' %')

        self.checkbox_keep_on = QCheckBox('keep LED on while adjusting')
        self.checkbox_keep_on.setChecked(True)

        layout = QGridLayout()
        layout.addWidget(QLabel('<b>405 nm intensity</b>'), 0, 0)
        layout.addWidget(self.slider, 0, 1)
        layout.addWidget(self.spinbox, 0, 2)
        layout.addWidget(self.btn_toggle, 1, 0, 1, 3)
        layout.addWidget(self.checkbox_keep_on, 2, 0, 1, 3)
        self.setLayout(layout)

        # throttle the serial writes issued while the slider is being dragged
        self.send_timer = QTimer(self)
        self.send_timer.setSingleShot(True)
        self.send_timer.setInterval(SEND_INTERVAL_MS)
        self.send_timer.timeout.connect(self.send_intensity)

        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        self.spinbox.valueChanged.connect(lambda _: self.send_timer.start())
        self.btn_toggle.toggled.connect(self.toggle_illumination)

        # push the starting intensity to the controller
        self.send_intensity()

    def send_intensity(self):
        intensity = self.spinbox.value()
        self.mcu.set_illumination(ILLUMINATION_SOURCE, intensity)
        # some firmware versions latch the new intensity only on the next
        # turn-on, so re-issue turn_on while the LED is already lit
        if self.is_on and self.checkbox_keep_on.isChecked():
            self.mcu.turn_on_illumination()

    def toggle_illumination(self, checked):
        self.is_on = checked
        if checked:
            self.mcu.set_illumination(ILLUMINATION_SOURCE, self.spinbox.value())
            self.mcu.turn_on_illumination()
            self.btn_toggle.setText('Turn OFF')
        else:
            self.mcu.turn_off_illumination()
            self.btn_toggle.setText('Turn ON')

    def closeEvent(self, event):
        # never leave the LED on
        try:
            self.mcu.turn_off_illumination()
            self.mcu.set_illumination(ILLUMINATION_SOURCE, 0)
        finally:
            self.mcu.close()
        event.accept()


def main():
    simulation = '--simulation' in sys.argv

    if simulation:
        mcu = microcontroller.Microcontroller_Simulation()
    else:
        mcu = microcontroller.Microcontroller(version=CONTROLLER_VERSION,
                                              sn=CONTROLLER_SN)

    app = QApplication(sys.argv)
    widget = KohlerWidget(mcu)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
