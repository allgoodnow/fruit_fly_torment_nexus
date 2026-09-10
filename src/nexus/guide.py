"""An offline guide shipped with the native application."""
from pathlib import Path
import sys

from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox


def guide_path():
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
    return root/'docs/user-guide.md'


class GuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Fruit Fly Nexus — Guide')
        self.resize(780, 680)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        try:
            self.browser.setMarkdown(guide_path().read_text())
        except OSError as error:
            self.browser.setPlainText(f'Guide unavailable: {error}')
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
