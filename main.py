"""
Coypu-Feeder — entry point (run from project root).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PySide6.QtWidgets import QApplication
from gui.theme import is_dark_mode, apply_theme
from gui.app import App


def main():
    app = QApplication(sys.argv)
    apply_theme(app, is_dark_mode())
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
