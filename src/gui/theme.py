"""
Theme helpers for PySide6.
Supports automatic dark/light mode detection (system theme) with
Windows registry fallback when Qt colour-scheme detection returns Unknown.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor, QGuiApplication
from PySide6.QtCore import Qt


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_dark_mode() -> bool:
    """Return True when the OS is in dark mode."""
    # Qt 6.5+ native detection
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass

    # Windows registry fallback (AppsUseLightTheme = 0 → dark)
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return val == 0
    except Exception:
        pass

    return False   # safe default: light


# ---------------------------------------------------------------------------
# Application-level palette + stylesheet
# ---------------------------------------------------------------------------

def apply_theme(app: QApplication, dark: bool) -> None:
    """Apply dark or light Fusion palette to *app*."""
    app.setStyle("Fusion")
    if dark:
        _apply_dark_palette(app)
    else:
        _apply_light_palette(app)
    _apply_stylesheet(app, dark)


def apply_dark_theme(app: QApplication) -> None:
    """Convenience: auto-detect system theme and apply."""
    apply_theme(app, is_dark_mode())


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

def _apply_dark_palette(app: QApplication) -> None:
    p = QPalette()
    dark   = QColor(45,  45,  48)
    darker = QColor(30,  30,  30)
    mid    = QColor(60,  60,  63)
    light  = QColor(80,  80,  84)
    text   = QColor(220, 220, 220)
    bright = QColor(255, 255, 255)
    accent = QColor(42,  130, 218)
    dis    = QColor(120, 120, 120)
    link   = QColor(100, 170, 255)

    p.setColor(QPalette.ColorRole.Window,          dark)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            darker)
    p.setColor(QPalette.ColorRole.AlternateBase,   dark)
    p.setColor(QPalette.ColorRole.ToolTipBase,     bright)
    p.setColor(QPalette.ColorRole.ToolTipText,     bright)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          mid)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      bright)
    p.setColor(QPalette.ColorRole.Link,            link)
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, bright)
    p.setColor(QPalette.ColorRole.Light,           light)
    p.setColor(QPalette.ColorRole.Midlight,        mid)
    p.setColor(QPalette.ColorRole.Mid,             mid)
    p.setColor(QPalette.ColorRole.Dark,            darker)
    p.setColor(QPalette.ColorRole.Shadow,          QColor(0, 0, 0))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       dis)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
    app.setPalette(p)


def _apply_light_palette(app: QApplication) -> None:
    """Standard Fusion light — just reset to default."""
    app.setPalette(QPalette())


def _apply_stylesheet(app: QApplication, dark: bool) -> None:
    if dark:
        app.setStyleSheet("""
            QToolTip { color:#fff; background:#2d2d30; border:1px solid #555; }
            QGroupBox { border:1px solid #555; border-radius:4px;
                        margin-top:8px; padding-top:4px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; color:#aaa; }
            QTabBar::tab { background:#3c3c3f; color:#ccc;
                           padding:6px 14px; border:1px solid #555;
                           border-bottom:none; border-radius:3px 3px 0 0; }
            QTabBar::tab:selected { background:#2a82da; color:#fff; }
            QScrollBar:vertical { width:8px; background:#1e1e1e; }
            QScrollBar::handle:vertical { background:#555; border-radius:4px; min-height:20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QPushButton { padding:5px 12px; border-radius:4px; }
            QPushButton:hover { background-color:#4a4a4f; }
            QProgressBar { border:1px solid #555; border-radius:4px;
                           text-align:center; color:#fff; }
            QProgressBar::chunk { background-color:#2a82da; border-radius:3px; }
        """)
    else:
        app.setStyleSheet("""
            QGroupBox { border:1px solid #ccc; border-radius:4px;
                        margin-top:8px; padding-top:4px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; }
            QPushButton { padding:5px 12px; border-radius:4px; }
            QProgressBar { border:1px solid #bbb; border-radius:4px; text-align:center; }
            QProgressBar::chunk { background-color:#2a82da; border-radius:3px; }
        """)
