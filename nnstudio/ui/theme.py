"""Dark theme: palette tokens plus the stylesheet built from them."""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont

BACKGROUND = "#0E1320"
SURFACE = "#151C2C"
SURFACE_ALT = "#1B2438"
ELEVATED = "#212C44"
BORDER = "#2C3852"
BORDER_STRONG = "#3B4A6B"
TEXT = "#E8EEFF"
TEXT_MUTED = "#8D9CBF"
ACCENT = "#5B8DEF"
ACCENT_HOVER = "#6F9CF5"
ACCENT_DEEP = "#3F6FD1"
INPUT_COLOR = "#38BDF8"
HIDDEN_COLOR = "#7C7CF0"
OUTPUT_COLOR = "#F5B44C"
SUCCESS = "#3DD68C"
WARNING = "#FFB020"
DANGER = "#FF6B6B"

QCOLORS = {
    "background": QColor(BACKGROUND),
    "surface": QColor(SURFACE),
    "surface_alt": QColor(SURFACE_ALT),
    "border": QColor(BORDER),
    "text": QColor(TEXT),
    "muted": QColor(TEXT_MUTED),
    "input": QColor(INPUT_COLOR),
    "hidden": QColor(HIDDEN_COLOR),
    "output": QColor(OUTPUT_COLOR),
    "accent": QColor(ACCENT),
}

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", "Ubuntu", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{ background-color: {BACKGROUND}; }}

QFrame#Card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

QLabel#CardTitle {{
    color: {TEXT};
    font-size: 14px;
    font-weight: 600;
    padding: 2px 0px;
}}

QLabel#Subtle {{ color: {TEXT_MUTED}; }}
QLabel#Hint {{ color: {TEXT_MUTED}; font-size: 12px; }}
QLabel#Warning {{ color: {WARNING}; font-size: 12px; }}
QLabel#Danger {{ color: {DANGER}; font-size: 12px; }}
QLabel#Success {{ color: {SUCCESS}; font-size: 12px; }}
QLabel#Metric {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#Result {{
    font-size: 17px;
    font-weight: 600;
    color: {SUCCESS};
    padding: 10px;
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#SectionLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0px 6px;
    color: {ACCENT};
}}

QPushButton {{
    background-color: {ELEVATED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 7px 14px;
    color: {TEXT};
    font-weight: 600;
}}
QPushButton:hover {{ background-color: {BORDER}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background-color: {SURFACE_ALT}; }}
QPushButton:disabled {{ color: #5A6784; background-color: {SURFACE_ALT}; border-color: {BORDER}; }}

QPushButton#Primary {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #0B1120;
}}
QPushButton#Primary:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#Primary:pressed {{ background-color: {ACCENT_DEEP}; }}
QPushButton#Primary:disabled {{ background-color: #2E3A55; border-color: #2E3A55; color: #63718F; }}

QPushButton#Danger {{ border-color: {DANGER}; color: {DANGER}; }}
QPushButton#Danger:hover {{ background-color: #3A2233; }}

QPushButton#Ghost {{
    background-color: transparent;
    border: 1px dashed {BORDER_STRONG};
    color: {TEXT_MUTED};
}}
QPushButton#Ghost:hover {{ color: {TEXT}; border-color: {ACCENT}; }}

QPushButton#Remove {{
    background-color: transparent;
    border: none;
    color: {TEXT_MUTED};
    font-size: 16px;
    font-weight: 700;
    padding: 0px;
}}
QPushButton#Remove:hover {{ color: {DANGER}; }}

/* min-height is load-bearing, not cosmetic: without it a crowded panel lets
   Qt squeeze these controls below their own minimumSizeHint and the value
   inside becomes unreadable. */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {ACCENT};
    selection-color: #0B1120;
}}

QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: #0B1120;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{
    color: #5A6784;
    background-color: #131A28;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {ELEVATED};
    border: none;
    width: 16px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {ACCENT_DEEP};
}}

QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {ELEVATED};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_DEEP};
    outline: none;
    padding: 4px;
}}

/* The latent-size control is the one knob in the app that teaches by being
   dragged, so it gets a real track and a grabbable handle rather than the
   Fusion default, which is nearly invisible on this background. */
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT_DEEP};
    border: 1px solid {ACCENT_DEEP};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0px;
    border-radius: 8px;
    background-color: {ACCENT};
    border: 2px solid {BACKGROUND};
}}
QSlider::handle:horizontal:hover {{ background-color: {ACCENT_HOVER}; }}
QSlider::handle:horizontal:disabled {{ background-color: #3E4A66; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER_STRONG};
    background-color: {SURFACE_ALT};
}}
QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background-color: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_MUTED};
    padding: 8px 16px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {ACCENT};
    border-color: {BORDER};
    border-bottom-color: {SURFACE};
}}

/* The top-level workspace switch is a different kind of choice from the
   stage tabs underneath, so it reads as a mode selector, not a step. */
QTabWidget#Workspaces > QTabBar::tab {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-right: 6px;
    padding: 10px 26px;
    font-size: 14px;
    font-weight: 700;
    color: {TEXT_MUTED};
}}
QTabWidget#Workspaces > QTabBar::tab:hover {{
    color: {TEXT};
    border-color: {BORDER_STRONG};
}}
QTabWidget#Workspaces > QTabBar::tab:selected {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #0B1120;
}}
QTabWidget#Workspaces > QTabBar {{ qproperty-drawBase: 0; }}

QTableWidget {{
    background-color: {SURFACE_ALT};
    alternate-background-color: #1F293D;
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QHeaderView::section {{
    background-color: {ELEVATED};
    color: {TEXT_MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item:selected {{ background-color: {ACCENT_DEEP}; color: {TEXT}; }}

QProgressBar {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 6px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_DEEP}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QScrollArea {{ border: none; background: transparent; }}
QSplitter::handle {{ background-color: transparent; }}
QSplitter::handle:hover {{ background-color: {BORDER}; }}

QStatusBar {{ background-color: {SURFACE}; color: {TEXT_MUTED}; border-top: 1px solid {BORDER}; }}
QToolTip {{
    background-color: {ELEVATED};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    padding: 6px;
    border-radius: 6px;
}}
"""


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
