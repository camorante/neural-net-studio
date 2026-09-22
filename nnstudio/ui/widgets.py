"""Small shared building blocks for the panels."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Card(QFrame):
    """A titled rounded container."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self._title = QLabel(title)
        self._title.setObjectName("CardTitle")
        layout.addWidget(self._title)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        layout.addLayout(self.body)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self.body.addWidget(widget, stretch)


def scrollable(content: QWidget) -> QScrollArea:
    """Wrap a tall panel so a short window scrolls instead of squeezing.

    Without this, Qt distributes the shortfall by shrinking spin boxes and
    combos below their own minimum height, which clips the value inside them.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    # AsNeeded, never AlwaysOff: if the content really is wider than the column
    # on some display, a scrollbar is honest and clipping is not.
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(content)
    return area


def compact_combo(combo: QComboBox, visible_chars: int = 16) -> QComboBox:
    """Stop a combo's longest entry from dictating the panel's width.

    By default QComboBox reports a minimum width wide enough to show its
    longest item, so one descriptive entry can force a whole sidebar wider
    than the window - and the layout then pays for it by squeezing every
    control's height. Capping the contents length breaks that chain; the
    popup still shows each entry in full.
    """
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(visible_chars)
    combo.setToolTip(combo.currentText())
    combo.currentTextChanged.connect(combo.setToolTip)
    return combo


def hint(text: str) -> QLabel:
    """Muted explanatory text that wraps and never dictates the panel width.

    A word-wrapped QLabel still reports a sizeable minimum width, which is
    enough to push a narrow sidebar wider than its column. Ignored horizontal
    policy takes it out of the width calculation entirely; it still reports
    the height it needs for the width it is given.
    """
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    return label


def section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionLabel")
    return label


def divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#2C3852; background:#2C3852; max-height:1px;")
    return line


def stretch_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Subtle")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setWordWrap(True)
    return label
