import sys
from PySide6.QtWidgets import QApplication, QGroupBox, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

app = QApplication([])
w = QWidget()
l = QVBoxLayout(w)

g = QGroupBox("Input Title")
g.setStyleSheet("""
QGroupBox {
    background-color: #FFDAB9;
    border-radius: 8px;
    border: none;
    padding-top: 25px;
}
QGroupBox::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 10px;
    top: 5px;
    color: #222;
    font-weight: bold;
}
""")
l.addWidget(g)
w.show()
# We can't really see it without GUI, but we ensure it parses without warning.
