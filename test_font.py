import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
app = QApplication([])
font_id = QFontDatabase.addApplicationFont("/Users/masteryeeeee/Documents/New project/mediapipe-bridge/Font/XQJF.ttf")
print("Font ID:", font_id)
if font_id != -1:
    print("Families:", QFontDatabase.applicationFontFamilies(font_id))
