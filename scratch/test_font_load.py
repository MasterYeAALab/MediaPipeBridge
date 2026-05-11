
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont

def test_font():
    app = QApplication([])
    # Go up one level from scratch to root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    font_path = os.path.join(root_dir, "Font", "XQJF.ttf")
    print(f"Checking path: {font_path}")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        print(f"Font ID: {font_id}")
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            print(f"Families: {families}")
            if families:
                font = QFont(families[0])
                print(f"Selected Font Family: {font.family()}")
        else:
            print("Failed to load font")
    else:
        print("Font file not found")

if __name__ == "__main__":
    test_font()
