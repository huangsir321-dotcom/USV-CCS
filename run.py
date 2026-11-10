from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from ui.main_window import MainWindow


def main():
    app = QApplication([])
    app.setApplicationName("USV Collision Avoidance Simulator")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
