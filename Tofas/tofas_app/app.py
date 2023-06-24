import os
from PyQt6 import QtWidgets, uic, QtGui, QtCore
from PyQt6.QtCore import QDir, QThread
from engine import engine
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/inovako/Inovako/emir_workspace/tensorrt_engines/tofas_engine/tofas_model.engine")
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=35)
    parser.add_argument('--exposure-time', type=list, default=[10000, 50000])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--check-interval', type=int, default=5)
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()
    return args

class EngineThread(QThread):
    def run(self):
        engine.run_test(parse_args())

class ImageViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        uic.loadUi('tofas_ui.ui', self)

        # Initialize attributes
        self.images = []
        self.index = 0

        # Find widgets
        self.imageLabel = self.findChild(QtWidgets.QLabel, 'label_image')
        self.backButton = self.findChild(QtWidgets.QPushButton, 'pushButton_geri')
        self.forwardButton = self.findChild(QtWidgets.QPushButton, 'pushButton_ileri')
        self.folderButton = self.findChild(QtWidgets.QPushButton, 'pushButton_sec')
        self.buttonStart = self.findChild(QtWidgets.QPushButton, "pushButton_baslat")
        self.buttonStop = self.findChild(QtWidgets.QPushButton, "pushButton_durdur")

        # Connect signals and slots
        self.backButton.clicked.connect(self.previous_image)
        self.forwardButton.clicked.connect(self.next_image)
        self.folderButton.clicked.connect(self.select_folder)
        self.buttonStart.clicked.connect(self.start_engine)
        self.buttonStop.clicked.connect(self.stop_engine)
        
        self.engineThread = None
    def select_folder(self):
        current_directory = os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', current_directory)

        if folder:
            self.images = QDir(folder).entryList(['*.png', '*.jpg', '*.jpeg'], QDir.Filter.Files)
            self.images = [os.path.join(folder, img) for img in self.images]
            self.index = 0
            self.display_image()

    def display_image(self):
        if self.images:
            pixmap = QtGui.QPixmap(self.images[self.index])
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size()))

    def previous_image(self):
        if self.images and self.index > 0:
            self.index -= 1
            self.display_image()

    def next_image(self):
        if self.images and self.index < len(self.images) - 1:
            self.index += 1
            self.display_image()
    def start_engine(self):
        self.engineThread = EngineThread()
        self.engineThread.start()
        
    def stop_engine(self):
        if self.engineThread is not None:
            engine.stop_engine()
            self.engineThread.quit()
            self.engineThread.wait()
            self.engineThread = None

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    window = ImageViewer()
    window.show()

    sys.exit(app.exec())

