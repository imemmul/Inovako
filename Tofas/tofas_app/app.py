import os
from PyQt6 import QtWidgets, uic, QtGui, QtCore
from PyQt6.QtCore import QDir, QThread, QSettings
from PyQt6.QtWidgets import QMessageBox
from engine import engine_v2, engine_v3
import argparse
import time

# TODO more communication between engines
# TODO select one of the cameras and run grayish detection on it, if detected run inference for all cameras with args.interval interval.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/emir/Desktop/dev/Inovako/tensorrt_engines/tofas_model.engine")
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=30)
    parser.add_argument('--exposure-time', type=list, default=[10000, 50000])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument('--check-interval', type=int, default=5)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--test-engine', action="store_true")
    parser.add_argument('--master', type=int, default=1)
    args = parser.parse_args()
    return args
args = parse_args()

class EngineThread(QThread):
    def run(self):
        if args.test_engine:
            if args.test:
                print(f"engine_v3 is running test")
                time.sleep(1)
                engine_v3.run_test(args)
            else:
                engine_v3.run_engine(args)
        else:
            print(f"engine_v2 is running deployment")
            engine_v2.run_engine(args)

class Inovako(QtWidgets.QMainWindow):
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
        self.exposure_time = self.findChild(QtWidgets.QSpinBox, 'exposure_time')
        self.exposure_time_2 = self.findChild(QtWidgets.QSpinBox, 'exposure_time_2')
        self.check_freq = self.findChild(QtWidgets.QSpinBox, 'check_frequency')
        self.gray_thres = self.findChild(QtWidgets.QSpinBox, 'gray_thres')
        self.interval = self.findChild(QtWidgets.QDoubleSpinBox, 'interval')
        self.ins_time_widget = self.findChild(QtWidgets.QLabel, 'inspection_time')
        # Connect signals and slots
        self.backButton.clicked.connect(self.previous_image)
        self.forwardButton.clicked.connect(self.next_image)
        self.folderButton.clicked.connect(self.select_folder)
        self.buttonStart.clicked.connect(self.start_stop_engine)
        self.exposure_time.setMaximum(1000000)
        self.exposure_time_2.setMaximum(1000000)
        self.engine_running = False
        self.engineThread = None
        self.ins_time_start = 0
        self.ins_time_stop = 0
    
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

    def start_stop_engine(self):
        if self.check_freq.value() != 0:
            args.check_interval = self.check_freq.value()
        # print(f"check interval set to {int(args.check_interval)}")
        exposure_list = [int(self.exposure_time.value()), int(self.exposure_time_2.value())]
        args.gray_thres = self.gray_thres.value()
        args.interval = self.interval.value()
        # print(f"gray_thres : {args.gray_thres}")
        if 0 not in exposure_list:
            args.exposure_time = exposure_list
            # print(f"what is set exposure time {args.exposure_time}")
            if self.engine_running:
                self.ins_time_stop = time.time()
                self.stop_engine()
            else:
                self.ins_time_start = time.time()
                self.start_engine()
        else:
            QMessageBox.information(self, 'Exposure Error', 'Please Enter Valid Range of Exposure Times')


    def start_engine(self):
        self.engine_running = True
        self.backButton.hide()
        self.forwardButton.hide()
        self.folderButton.hide()
        self.buttonStart.setStyleSheet("background-color: red; color: black; font: 32px")
        self.buttonStart.setText("Durdur")
        self.engineThread = EngineThread()
        self.engineThread.start()

    def format_time(self, ins_time):
        hours, remainder = divmod(ins_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    
    def stop_engine(self):
        if self.engineThread is not None:
            if args.test_engine:
                print(f"engine_v2 is running")
                engine_v3.stop_engine()
            else:
                engine_v3.stop_engine()
            self.engineThread.quit()
            self.engineThread.wait()
            self.engineThread = None
        self.backButton.show()
        self.forwardButton.show()
        self.folderButton.show()
        self.engine_running = False
        self.buttonStart.setStyleSheet("background-color: rgb(142, 236, 186); color: black; font: 32px")
        self.buttonStart.setText("Baslat")
        ins_time = self.ins_time_stop - self.ins_time_start
        self.ins_time_widget.setText(self.format_time(ins_time=ins_time))

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    window = Inovako()
    window.show()

    sys.exit(app.exec())

