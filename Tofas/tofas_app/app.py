import os
from PyQt6 import QtWidgets, uic, QtGui, QtCore
from PyQt6.QtCore import QDir, QThread, QSettings
from PyQt6.QtWidgets import QMessageBox
from engine import engine_v3_parallel, engine_v3_single, engine_v3_grouping
from engine.engine_v3_single import list_devices
from engine.categorize import categorize_create_folder
import argparse
import time
import sys

# TODO more communication between engines DONE
# TODO select one of the cameras and run grayish detection on it, if detected run inference for all cameras with args.interval interval. DONE
# TODO device opened by other client error handlers (pop message notifi)
# TODO parallel inference
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/inovako/Desktop/TOFAS/tofas_model.engine")
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
    parser.add_argument('--filter-cam', type=str)
    parser.add_argument('--filter-expo', type=str)
    parser.add_argument('--master', type=int, default=1)
    parser.add_argument('--group-size', type=int, default=2)
    args = parser.parse_args()
    return args
args = parse_args()


# TODO more clearer EngineThread
class EngineThread(QThread):
    def run(self):
        if args.test_engine:
            if args.test:
                print(f"engine_v3 is running test")
                time.sleep(1)
                #engine_v3_parallel.run_test(args)
            else:
                engine_v3_parallel.run_engine(args)
        else:
            print(f"engine_v2 is running deployment")
            engine_v3_grouping.run_engine(args)
    def stop_engine_thread(self):
        if args.test_engine:
            if args.test:
                print(f"engine_v3 is running test")
                time.sleep(1)
                #engine_test.stop_engine()
            else:
                engine_v3_parallel.stop_engine()
        else:
            print(f"engine_v2 is running deployment")
            engine_v3_single.stop_engine()

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
        self.master_select = self.findChild(QtWidgets.QComboBox, 'master_select')
        self.master_select.addItems(list_devices(args))
        self.master_select.currentIndexChanged.connect(self.master_selected)
        self.filter_select_cam = self.findChild(QtWidgets.QComboBox, 'filter_select')
        self.filter_select_expo = self.findChild(QtWidgets.QComboBox, 'filter_select_expo')
        self.filter_select_cam.addItems(list_devices(args))
        self.filter_select_cam.currentIndexChanged.connect(self.filter_selection_cam)
        self.filter_select_expo.currentIndexChanged.connect(self.filter_selection_expo)
        self.image_name = self.findChild(QtWidgets.QLabel, 'image_name')
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
    
    def filter_selection_cam(self, cam_id):
        args.filter_cam = list_devices(args)[cam_id]
    
    def filter_selection_expo(self, cam_id):
        args.filter_expo = args.exposure_time[cam_id]

    def master_selected(self, cam_id):
        print(f"master selected with cam id: {cam_id}")
        args.master = cam_id

    def select_folder(self):
        run_id = len(os.listdir(args.out_dir))
        current_directory = args.out_dir + f"run_{run_id}/" + self.filter_select_cam.currentText() + "/" + self.filter_select_expo.currentText() + "/DET/"
        print(f"current dir {current_directory}")
        self.images = QDir(current_directory).entryList(['*.png', '*.jpg', '*.jpeg'], QDir.Filter.Files)
        self.images = [os.path.join(current_directory, img) for img in sorted(self.images)]
        self.index = 0
        self.display_image()

    def display_image(self):
        if self.images:
            print(f"setting imagename : {self.images[self.index].split('/')}")
            image_name = self.images[self.index].split('/')[-1]
            self.image_name.setText(image_name)
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
                categorize_create_folder(out_dir=args.out_dir, cams_name=list_devices(args), exposures=exposure_list)
                self.filter_select_expo.clear()
                self.filter_select_expo.addItems(map(str, args.exposure_time))
                self.ins_time_start = time.time()
                self.start_engine()
        else:
            QMessageBox.information(self, 'Exposure Error', 'Please Enter Valid Range of Exposure Times')

    def pop_up_message(self, error_name, error_text):
        QMessageBox.information(self, error_name, error_text)

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
            print(f"engine_v3 is stopping")
            self.engineThread.stop_engine_thread()
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

def pop_up_call(error_name, error_text):
    window.pop_up_message(error_name, error_text)


app = QtWidgets.QApplication(sys.argv)

window = Inovako()
window.show()

sys.exit(app.exec())