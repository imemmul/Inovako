from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QWidget, QLineEdit, QComboBox, QTableWidget, QDialog, QTableWidgetItem
from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets, QtGui, QtTest
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
import sys
from PIL.ImageQt import ImageQt as QtImage
from PIL import Image
import os
import cv2
import engine
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='./tofas_model.engine')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--out-dir', type=str, default='./output')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=35)
    parser.add_argument('--exposure-time', type=list, default=[10000, 50000])
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--check-interval', type=int, default=5)
    args = parser.parse_args()
    return args

class Run_Thread(QThread):
    
    def run(self):
        engine.run_engine(parse_args())

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("tofas_ui.ui", self)

        # Register

        # Label
        self.label_image = self.findChild(QLabel, "label_image")
        self.label_time = self.findChild(QLabel, "label_sure")

        #Signal slot

        # Button Register
        self.button_Baslat = self.findChild(QPushButton, "pushButton_baslat")
        self.button_Durdur = self.findChild(QPushButton, "pushButton_durdur")
        self.button_Geri = self.findChild(QPushButton, "pushButton_geri")
        self.button_ileri = self.findChild(QPushButton, "pushButton_ileri")
        self.button_sec = self.findChild(QPushButton, "pushButton_sec")

        self.Widget_Navigate = self.findChild(QWidget, "widget")
        # Widget Register
        #self.WidgetPredict = self.findChild(QWidget, "widget_Predict")

        # Button Register
        #self.button_Enter = self.findChild(QPushButton, "pushButton_Enter")

        # LineEdit Register
        #self.lineEdit_UserName = self.findChild(QLineEdit, "lineEdit_UserName")

        
        self.button_Durdur.clicked.connect(self.durdur)
        self.button_sec.clicked.connect(self.resimSec)

        self.button_Geri.clicked.connect(self.geri)
        self.button_ileri.clicked.connect(self.ileri)



        self.button_Baslat.clicked.connect(self.baslat)

        
        # InVisible
        self.button_Durdur.setHidden(True)
        self.max = 0
        # Show the app
        self.show()

    def baslat(self):
        self.Widget_Navigate.setHidden(True)
        self.button_Durdur.setHidden(False) # making visilbe ????
        self.baslatma = Run_Thread()
        self.baslatma.start()

    def start_stop(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(1000)
            

    def durdur(self):
        self.Widget_Navigate.setHidden(False)
        self.button_Durdur.setHidden(True) 
        engine.stop_engine()
        asdf = "Durdu"
        self.label_image.setText(asdf)
        

    def resimSec(self):
        self.fname = QFileDialog.getOpenFileName(
            self, "Open File", "output", "ALL Files (*);;PNG Files(*.png);;Jpg Files(*.jpg")
        # ileri buton icin
        self.dizin_yolu = os.path.dirname(self.fname[0])
        a = (self.fname[0].split('/')[-1]).split('.')[0]
        self.b = str(int(a) + 1)
        self.open_image(self.fname[0])

    def ileri(self):
        acilacak = self.dizin_yolu + '/' + self.b + '.jpg'
        print(acilacak)
        self.b = str(int(self.b) + 1)  # for next image
        if int(self.b) > int(self.max):
            self.max = self.b
        self.open_image(acilacak)

    def geri(self):
        acilacak = self.dizin_yolu + '/' + self.b + '.jpg'
        print(acilacak)
        self.b = str(int(self.b) - 1)
        if int(self.b) == 0:
            self.b = str(self.max)
        print(self.b)
        self.open_image(acilacak)

    def open_image(self, dosya_yolu):
        print("Dosya yolu", dosya_yolu)
        try:
            image = cv2.imread(dosya_yolu)
            if image is not None:
                display_photo = image
                display_photo = cv2.resize(
                    display_photo, (500, 500), interpolation=cv2.INTER_AREA)
                rgb_image = cv2.cvtColor(display_photo, cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(rgb_image).convert('RGBA')
                qpixmap = QtGui.QPixmap.fromImage(QtImage(PIL_image))
                self.label_image.setPixmap(qpixmap)
            else:
                print("Image not found or couldn't be read.")
        except Exception as e:
            print(e)
            print("Resim Secilmedi")

# Inıtialize the App
app = QApplication(sys.argv)
UIWindow = UI()

app.exec()


