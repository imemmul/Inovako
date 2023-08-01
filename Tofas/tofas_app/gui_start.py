import sys
import os
import random
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal, QRectF, QSizeF, pyqtProperty
from PyQt6.QtGui import QPixmap, QFont, QPainter, QPen, QColor, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QSpinBox, QDoubleSpinBox, QFileDialog, QFrame, QLineEdit, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QTabBar, QStackedWidget, QListWidget, QListWidgetItem, QRadioButton, QButtonGroup, QSizePolicy, QSpacerItem 
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QListWidget, QStyleOptionViewItem, QStyle
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QApplication
from PyQt6.QtGui import QPixmap
import argparse
from PyQt6 import QtWidgets, QtCore
from engine.categorize import categorize_create_folder
import time
from functools import partial
from engine import engine_v3_grouping
from engine.engine_v3_grouping import list_devices, update_status

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--engine', type=str, default="/home/inovako/Desktop/Inovako/tensorrt_engines/tofas_model.engine")
    parser.add_argument('--engine', type=str, default="/home/inovako/Desktop/Inovako/tensorrt_engines/tofas_model_v4.engine")
    parser.add_argument('--out-dir', type=str, default='./output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=10)
    parser.add_argument('--exposure-time', type=int, default=850)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=float, default=0.25)
    parser.add_argument('--check-interval', type=int, default=1)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--test-engine', action="store_true")
    parser.add_argument('--filter-cam', type=str)
    parser.add_argument('--filter-expo', type=str)
    parser.add_argument('--master', type=int, default=0)
    parser.add_argument('--group-size', type=int, default=2) # TODO change this with 2
    parser.add_argument('--wait-time', type=int, default=10)
    args = parser.parse_args()
    return args
args = parse_args()



class WorkerThread(QThread):
    finished = pyqtSignal()

    def run(self):
        # If test_engine flag is True, run test engine, otherwise run normal engine
        # If test flag is True, run in test mode, otherwise run normally
        # if args.test_engine:
        print("running engine_v3_grouping")
        engine_v3_grouping.run_engine(args)
        
    def stop_engine_thread(self):
        # if args.test_engine:
        print("stopping engine_v3_grouping")
        engine_v3_grouping.stop_engine()

class CustomListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.current_index = 0

        # Set the style sheet with both background color and border properties
        #self.setStyleSheet("QListView { background-color: rgba(30, 30, 30, 255); color: white; border: 2px solid black;}")

    def paintEvent(self, event):
        # Custom background color
        painter = QPainter(self.viewport())
        #painter.fillRect(event.rect(), QColor(30, 30, 30, 255))  # Change color values as per your preference

        # Call the original paintEvent to draw items
        super().paintEvent(event)
        
class PhotoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hovered = False
        self.hover_x = 0
        self.hover_y = 0
        self.hover_text = ""
        self.drawing_enabled = False  # Variable to track if drawing is enabled

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if self.drawing_enabled:
            self.hover_x = event.x()
            self.hover_y = event.y()
            self.update()
            self.clicked.emit(self.hover_x, self.hover_y)

    @pyqtProperty(bool)
    def hovered(self):
        return self._hovered

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(255, 0, 0, 200), 2))
            painter.setBrush(QColor(0, 0, 0, 150))
            text_rect = painter.boundingRect(QRectF(QPointF(self.hover_x, self.hover_y - 25), QSizeF(100, 25)), Qt.AlignmentFlag.AlignCenter, self.hover_text)
            painter.drawRect(text_rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.hover_text)

    def enable_drawing(self):
        self.drawing_enabled = True

    def disable_drawing(self):
        self.drawing_enabled = False


# TODO placeholder parca display edecek, photo_frame buyuk ana parcayi display edecek
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.engine_running = False
        self.status_check_timer = QtCore.QTimer()
        self.status_check_timer.timeout.connect(self.check_engine_status)
        self.status_check_timer.start(200) # 100ms 0.1 sec check status
        self.ins_time_start = None
        self.ins_time_stop = None

        icon_path = "logo/inovako_logo_negatif.png"
        self.setWindowIcon(QIcon(icon_path))

        self.setWindowTitle("STAMPIX 1.0.0")
        # self.setGeometry(0, 0, 1920, 1080)
        # self.setFixedSize(1920, 1080)
        # self.showFullScreen()

        self.setContentsMargins(30, 0, 0, 0)
        self.setStyleSheet("background-color: rgba(10, 21, 39, 255); color: white;")


        self.title_label = None
        self.current_image_info = []  # Geçerli resmin etiketlerini saklayacağımız liste

        self.current_photo_index = 0
        self.detection_history = []  # Tüm fotoğraflardaki Hole değerlerini burada tutacağız
        self.photo_directory = f"/home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/output/run_{len(os.listdir(args.out_dir))}/Basler a2A2600-64umBAS (40359004)/{args.exposure_time}/DET/" # TODO will be updated in check_engine_status
        self.crack_directoty = None
        self.photo_paths = []
        self.part_name = os.path.basename(self.photo_directory)
        self.part_id = random.randint(1, 1000)
        self.hole_count = 0

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Ana pencerenin arka planını belirle
        main_widget.setStyleSheet("background-color: rgba(10,21,39,255); color: white;")

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        left_panel = QWidget(self)
        left_panel.setFixedWidth(500)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        head_layout = QHBoxLayout()
        main_layout.addLayout(head_layout)

        label_text = "<div>"
        label_text += "<span style='font-size: 24pt; font-weight: bold;'> STAMPAI. </span> "
        label_text += "<span style='font-size: 18pt;'>AI for the Inspection of Pressed Sheet Metal</span>"
        label_text += "</div>"
        self.left_aligned_label = QLabel(label_text)
        head_layout.addWidget(self.left_aligned_label)

        logo_label = QLabel(self)
        logo_pixmap = QPixmap("./logo/inovako_logo_300.png")
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        head_layout.addWidget(logo_label)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Live")
        self.tab_bar.addTab("Defects")
        self.tab_bar.addTab("Settings")
        self.tab_bar.setFixedWidth(1000)
        self.tab_bar.setFixedHeight(50)
        self.tab_bar.currentChanged.connect(self.on_tab_changed)

        font = QFont()
        font.setPointSize(14)
        self.tab_bar.setFont(font)

        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.setFont(QFont("Arial", 14))
        self.start_stop_button.setFixedSize(200, 50)
        self.start_stop_button.setStyleSheet("background-color : rgba(229,30,73,255); border: 2px solid black;")  # Siyah kenarlık eklemek için stil yönlendirmesi
        self.start_stop_button.clicked.connect(self.start_stop)
        tab_layout = QHBoxLayout()
        tab_layout.addWidget(self.tab_bar)
        tab_layout.addSpacerItem(QSpacerItem(40, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        tab_layout.addWidget(self.start_stop_button)

        main_layout.addLayout(tab_layout)

        self.tab_bar.setTabEnabled(1, False)  # "Defects" tab
        #self.tab_bar.setTabEnabled(2, False)  # "Settings" tab

        tab_layout.setContentsMargins(0, 0, 30, 0)

        # Add the tab_layout to the main_layout
        main_layout.addLayout(tab_layout)

        self.left_bottom_widget = QWidget(self)
        self.left_bottom_layout = QVBoxLayout()
        self.left_bottom_widget.setLayout(self.left_bottom_layout)

        left_section_layout = QHBoxLayout()

        self.previous_photo_button = QPushButton()
        self.set_logo(self.previous_photo_button, 'yontus/sol.png')
        self.previous_photo_button.clicked.connect(self.go_back)
        self.previous_photo_button.setFixedSize(75, 252)
        self.previous_photo_button.setStyleSheet("background-color: rgba(229,30,73,255);border: 2px solid black;")
        left_section_layout.addWidget(self.previous_photo_button)

        # Add a 100x100 photo placeholder
        self.photo_placeholder = QLabel(self)
        self.photo_placeholder.setFixedSize(300, 250)
        self.photo_placeholder.setStyleSheet("background-color: black; color: white; border: 1px solid black;")
        left_section_layout.addWidget(self.photo_placeholder)

        self.change_photo_button = QPushButton()
        self.set_logo(self.change_photo_button, 'yontus/sag.png')
        self.change_photo_button.setFixedSize(75, 252)
        self.change_photo_button.clicked.connect(self.change_photo)
        self.change_photo_button.setStyleSheet("background-color: rgba(229,30,73,255);border: 2px solid black;")
        left_section_layout.addWidget(self.change_photo_button)

        self.left_bottom_layout.addLayout(left_section_layout)
        
        right_panel = QWidget(self)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        

        self.stacked_widget = QStackedWidget()
        self.setup_canli_page()
        self.setup_kul_page()
        self.setup_ayarlar_page()
        main_layout.addWidget(self.stacked_widget)

        self.load_photos()

        # self.worker_thread = WorkerThread()
        # self.worker_thread.finished.connect(self.on_thread_finished)
        update_status(2)

    def on_thread_finished(self):
        print("Thread işlemi tamamlandı!")
        # İşlem sonucunu ana thread'de kullanabilirsiniz
        # Ancak arayüz bileşenlerine dokunmayı unutmayın
    def start_stop(self):
        # print(f"gray_thres : {args.gray_thres}")
        if args.exposure_time != 0:
            print(f"exposure time set to {args.exposure_time}")
            # print(f"what is set exposure time {args.exposure_time}")
            if self.engine_running:
                self.start_stop_button.clicked.disconnect()
                self.start_stop_button.clicked.connect(lambda: self.pop_status_engine("Engine is stopping", "Please wait, engine is stopping."))
                self.ins_time_stop = time.time()
                # update_status(command=1) # stopping process signal
                self.stop_engine()
                # time.sleep(3)
                # update_status(2) # engine started and ready to action from now on any interaction is able to stop the engine
            else:
                self.start_stop_button.clicked.disconnect()
                self.start_stop_button.clicked.connect(lambda: self.pop_status_engine("Engine is running", "Please wait, engine is running."))
                categorize_create_folder(out_dir=args.out_dir, cams_name=list_devices(args), exposure=args.exposure_time)
                self.ins_time_start = time.time()
                # update_status(command=0) # start process signal
                self.start_engine()
                time.sleep(2)
                # update_status(2) # engine started and ready to action from now on any interaction is able to stop the engine
                
        else:
            QMessageBox.information(self, 'Exposure Error', 'Please Enter Exposure Time')

    def start_engine(self):
        self.engine_running = True
        # self.backButton.hide()
        # self.forwardButton.hide()
        # self.folderButton.hide()
        self.start_stop_button.setText("Stop")
        self.engineThread = WorkerThread()
        self.engineThread.start()

    def pop_status_engine(self, warning, warning_desc):
        QMessageBox.information(self, warning, warning_desc)

    def check_engine_status(self): # TODO more safer status checker needed
        # self.update_display()
        with open('./status.txt', 'r') as f:
            status = f.read().strip()
            # print(f"checking status of engine got: {status}")
            if status:
                if int(status) == 3: # if stop signal interrupt comes from the engine
                    self.ins_time_stop = time.time()
                    self.stop_engine()
                    self.pop_status_engine(warning="Timed Out", warning_desc=f"It's been {args.wait_time} seconds, engine is stopped.")
                elif int(status) == 1 or int(status) == 0: # if it is in action
                    try:
                        self.start_stop_button.setEnabled(False)
                        # self.start_stop_button.clicked.disconnect()
                    except Exception as e:
                        print(f"face with error : {e}")
                else: # if status is 2
                    try:
                        self.start_stop_button.setEnabled(True)
                        self.start_stop_button.clicked.disconnect()
                    except Exception as e:
                        print(f"face with error while disconnecting the button: {e}")
                    self.start_stop_button.clicked.connect(self.start_stop)
    def stop_engine(self):
        if self.engineThread is not None:
            print(f"engine_v3 is stopping")
            self.engineThread.stop_engine_thread()
            self.engineThread.quit()
            self.engineThread.wait()
            self.engineThread = None
        # self.backButton.show()
        # self.forwardButton.show()
        # self.folderButton.show()
        self.engine_running = False
        self.start_stop_button.setText("Start")
        # ins_time = self.ins_time_stop - self.ins_time_start
        # self.ins_time_widget.setText(self.format_time(ins_time=ins_time)) # TODO SELIM muayene suresi yazan bir yer eklenmesi gerekiyor ve bu hesap ona set edilecek.
    def show_page(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def on_tab_changed(self, index):
        self.show_page(index)

    def setup_canli_page(self):
        canli_widget = QWidget(self)
        canli_layout = QHBoxLayout()
        canli_widget.setLayout(canli_layout)

        left_panel_widget = QWidget(self)
        left_panel_widget.setFixedWidth(500)

        left_panel_layout = QVBoxLayout()
        left_panel_widget.setLayout(left_panel_layout)


        left_panel_layout.addSpacerItem(QSpacerItem(20, 10))

        # PART label
        part_label_text = "<span style='font-size: 14pt; font-weight: bold;'>PART:</span> " + str(self.part_name)
        part_label = QLabel(part_label_text)
        part_label.setStyleSheet("color: white;")
        part_label.setOpenExternalLinks(True)
        left_panel_layout.addWidget(part_label)

        # PART ID label
        part_id_label_text = "<span style='font-size: 14pt; font-weight: bold;'>PART ID:</span> " + "20230717.198"  # str(self.part_id)
        part_id_label = QLabel(part_id_label_text)
        part_id_label.setStyleSheet("color: white;")
        part_id_label.setOpenExternalLinks(True)
        left_panel_layout.addWidget(part_id_label)

        left_panel_layout.addSpacerItem(QSpacerItem(20, 50))


        # Create a QFrame to hold the new label and style it with a filled box
        boxed_label_frame = QFrame()
        boxed_label_frame.setStyleSheet("background-color: rgba(0, 0, 0, 150);")  # Remove border

        # Create a QLabel for the new label and set its style
        boxed_label = QLabel("Defect History")
        boxed_label.setStyleSheet("color: white; text-align: center;")  # Center the text
        
        # Create a QVBoxLayout to place the new label inside the frame
        boxed_label_layout = QVBoxLayout()
        boxed_label_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the content
        boxed_label_layout.addWidget(boxed_label)
        boxed_label_frame.setLayout(boxed_label_layout)

        # Add the filled box with the new label to the left panel layout
        left_panel_layout.addWidget(boxed_label_frame)


        # Create the QListWidget
        self.label_list = CustomListWidget()
        self.label_list.setStyleSheet("QListView { background-color: black; color: white; border: 2px solid black;}")
        self.label_list.setFixedSize(375, 150) # Set the desired height for the QListWidget

        # Create custom QListWidgetItems as buttons with a height of 50
        hole_item = QListWidgetItem("Hole")
        hole_item.setSizeHint(QSize(100, 50))

        crack_item = QListWidgetItem("Crack")
        crack_item.setSizeHint(QSize(100, 50))

        all_item = QListWidgetItem("All")
        all_item.setSizeHint(QSize(100, 50))

        # Add the custom QListWidgetItems to the list
        self.label_list.addItem(hole_item)
        self.label_list.addItem(crack_item)
        self.label_list.addItem(all_item)

        
        # Add the QListWidget to the layout
        
        left_panel_layout.addWidget(self.label_list)
        
        self.label_list.itemClicked.connect(self.on_label_list_item_clicked)

        self.prev_list_button = QPushButton()
        self.set_logo(self.prev_list_button, 'yontus/ust.png')
        self.prev_list_button.clicked.connect(self.show_previous_list)
        self.prev_list_button.setFixedSize(75, 75)
        self.prev_list_button.setStyleSheet("background-color: rgba(229,30,73,255);border: 2px solid black;")

        self.next_list_button = QPushButton()
        self.set_logo(self.next_list_button, 'yontus/alt.png')
        self.next_list_button.clicked.connect(self.show_next_list)
        self.next_list_button.setFixedSize(75, 75)
        self.next_list_button.setStyleSheet("background-color: rgba(229,30,73,255);border: 2px solid black;")


        # Create a layout for the buttons (Previous List and Next List)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_list_button)
        button_layout.addWidget(self.next_list_button)


        # Create a vertical layout for the buttons
        vertical_button_layout = QVBoxLayout()
        vertical_button_layout.addWidget(self.prev_list_button)
        vertical_button_layout.addWidget(self.next_list_button)

        # Create a horizontal layout for the buttons and label_list
        buttons_and_list_layout = QHBoxLayout()
        buttons_and_list_layout.addWidget(self.label_list)         # Add the label_list to the horizontal layout
        buttons_and_list_layout.addLayout(vertical_button_layout)  # Add the vertical layout with buttons to the horizontal layout

        # Add the buttons and label_list layout to the left panel layout
        left_panel_layout.addLayout(buttons_and_list_layout)
        # Add the buttons layout to the left panel layout
        left_panel_layout.addLayout(button_layout)
        left_panel_layout.addStretch(1)
        left_panel_layout.addWidget(self.left_bottom_widget)
        # Add an expanding spacer to push the content to the top
        
        


        right_section_widget = QWidget(self)
        right_section_layout = QVBoxLayout()
        right_section_widget.setLayout(right_section_layout)



        self.photo_frame = QFrame()
        self.photo_frame.setFrameShape(QFrame.Shape.Panel)
        self.photo_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.photo_frame.setStyleSheet("background-color: black; color: white")
        self.photo_layout = QVBoxLayout()
        self.photo_frame.setLayout(self.photo_layout)
        self.photo_label = QLabel()
        self.photo_label.setStyleSheet("background-color: black; color: white; border: 2px solid #3a7ca5;")

        self.photo_layout.addWidget(self.photo_label)


        right_section_layout.addWidget(self.photo_frame)

        canli_layout.addWidget(left_panel_widget)
        canli_layout.addWidget(right_section_widget)

        self.stacked_widget.addWidget(canli_widget)

    
    def handle_list_item_clicked(self, item):
        selected_label = item.text()
        if selected_label == "Hole":
            self.draw_holes()
        elif selected_label == "Crack":
            self.draw_cracks()
        elif selected_label == "All":
            self.draw_all()


    def show_previous_list(self):
        current_index = self.label_list.currentIndex()
        if current_index.row() > 0:
            self.label_list.setCurrentRow(current_index.row() - 1)
            self.handle_list_item_clicked(self.label_list.currentItem())

    def show_next_list(self):
        current_index = self.label_list.currentIndex()
        if current_index.row() < self.label_list.count() - 1:
            self.label_list.setCurrentRow(current_index.row() + 1)
            self.handle_list_item_clicked(self.label_list.currentItem())

    def on_label_list_item_clicked(self, item):
        self.handle_list_item_clicked(item)

    def on_label_list_item_clicked(self, item):
        selected_label = item.text()
        if selected_label == "Hole":
            self.draw_holes()
        elif selected_label == "Crack":
            self.draw_cracks()
        elif selected_label == "All":
            self.draw_all()

    
    def draw_holes(self):
        self.clear_drawings()
        for info in self.current_image_info:
            class_id = int(info[1])
            x = int(info[2])
            y = int(info[3])
            if class_id == 0:  # Only draw cracks for class_id 1
                self.draw_circle(x, y, class_id)

    def draw_cracks(self):
        self.clear_drawings()
        for info in self.current_image_info:
            class_id = int(info[1])
            x = int(info[2])
            y = int(info[3])
            if class_id == 1:  # Only draw cracks for class_id 1
                self.draw_circle(x, y, class_id)

    def draw_all(self):
        self.clear_drawings()
        for info in self.current_image_info:
            class_id = int(info[1])
            x = int(info[2])
            y = int(info[3])
            self.draw_circle(x, y, class_id)

    def setup_kul_page(self):
        kul_widget = QWidget(self)
        kul_layout = QVBoxLayout()
        kul_widget.setLayout(kul_layout)

        kul_content_label = QLabel("This is the KUL page.")
        kul_content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        kul_layout.addWidget(kul_content_label)

        kul_widget.setStyleSheet("background-color: rgba(255, 255, 255, 255); color: black;")

        self.stacked_widget.addWidget(kul_widget)

   

    def setup_ayarlar_page(self):
        ayarlar_widget = QWidget(self)
        ayarlar_layout = QHBoxLayout()  # Main layout for the "Settings" page
        ayarlar_widget.setLayout(ayarlar_layout)
        ayarlar_widget.setStyleSheet("background-color: rrgba(10, 21, 39, 255); color: black;")

        # Add the photo frame (left section)
        photo_frame = QFrame()
        photo_frame.setFixedWidth(1500)  # Set the width of the photo_frame to 1000
        photo_frame.setFrameShape(QFrame.Shape.Panel)
        photo_frame.setFrameShadow(QFrame.Shadow.Raised)
        photo_frame.setStyleSheet("background-color: black; color: white")
        photo_layout = QVBoxLayout()
        photo_frame.setLayout(photo_layout)

        # Placeholder photo
        photo_placeholder = QLabel(self)
        photo_placeholder.setFixedSize(300, 250)
        photo_placeholder.setStyleSheet("background-color: black; color: white; border: 1px solid black;")
        photo_layout.addWidget(photo_placeholder)

        # Add the photo frame to the left side
        ayarlar_layout.addWidget(photo_frame)

        # Create a layout for the right section (other widgets)
        right_section_layout = QVBoxLayout()

        # Exposure Time SpinBox
        exposure_time_label = QLabel("Exposure Time:")
        exposure_time_label.setStyleSheet("color: white;")  # Set text color to white
        exposure_time_spinbox = QSpinBox()
        # Set appropriate range for exposure time (adjust min and max values as needed)
        exposure_time_spinbox.setRange(10, 1000000)
        right_section_layout.addWidget(exposure_time_label)
        right_section_layout.addWidget(exposure_time_spinbox)

        # Check Frequency DoubleSpinBox
        check_frequency_label = QLabel("Check Frequency:")
        check_frequency_label.setStyleSheet("color: white;")  # Set text color to white
        check_frequency_spinbox = QDoubleSpinBox()
        # Set appropriate range for check frequency (adjust min, max, and step values as needed)
        check_frequency_spinbox.setRange(0.1, 10.0)
        check_frequency_spinbox.setSingleStep(0.1)
        right_section_layout.addWidget(check_frequency_label)
        right_section_layout.addWidget(check_frequency_spinbox)

        # Capture Frequency DoubleSpinBox
        capture_frequency_label = QLabel("Capture Frequency:")
        capture_frequency_label.setStyleSheet("color: white;")  # Set text color to white
        capture_frequency_spinbox = QDoubleSpinBox()
        # Set appropriate range for capture frequency (adjust min, max, and step values as needed)
        capture_frequency_spinbox.setRange(0.1, 10.0)
        capture_frequency_spinbox.setSingleStep(0.1)
        right_section_layout.addWidget(capture_frequency_label)
        right_section_layout.addWidget(capture_frequency_spinbox)

        # Gray Threshold SpinBox
        gray_threshold_label = QLabel("Gray Threshold:")
        gray_threshold_label.setStyleSheet("color: white;")  # Set text color to white
        gray_threshold_spinbox = QSpinBox()
        # Set appropriate range for gray threshold (adjust min and max values as needed)
        gray_threshold_spinbox.setRange(0, 255)
        right_section_layout.addWidget(gray_threshold_label)
        right_section_layout.addWidget(gray_threshold_spinbox)
        exposure_time_spinbox.valueChanged.connect(partial(self.update_args, widget_name="exposure_time"))
        gray_threshold_spinbox.valueChanged.connect(partial(self.update_args, widget_name="gray_thres"))
        check_frequency_spinbox.valueChanged.connect(partial(self.update_args, widget_name="check_freq"))
        capture_frequency_spinbox.valueChanged.connect(partial(self.update_args, widget_name="capture_freq"))
        # Add the right section layout to the main layout
        ayarlar_layout.addLayout(right_section_layout)

        self.stacked_widget.addWidget(ayarlar_widget)

    def update_args(self, value, widget_name):
        if widget_name == "exposure_time":
            args.exposure_time = value
        elif widget_name == "gray_thres":
            args.gray_thres = value
        elif widget_name == "capture_freq":
            args.interval = value 
        else:
            args.check_interval = value
        print(f"setting {widget_name} = {value}")
    def load_photos(self):
        self.photo_paths = []
        self.photo_directory = f"/home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/output/run_{len(os.listdir(args.out_dir))}/Basler a2A2600-64umBAS (40359004)/{args.exposure_time}/DET/"
        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

        for file_name in os.listdir(self.photo_directory):
            if file_name.lower().endswith(".jpg"):
                file_path = os.path.join(self.photo_directory, file_name)
                self.photo_paths.append(file_path)

    def change_photo(self):
        self.load_photos()
        if len(self.photo_paths) == 0:
            return  # Hiç fotoğraf yoksa işlemi iptal et

        # Eski fotoğrafın Hole değerlerini güncel resim bilgilerini kullanarak tut
        if self.current_image_info:
            for info in self.current_image_info:
                class_id = int(info[1])
                x = int(info[2])
                y = int(info[3])
                self.detection_history.append(f"Hole {self.hole_count + 1}: (X: {x}, Y: {y})")
                self.hole_count += 1
        self.current_photo_index += 1
        if self.current_photo_index >= len(self.photo_paths):
            self.current_photo_index = 0

        # Resim değiştirildiğinde, güncel resim bilgilerini alıyoruz
        self.current_image_info = self.get_image_info()

        # Reset the detection history list here
        self.detection_history = []

        file_path = self.photo_paths[self.current_photo_index]
        self.photo_pixmap = QPixmap(file_path, )
        self.photo_pixmap = self.photo_pixmap.scaled(self.photo_label.size())
        self.update_photo()

        # Güncel resim bilgilerine göre otomatik olarak kırmızı yuvarlakları çiz
        self.auto_draw_circles()

         # Get the photo_label from the photo_layout
        photo_label = self.photo_layout.itemAt(0).widget()
        if isinstance(photo_label, QLabel):
            # Set the photo_pixmap to the photo_placeholder
            self.photo_label.pixmap()
            self.photo_placeholder.setPixmap(photo_label.pixmap())
            self.photo_placeholder.setScaledContents(True)

    def go_back(self):
        if len(self.photo_paths) == 0:
            return  # Hiç fotoğraf yoksa işlemi iptal et

        # Eski fotoğrafın Hole değerlerini güncel resim bilgilerini kullanarak tut
        if self.current_image_info:
            for info in self.current_image_info:
                class_id = int(info[1])
                x = int(info[2])
                y = int(info[3])
                self.detection_history.append(f"Hole {self.hole_count + 1}: (X: {x}, Y: {y})")
                self.hole_count += 1

        self.current_photo_index -= 1
        if self.current_photo_index < 0:
            self.current_photo_index = len(self.photo_paths) - 1

        # Resim değiştirildiğinde, güncel resim bilgilerini alıyoruz
        self.current_image_info = self.get_image_info()

        # Reset the detection history list here
        self.detection_history = []

        file_path = self.photo_paths[self.current_photo_index]
        self.photo_pixmap = QPixmap(file_path)
        self.photo_pixmap = self.photo_pixmap.scaled(self.photo_label.size())
        self.update_photo()

        # Güncel resim bilgilerine göre otomatik olarak kırmızı yuvarlakları çiz
        self.auto_draw_circles()

         # Get the photo_label from the photo_layout
        photo_label = self.photo_layout.itemAt(0).widget()
        if isinstance(photo_label, QLabel):
            # Set the photo_pixmap to the photo_placeholder
            self.photo_placeholder.setPixmap(photo_label.pixmap())

    def update_photo(self):
        if self.photo_pixmap:
            self.photo_layout.removeWidget(self.photo_frame.findChild(QLabel))
            self.photo_label = QLabel()
            self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.photo_placeholder.setScaledContents(True)
            self.photo_label.setScaledContents(True)
            f"/home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/output/run_{len(os.listdir(args.out_dir))}/Basler a2A2600-64umBAS (4036004)/{args.exposure_time}/DET/"
            self.photo_label.setPixmap(self.photo_pixmap.scaled(self.photo_label.size()))
            self.photo_layout.addWidget(self.photo_label)

    def prompt_draw_coordinates(self):
        input_dialog = DrawCoordinatesInputDialog()
        if input_dialog.exec() == QDialog.DialogCode.Accepted:
            x = input_dialog.get_x()
            y = input_dialog.get_y()
            self.draw_circle(x, y)

    def draw_circle(self, x, y, class_id):
        if self.photo_pixmap:
            painter = QPainter(self.photo_pixmap)
            painter.setPen(QPen(QColor(255, 0, 0, 128), 2))
            x = int(x)
            y = int(y)

            if class_id == 0:
                # Draw a red circle
                radius = 50
                painter.setBrush(QColor(255, 0, 0, 128))
                painter.drawEllipse(QPointF(x, y), radius, radius)
            elif class_id == 1:
                # Draw a blue circle
                radius = 30
                painter.setBrush(QColor(0, 0, 255, 128))
                painter.drawEllipse(QPointF(x, y), radius, radius)
            else:
                # Invalid class, skip drawing
                painter.end()
                return

            # Draw the x and y coordinates on the circle
            text = f"{x}x{y}"
            font = QFont()
            font.setPointSize(12)
            painter.setFont(font)
            text_rect = painter.boundingRect(QRectF(QPointF(x, y - 25), QSizeF(100, 25)), Qt.AlignmentFlag.AlignCenter, text)
            painter.setPen(QPen(QColor(255, 0, 0, 200), 2))
            painter.setBrush(QColor(0, 0, 0, 150))
            painter.drawRect(text_rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

            painter.end()
            f"/home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/output/run_{len(os.listdir(args.out_dir))}/Basler a2A2600-64umBAS (4036004)/{args.exposure_time}/DET/"
            self.photo_label.setPixmap(self.photo_pixmap.scaled(self.photo_label.size()))

            self.hole_count += 1
            self.detection_history.append(f"Hole {self.hole_count}")


    def set_logo(self, button, logo_path):
        pixmap = QPixmap(logo_path).scaled(50, 50)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.rect().size())


    def clear_drawings(self):
        # TODO handle index out of error
        if len(self.photo_paths) > 0:
            self.photo_pixmap = QPixmap(self.photo_paths[self.current_photo_index])
            self.update_photo()

            self.hole_count = 0
            self.detection_history = []

    def get_image_info(self):
        # Güncel resmin adını alıp txt dosyasındaki ilgili satırları filtreleyerek döndürüyoruz
        current_image_name = os.path.basename(self.photo_paths[self.current_photo_index])
        image_info = []

        with open("fotoName.txt", "r") as file:
            lines = file.readlines()

        for line in lines:
            info = line.strip().split()

            if len(info) == 4 and info[0] == current_image_name:
                image_info.append(info)

        return image_info

    def auto_draw_circles(self):
        # Güncel resim bilgilerine göre kırmızı ve mavi yuvarlakları çiz
        image_path = self.photo_paths[self.current_photo_index]
        # TODO draw circles on image
        if not os.path.exists(image_path):
            print(f"{image_path} dosyası bulunamadı.")
            return

        image = QPixmap(image_path)
        painter = QPainter(image)

        for info in self.current_image_info:
            class_id = int(info[1])
            x = int(info[2])
            y = int(info[3])

            if class_id == 0:
                # Draw a red circle
                painter.setPen(QPen(QColor(255, 0, 0, 128), 2))
                painter.setBrush(QColor(255, 0, 0, 128))
                radius = 50
                painter.drawEllipse(QPointF(x, y), radius, radius)

                # Draw the x and y coordinates on the circle
                text = f"{x}x{y}"
                font = QFont()
                font.setPointSize(12)
                painter.setFont(font)
                text_rect = painter.boundingRect(QRectF(QPointF(x, y - 25), QSizeF(100, 25)), Qt.AlignmentFlag.AlignCenter, text)
                painter.setPen(QPen(QColor(255, 0, 0, 200), 2))
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRect(text_rect)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

            elif class_id == 1:
                # Draw a blue circle
                painter.setPen(QPen(QColor(0, 0, 255, 128), 2))
                painter.setBrush(QColor(0, 0, 255, 128))
                radius = 30
                painter.drawEllipse(QPointF(x, y), radius, radius)

                # Draw the x and y coordinates on the circle
                text = f"{x}x{y}"
                font = QFont()
                font.setPointSize(12)
                painter.setFont(font)
                text_rect = painter.boundingRect(QRectF(QPointF(x, y - 25), QSizeF(100, 25)), Qt.AlignmentFlag.AlignCenter, text)
                painter.setPen(QPen(QColor(0, 0, 255, 200), 2))
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRect(text_rect)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()
        self.photo_pixmap = image
        f"/home/inovako/Desktop/Inovako/Inovako/Tofas/tofas_app/output/run_{len(os.listdir(args.out_dir))}/Basler a2A2600-64umBAS (4036004)/{args.exposure_time}/DET/"
        self.photo_label.setPixmap(self.photo_pixmap.scaled(self.photo_label.size()))


class DrawCoordinatesInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Çizim Koordinatları")
        self.setModal(True)

        self.x_line_edit = QLineEdit()
        self.y_line_edit = QLineEdit()

        form_layout = QFormLayout()
        form_layout.addRow("X Koordinatı:", self.x_line_edit)
        form_layout.addRow("Y Koordinatı:", self.y_line_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    def get_x(self):
        return self.x_line_edit.text()

    def get_y(self):
        return self.y_line_edit.text()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())