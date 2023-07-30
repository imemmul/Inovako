import sys
import os
import random
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal, QRectF, QSizeF, pyqtProperty
from PyQt6.QtGui import QPixmap, QFont, QPainter, QPen, QColor, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QFileDialog, QFrame, QLineEdit, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QTabBar, QStackedWidget, QListWidget, QListWidgetItem, QRadioButton, QButtonGroup, QSizePolicy, QSpacerItem 
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QListWidget, QStyleOptionViewItem, QStyle
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QApplication
from PyQt6.QtGui import QPixmap



class WorkerThread(QThread):
    finished = pyqtSignal()

    def run(self):
        # Uzun süren işlemleri burada yapabilirsiniz
        # Ancak arayüz bileşenlerine dokunmayın, sadece ana thread'de yapın
        # İşlem tamamlandığında finished sinyalini gönderin
        self.finished.emit()

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



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        icon_path = "logo/inovako_logo_negatif.png"
        self.setWindowIcon(QIcon(icon_path))

        self.setWindowTitle("STAMPIX 1.0.0")
        self.setGeometry(0, 0, 1920, 1080)
        self.setFixedSize(1920, 1080)
        self.showFullScreen()

        self.setContentsMargins(30, 0, 0, 0)
        self.setStyleSheet("background-color: rgba(10, 21, 39, 255); color: white;")


        self.title_label = None
        self.current_image_info = []  # Geçerli resmin etiketlerini saklayacağımız liste

        self.current_photo_index = 0
        self.detection_history = []  # Tüm fotoğraflardaki Hole değerlerini burada tutacağız
        self.photo_directory = "./AHTG2G7G_025"
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
        #self.tab_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Yazı boyutunu değiştirmek için bir QFont oluşturun
        font = QFont()
        font.setPointSize(14)  # Yazı boyutunu istediğiniz gibi ayarlayabilirsiniz
        self.tab_bar.setFont(font)

        main_layout.addWidget(self.tab_bar)

        # Disable the "Detects" and "Settings" tabs
        self.tab_bar.setTabEnabled(1, False)  # "Detects" tab
        self.tab_bar.setTabEnabled(2, False)  # "Settings" tab

        self.left_bottom_widget = QWidget(self)
        self.left_bottom_layout = QVBoxLayout()
        self.left_bottom_widget.setLayout(self.left_bottom_layout)

        left_section_layout = QHBoxLayout()

        self.previous_photo_button = QPushButton()
        self.set_logo(self.previous_photo_button, 'yontus/sol.png')
        self.previous_photo_button.clicked.connect(self.go_back)
        self.previous_photo_button.setFixedSize(75, 252)
        self.previous_photo_button.setStyleSheet("border: 2px solid black;")
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
        self.change_photo_button.setStyleSheet("border: 2px solid black;")
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

        self.worker_thread = WorkerThread()
        self.worker_thread.finished.connect(self.on_thread_finished)
    
    def on_thread_finished(self):
        print("Thread işlemi tamamlandı!")
        # İşlem sonucunu ana thread'de kullanabilirsiniz
        # Ancak arayüz bileşenlerine dokunmayı unutmayın

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
        self.prev_list_button.setStyleSheet("border: 2px solid black;")  # Siyah kenarlık eklemek için stil yönlendirmesi

        self.next_list_button = QPushButton()
        self.set_logo(self.next_list_button, 'yontus/alt.png')
        self.next_list_button.clicked.connect(self.show_next_list)
        self.next_list_button.setFixedSize(75, 75)
        self.next_list_button.setStyleSheet("border: 2px solid black;")  # Siyah kenarlık eklemek için stil yönlendirmesi


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
        #self.photo_label = PhotoLabel(self)
        #self.photo_label.setStyleSheet("background-color: black; color: white; border: 2px solid #3a7ca5;")

        #self.photo_layout.addWidget(self.photo_label)


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
        ayarlar_layout = QVBoxLayout()
        ayarlar_widget.setLayout(ayarlar_layout)

        ayarlar_content_label = QLabel("This is the AYARLAR page.")
        ayarlar_content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ayarlar_layout.addWidget(ayarlar_content_label)

        ayarlar_widget.setStyleSheet("background-color: rgba(255, 255, 255, 255); color: black;")

        self.stacked_widget.addWidget(ayarlar_widget)

    def load_photos(self):
        self.photo_paths = []

        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

        for file_name in os.listdir(self.photo_directory):
            if file_name.lower().endswith(".jpg"):
                file_path = os.path.join(self.photo_directory, file_name)
                self.photo_paths.append(file_path)

    def change_photo(self):
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
        self.photo_pixmap = QPixmap(file_path)
        self.update_photo()

        # Güncel resim bilgilerine göre otomatik olarak kırmızı yuvarlakları çiz
        self.auto_draw_circles()

         # Get the photo_label from the photo_layout
        photo_label = self.photo_layout.itemAt(0).widget()
        if isinstance(photo_label, PhotoLabel):
            # Set the photo_pixmap to the photo_placeholder
            self.photo_label.pixmap()
            self.photo_placeholder.setPixmap(photo_label.pixmap())

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
        self.update_photo()

        # Güncel resim bilgilerine göre otomatik olarak kırmızı yuvarlakları çiz
        self.auto_draw_circles()

         # Get the photo_label from the photo_layout
        photo_label = self.photo_layout.itemAt(0).widget()
        if isinstance(photo_label, PhotoLabel):
            # Set the photo_pixmap to the photo_placeholder
            self.photo_placeholder.setPixmap(photo_label.pixmap())

    def update_photo(self):
        if self.photo_pixmap:
            self.photo_layout.removeWidget(self.photo_frame.findChild(QLabel))
            self.photo_label = PhotoLabel()
            self.photo_label.setPixmap(self.photo_pixmap)
            self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

            self.photo_label.setPixmap(self.photo_pixmap)

            self.hole_count += 1
            self.detection_history.append(f"Hole {self.hole_count}")


    def set_logo(self, button, logo_path):
        pixmap = QPixmap(logo_path).scaled(50, 50)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.rect().size())


    def clear_drawings(self):
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
        self.photo_label.setPixmap(self.photo_pixmap)


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
