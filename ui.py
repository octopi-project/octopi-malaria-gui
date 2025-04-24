import sys
import numpy as np
import threading
from queue import Empty
from utils import numpy2png_ui as numpy2png
import numpy as np

import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QStyleFactory, QFileDialog,
    QComboBox, QCheckBox, QGroupBox, QGridLayout,QSpinBox, QFrame, QDialog, QDoubleSpinBox
)
from PyQt5.QtGui import QImage, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg
from widgets import VirtualImageListWidget, ExpandableImageWidget

import time, os

from utils import SharedConfig

import cv2

MINIMUM_SCORE_THRESHOLD = 0.5  # Adjust this value as needed

class ImageAnalysisUI(QMainWindow):
    shutdown_signal = pyqtSignal()
    

    def __init__(self, start_event,shared_config:SharedConfig):
        super().__init__()
        self.start_event = start_event
        self.shared_config = shared_config
        self.logger = self.shared_config.setup_process_logger()
        self.setWindowTitle("Octopi")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Set the application style to Fusion for a more modern look
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        # Set a custom color palette
        palette = self.palette()
        palette.setColor(palette.Window, QColor("#ECF0F1"))
        palette.setColor(palette.WindowText, QColor("#2C3E50"))
        palette.setColor(palette.Button, QColor("#2C3E50"))
        palette.setColor(palette.ButtonText, QColor("#FFFFFF"))
        palette.setColor(palette.Highlight, QColor("#3498DB"))
        self.setPalette(palette)
        
        self.setStyleSheet("""
        * { font-size: 20px;  }
            QMainWindow { background-color: #ECF0F1; }
            QTabWidget::pane { border: 1px solid #BDC3C7; background-color: white; }
            QTabBar::tab { 
                background-color: #ECF0F1; 
                padding: 8px 16px; 
                margin-right: 2px; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { 
                background-color: white; 
                border-bottom: 2px solid #3498DB; 
            }
            QPushButton { 
                background-color: #2C3E50; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357eab; }
            QLineEdit { 
                padding: 6px; 
                border: 1px solid #BDC3C7; 
                border-radius: 4px; 
                background-color: white;
            }
            QLabel { color: #2C3E50; }
            QTableWidget { 
                gridline-color: #BDC3C7;
                selection-background-color: #3498DB;
            }
            QTableWidget::item:hover { background-color: #3ea8c2; }
        """)
        self.image_lock = threading.Lock()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.patient_id = ""
        self.setup_ui()

        self.image_cache = {}
        self.fov_image_cache = {}
        self.fov_data = {}
        self.max_cache_size = 1
        self.current_fov_index = -1
        self.selected_fov_id = None

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)

        self.fov_image_data = {} 
        self.fov_coordinates_data = {}  # Store coordinates for each FOV

        self.first_fov_time = None
        self.latest_fov_time = None
        
        # Current bounding box for highlighting clicked spots
        self.current_bbox = None

    def setup_ui(self):

        # Patient ID Label (initially empty)
        self.patient_id_label = QLabel("")
        self.patient_id_label.setAlignment(Qt.AlignCenter)
        self.patient_id_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2C3E50;
            padding: 10px;
            background-color: #e6f2ff;
            border-radius: 5px;
        """)
        self.main_layout.addWidget(self.patient_id_label)

        # Top layout with shutdown button
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.new_patient_button = QPushButton("New Patient")
        self.new_patient_button.clicked.connect(self.new_patient)
        self.new_patient_button.setStyleSheet("""
            QPushButton { 
                background-color: #48abe8; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357eab; }
        """)
        top_layout.addWidget(self.new_patient_button)

        # Add this after the "New Patient" button
        self.load_patient_button = QPushButton("Load Patient")
        self.load_patient_button.clicked.connect(self.load_patient)
        self.load_patient_button.setStyleSheet("""
            QPushButton { 
                background-color: #28a745; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        top_layout.addWidget(self.load_patient_button)

        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.shutdown)
        self.shutdown_button.setStyleSheet("""
            QPushButton { 
                background-color: #dc3545; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        top_layout.addWidget(self.shutdown_button)
        self.main_layout.addLayout(top_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

         # Start Tab
        start_tab = QWidget()

        # Card frame
        card = QFrame(self)
        card.setObjectName("card")
        card.setFixedSize(400, 500)
        card.setStyleSheet("""
            #card {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 25px;
            }
        """)
        card_layout = QVBoxLayout(card)

        # Welcome label
        welcome_label = QLabel("Welcome to Octopi")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #2C3E50;
            padding: 10px;
        """)

        card_layout.addWidget(welcome_label)

        # Patient ID input
        patient_id_layout = QVBoxLayout()
        patient_id_label = QLabel("Patient ID:")
        patient_id_label.setStyleSheet("""
            margin-bottom: 0px; 
        """)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter Patient ID")
        self.patient_id_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 4px;
                margin-bottom: 65px; 
                margin-top: 1px;
            }
        """)
        patient_id_layout.addWidget(patient_id_label)
        patient_id_layout.addWidget(self.patient_id_input)
        card_layout.addLayout(patient_id_layout)

        # To Loading Position button
        self.loading_position_button = QPushButton("To Loading Position")
        card_layout.addWidget(self.loading_position_button)
        self.loading_position_button.clicked.connect(self.move_to_loading_position)

        # Start Scanning button
        self.start_button = QPushButton("Start Scanning")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        card_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_analysis)

        start_layout = QHBoxLayout(start_tab)
        start_layout.addWidget(card, alignment=Qt.AlignCenter)

        self.tab_widget.addTab(start_tab, "Start")

        # FOV Tab
        fov_tab = QWidget()
        fov_layout = QHBoxLayout(fov_tab)
        splitter = QSplitter(Qt.Horizontal)
        fov_layout.addWidget(splitter)

        # Left column: FOV image viewer
        fov_image_widget = QWidget()
        left_layout = QVBoxLayout(fov_image_widget)
        
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.show_previous_fov)
        self.next_button.clicked.connect(self.show_next_fov)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)

        self.fov_image_view = pg.ImageView()
        self.setup_fov_image_view(self.fov_image_view)
        left_layout.addWidget(self.fov_image_view)

        splitter.addWidget(fov_image_widget)

        # Middle column: Positive spots display
        positive_spots_widget = QWidget()
        middle_layout = QVBoxLayout(positive_spots_widget)

        self.positive_images_widget = ExpandableImageWidget()
        middle_layout.addWidget(self.positive_images_widget)

        splitter.addWidget(positive_spots_widget)

        # Right column: FOV list
        fov_list_widget = QWidget()
        right_layout = QVBoxLayout(fov_list_widget)

        # Add a new label for average processing time
        self.avg_processing_time_label = QLabel("Avg Processing Time: N/A")
        self.avg_processing_time_label.setStyleSheet("""
            font-size: 14px;
            color: #2C3E50;
            padding: 5px;
            background-color: #ECF0F1;
            border-radius: 3px;
        """)
        right_layout.addWidget(self.avg_processing_time_label)

        # Timer to update average processing time
        self.update_avg_timer = QTimer(self)
        self.update_avg_timer.timeout.connect(self.update_avg_processing_time)
        self.update_avg_timer.start(1000) 

        self.stats_label_small = QLabel("FoVs: 0 | RBCs: 0 | Parasites / μl: 0")
        self.stats_label_small.setStyleSheet("""
            font-size: 14px;
        """)     
        # add on top of the fov table
        right_layout.addWidget(self.stats_label_small)

        self.fov_table = QTableWidget()
        self.fov_table.setColumnCount(3)
        self.fov_table.setHorizontalHeaderLabels(["FOV id", "RBCs", "Positives"])
        self.fov_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fov_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fov_table.itemClicked.connect(self.fov_table_item_clicked)
        right_layout.addWidget(self.fov_table)
        
        splitter.addWidget(fov_list_widget)

        total_width = self.width()
        unit = total_width / 10  
        splitter.setSizes([int(4*unit), int(4*unit), int(2*unit)])

        self.tab_widget.addTab(fov_tab, "FOVs List")

         # Cropped Images Tab
        self.cropped_tab = QWidget()
        self.cropped_layout = QVBoxLayout(self.cropped_tab)

        self.stats_label = QLabel("FoVs: 0 | Total RBC Count: 0 | Total Malaria Positives: 0 | Parasites / μl: 0 | Parasitemia: 0%")
        self.stats_label.setStyleSheet("""
            margin-top: 10px; 
            margin-bottom: 10px;
            font-size: 35px;
            color: black;
        """)
        self.cropped_layout.addWidget(self.stats_label)

        self.virtual_image_list = VirtualImageListWidget()
        self.cropped_layout.addWidget(self.virtual_image_list)

        self.tab_widget.addTab(self.cropped_tab, "Malaria Detection Report")

        # A tab for live view
        live_view_tab = QWidget()
        live_view_layout = QHBoxLayout(live_view_tab)

        # Left side container for controls
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        self.channel_combo = QComboBox()
        self.load_channels()
        self.channel_combo.currentIndexChanged.connect(self.switch_channel)
        channel_layout.addWidget(self.channel_combo)
        left_layout.addWidget(channel_group)

        # Control buttons
        self.live_button = QPushButton("LIVE")
        self.live_button.clicked.connect(self.toggle_live_view)
        left_layout.addWidget(self.live_button)

        # Live position display
        self.live_position_label = QLabel("X: 0, Y: 0, Z: 0")
        self.live_position_label.setStyleSheet("""
            color: #2C3E50;
            background-color: #ECF0F1;
            border-radius: 5px;
            padding: 5px;
            margin: 5px;
        """)
        left_layout.addWidget(self.live_position_label)
        
        self.auto_focus_calibration_button = QPushButton("Auto Focus Calibration")
        self.auto_focus_calibration_button.clicked.connect(self.auto_focus_calibration)
        left_layout.addWidget(self.auto_focus_calibration_button)

        # Add some stretch to push everything to the top
        left_layout.addStretch(1)
        # Add left container to main layout
        live_view_layout.addWidget(left_container)


        # Live view graph
        self.live_view_graph = pg.GraphicsLayoutWidget()
        self.live_view_plot = self.live_view_graph.addPlot()
        self.live_view_image = pg.ImageItem()
        self.live_view_plot.setAspectLocked(True, ratio=1)
        # hide axis
        self.live_view_plot.hideAxis('left')
        self.live_view_plot.hideAxis('bottom')
        # get rid of the margin
        self.live_view_plot.layout.setContentsMargins(0, 0, 0, 0)

        self.live_view_plot.addItem(self.live_view_image)
        live_view_layout.addWidget(self.live_view_graph)

        self.tab_widget.addTab(live_view_tab, "Live View")

        # Timer for updating live view
        self.live_view_timer = QTimer(self, interval=int(1.0 / self.shared_config.frame_rate.value * 1000))
        self.live_view_timer.timeout.connect(self.update_live_view)

        # Timer for updating live position
        self.live_position_timer = QTimer(self)
        self.live_position_timer.timeout.connect(self.update_live_position)
        self.live_position_timer.start(100)

        # a tab for settings
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        # Directory selection
        directory_group = QGroupBox("Save Directory")
        directory_layout = QHBoxLayout(directory_group)
        self.directory_input = QLineEdit()
        # set a fixed width for the input field
        self.directory_input.setFixedWidth(500)
        # set a default input
        self.directory_input.setText(os.path.join(os.getcwd(), "saved_data"))
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_directory)
        directory_layout.addWidget(self.directory_input)
        directory_layout.addWidget(self.browse_button)
        settings_layout.addWidget(directory_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Image options
        options_group = QGroupBox("Image Saving Options")
        options_layout = QVBoxLayout(options_group)
        self.bf_image_check = QCheckBox("Bright Field (left and right half)")
        self.fluo_image_check = QCheckBox("Fluorescence")
        self.dpc_image_check = QCheckBox("DPC")
        self.positives_images_check = QCheckBox("Spots Images")
        self.bf_image_check.setChecked(False)
        self.fluo_image_check.setChecked(True)
        self.dpc_image_check.setChecked(True)
        self.positives_images_check.setChecked(True)
        options_layout.addWidget(self.bf_image_check)
        options_layout.addWidget(self.fluo_image_check)
        options_layout.addWidget(self.dpc_image_check)
        options_layout.addWidget(self.positives_images_check)
        settings_layout.addWidget(options_group, alignment=Qt.AlignTop | Qt.AlignLeft)


        # Position selection
        position_group = QGroupBox("Field of View Selection")
        position_layout = QGridLayout(position_group)
        position_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_input = QSpinBox()
        self.x_input.setRange(2, 50)  # Adjust the range as needed
        self.x_input.setValue(8) 
        self.x_input.setStyleSheet("QSpinBox { width: 1px; height: 25px; }")
        position_layout.addWidget(self.x_input, 0, 1)
        position_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_input = QSpinBox()
        self.y_input.setRange(2, 20)  # Adjust the range as needed
        self.y_input.setValue(8)  
        self.y_input.setStyleSheet("QSpinBox { width: 1px; height: 25px; }")
        position_layout.addWidget(self.y_input, 1, 1)
        # shrink the first column to a certain ratio
        position_layout.setColumnStretch(0, 1)
        position_layout.setColumnStretch(1, 5)
        settings_layout.addWidget(position_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Add Minimum Score Threshold input
        threshold_group = QGroupBox("Detection Threshold")
        threshold_layout = QHBoxLayout(threshold_group)
        threshold_layout.addWidget(QLabel("Minimum Score Threshold:"))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(MINIMUM_SCORE_THRESHOLD)
        self.threshold_input.setDecimals(2)
        self.threshold_input.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_input)
        settings_layout.addWidget(threshold_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        settings_layout.addStretch(1)

        settings_tab.setLayout(settings_layout)
        self.tab_widget.addTab(settings_tab, "Settings")

    def update_threshold(self, value):
        global MINIMUM_SCORE_THRESHOLD
        MINIMUM_SCORE_THRESHOLD = value



    def switch_channel(self):
        index = self.channel_combo.currentIndex()
        self.shared_config.set_channel_selected(index)
    
    def auto_focus_calibration(self):
        # if live is on, turn it off
        if self.shared_config.is_live_view_active.value:
            self.stop_live_view()

        self.live_view_timer.start()
        self.shared_config.is_auto_focus_calibration.value = True
        # generate a window saying it is doing the calibration while shared_config.is_auto_focus_calibration.value == True
        self.calibration_dialog = AutoFocusDialog(self, title="Auto-focus calibration", message="The system is calibrating the auto-focus. Please wait...")
        # Start the timer before showing the dialog
        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.check_calibration_status)
        self.calibration_timer.start(1000)  # Check every 500 ms

        self.calibration_dialog.show()
        QApplication.processEvents() 

    def setup_fov_image_view(self,image_view):
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.ui.histogram.hide()
        image_view.view.setMouseEnabled(x=True, y=True)
        image_view.view.setBackgroundColor((255, 255, 255))
        # Create a box item for bounding box display
        self.bbox_item = pg.ROI((0, 0), (0, 0), pen=pg.mkPen('r', width=2))
        self.bbox_item.hide()
        image_view.view.addItem(self.bbox_item)

    def load_channels(self):
        try:
            tree = ET.parse('channel_configurations.xml')
            root = tree.getroot()
            channels = [mode.get('Name') for mode in root.findall('mode')]
            self.shared_config.set_channels_list(channels)
            self.channel_combo.addItems(channels)
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML: {e}")
        except FileNotFoundError:
            self.logger.error("channel_configurations.xml file not found")

    def toggle_live_view(self):
        # check if scanning is in progress, if so show a window saying that scanning is in progress
        if self.start_button.text() == "Scanning in progress":
            QMessageBox.warning(self, "Warning", "Scanning is in progress. Please wait until scanning is complete before starting live view. (By clicking New Patient)", QMessageBox.Ok)
            return
        if self.shared_config.is_live_view_active.value:
            self.stop_live_view()
        else:
            self.start_live_view()

    def start_live_view(self):
        self.live_button.setText("STOP LIVE")
        self.live_button.setStyleSheet("""
            QPushButton { 
                background-color: #48abe8; 
            }
            QPushButton:hover { background-color: #357eab; }
        """)
        self.live_view_timer.start()
        self.shared_config.is_live_view_active.value = True

    def stop_live_view(self):
        self.live_button.setText("LIVE")
        self.live_button.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
            }
            QPushButton:hover { background-color: #357eab; }
        """)
        self.live_view_timer.stop()
        # clear up the image
        self.live_view_image.clear()
        self.shared_config.is_live_view_active.value = False
    
    def update_live_view(self):
        # Generate a random image
        image = self.shared_config.get_live_view_image()
        self.live_view_image.setImage(image)
 
    def update_live_position(self):
        x = self.shared_config.live_x.value
        y = self.shared_config.live_y.value
        z = self.shared_config.live_z.value
        self.live_position_label.setText(f"X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")


    def move_to_loading_position(self):
        if self.loading_position_button.text() == "To Loading Position":

            with self.shared_config.position_lock:
                if not self.shared_config.to_scanning.value:
                    self.shared_config.set_to_loading()   
                    self.loading_position_button.setText("To Scanning Position")
                    self.loading_position_button.setStyleSheet("""
                        QPushButton { 
                            background-color: #48abe8; 
                        }
                        QPushButton:hover { background-color: #357eab; }
                    """)
        else:
            with self.shared_config.position_lock:
                if not self.shared_config.to_loading.value:
                    self.shared_config.set_to_scanning()
                    self.loading_position_button.setText("To Loading Position")
                    self.loading_position_button.setStyleSheet("""
                        QPushButton {
                            background-color: #2C3E50;
                        }
                        QPushButton:hover { background-color: #357eab; }
                    """)

    def shutdown(self):
        self.new_patient()
        self.shutdown_signal.emit()
        self.close()

    def new_patient(self):

        try:
            stats_path = os.path.join(self.shared_config.get_path(), "stats.txt")
            if not os.path.exists(stats_path):
                with open(stats_path, "w") as f:
                # write what is shown in the stats_label
                    f.write(self.stats_label.text())
            rbc_path = os.path.join(self.shared_config.get_path(), "rbc_counts.csv")
            if not os.path.exists(rbc_path):
                # save the RBCs count as a csv
                with open(rbc_path, "w") as f:
                    for fov_id, data in self.fov_data.items():
                        f.write(f"{fov_id},{data['rbc_count']}\n")
        except Exception as e:
            self.logger.error(f"Error saving stats: {e}")

        # Clear all caches and data
        self.image_cache.clear()
        self.fov_image_cache.clear()

        self.fov_data.clear()
        self.fov_image_data.clear()

        # clear bounding box
        self.bbox_item.hide()

        # Reset UI elements
        self.fov_table.setRowCount(0)
        self.virtual_image_list.clear()
        self.fov_image_view.clear()
        self.patient_id_label.setText("")
        self.positive_images_widget.image_list.clear()
        self.stats_label.setText("FoVs: 0 | RBCs count: 0 | Positives: 0 | Parasites / μl: 0 | Parasitemia: 0%")
        self.stats_label_small.setText("FoVs: 0 | RBCs: 0 | Parasites / μl: 0")
        
        # Reset other variables
        self.current_fov_index = -1
        self.selected_fov_id = None
        self.patient_id = ""
        
        # Clear the patient ID input and re-enable the start button
        self.patient_id_input.clear()
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Scanning")
        
        # Switch back to the start tab
        self.tab_widget.setCurrentIndex(0)

        # Signal the main process to stop
        self.start_event.clear()

        self.first_fov_time = None
        self.latest_fov_time = None

        self.shared_config.set_auto_focus_indicator(False)


    def update_avg_processing_time(self):
        if self.first_fov_time is not None and self.latest_fov_time:
            total_time = self.latest_fov_time - self.first_fov_time
            avg_time = total_time / len(self.fov_data) if len(self.fov_data) > 0 else 0
            self.avg_processing_time_label.setText(f"Avg Processing Time: {avg_time:.3f} s")
        else:
            self.avg_processing_time_label.setText("Avg Processing Time: N/A")

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)
    
    def update_cropped_images(self, fov_id, images, scores, coordinates=None):
        
        with self.image_lock:
            malaria_positives = 0
            updated_images = []
            updated_coords = []
            for idx, (img, score) in enumerate(zip(images, scores)):
                img_hash = hash(img.tobytes())
                if img_hash not in self.image_cache:
                    overlay_img = numpy2png(img, resize_factor=None)
                    if overlay_img is not None:
                        qimg = self.create_qimage(overlay_img)
                        coord = None
                        if coordinates is not None and idx < len(coordinates):
                            coord = coordinates[idx]
                        self.image_cache[img_hash] = (qimg, score, coord)
                        if score >= MINIMUM_SCORE_THRESHOLD:
                            malaria_positives += 1
                            updated_images.append((qimg, score))
                            updated_coords.append(coord)
                else:
                    qimg, cached_score, coord = self.image_cache[img_hash]
                    if cached_score >= MINIMUM_SCORE_THRESHOLD:
                        malaria_positives += 1
                        updated_images.append((qimg, cached_score))
                        updated_coords.append(coord)
            
            self.update_malaria_positives(fov_id, malaria_positives)
            self.fov_image_data[fov_id] = updated_images
            self.fov_coordinates_data[fov_id] = updated_coords

        # Update only the changed FOV
        self.virtual_image_list.update_images(updated_images, fov_id, updated_coords)
        self.update_stats()

        # Update the positive images widget if this is the selected FOV
        if fov_id == self.selected_fov_id:
            self.update_positive_images(fov_id)

    def update_all_fov_images(self):
        self.virtual_image_list.clear()
        for fov_id, images in self.fov_image_data.items():
            self.virtual_image_list.update_images(images, fov_id)
        self.update_stats()
    
    def create_qimage(self, overlay_img):
        height, width, channel = overlay_img.shape
        bytes_per_line = 3 * width
        return QImage(overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def update_display(self):
        self.display_cropped_images(float(self.score_filter.text() or 0))

    def update_fov_list(self, fov_id):
        row_position = self.fov_table.rowCount()
        self.fov_table.insertRow(row_position)
        self.fov_table.setItem(row_position, 0, QTableWidgetItem(fov_id))
        self.fov_table.setItem(row_position, 1, QTableWidgetItem("0"))  # Initial RBC Count
        self.fov_table.setItem(row_position, 2, QTableWidgetItem("0"))  # Initial Malaria Positives
        self.fov_data[fov_id] = {'rbc_count': 0, 'malaria_positives': 0}

        current_time = time.time()
        if self.first_fov_time is None:
            self.first_fov_time = current_time

    def update_fov_image(self, fov_id, dpc_image, fluorescent_image):
        # Combine DPC and fluorescent images
        overlay_img = self.create_overlay(dpc_image, fluorescent_image)
        
        # Cache the numpy array
        self.fov_image_cache[fov_id] = overlay_img
        
        # Limit cache size
        if len(self.fov_image_cache) > self.max_cache_size:
            oldest_fov = next(iter(self.fov_image_cache))
            del self.fov_image_cache[oldest_fov]
        
        self.selected_fov_id = fov_id
        self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
        self.display_current_fov()

    def display_current_fov(self):
        if self.selected_fov_id and self.selected_fov_id in self.fov_image_cache:
            overlay_img = self.fov_image_cache[self.selected_fov_id]

            # Update the PyQtGraph ImageView
            self.fov_image_view.setImage(overlay_img, autoLevels=False, levels=(0, 255))

            # Highlight the selected row in the table
            row = self.find_fov_row(self.selected_fov_id)
            if row is not None:
                self.fov_table.selectRow(row)
        else:
            self.logger.error(f"No FOV image available to display")

    def show_previous_fov(self):
        # Clear any existing bounding box
        self.bbox_item.hide()
        
        current_row = self.fov_table.currentRow()
        if current_row > 0:
            previous_row = current_row - 1
            fov_id = self.fov_table.item(previous_row, 0).text()
            self.load_fov_cache(fov_id)
            self.fov_table.selectRow(previous_row)
            self.update_positive_images(fov_id)

    def show_next_fov(self):
        # Clear any existing bounding box
        self.bbox_item.hide()
        
        current_row = self.fov_table.currentRow()
        if current_row < self.fov_table.rowCount() - 1:
            next_row = current_row + 1
            fov_id = self.fov_table.item(next_row, 0).text()
            self.load_fov_cache(fov_id)
            self.fov_table.selectRow(next_row)
            self.update_positive_images(fov_id)

    def fov_table_item_clicked(self, item):
        # Clear any existing bounding box
        self.bbox_item.hide()
        
        fov_id = self.fov_table.item(item.row(), 0).text()
        self.load_fov_cache(fov_id)

    def load_fov_cache(self, fov_id):
        # Clear any existing bounding box
        self.bbox_item.hide()
        
        if fov_id in self.fov_image_cache:
            self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
        else:
            try:
                #filename = f"{self.shared_config.get_path()}/{fov_id}_overlay.npy"
                #img_array = np.load(filename)

                if self.shared_config.SAVE_NPY.value:
                    dpc = np.load(f"{self.shared_config.get_path()}/{fov_id}_dpc.npy")
                    fluorescent = np.load(f"{self.shared_config.get_path()}/{fov_id}_fluorescent.npy")
                else:
                    dpc = cv2.imread(f"{self.shared_config.get_path()}/{fov_id}_dpc.bmp", cv2.IMREAD_GRAYSCALE)
                    fluorescent = cv2.imread(f"{self.shared_config.get_path()}/{fov_id}_fluorescent.bmp")

                #print(f"DPC shape: {dpc.shape} and dtype {dpc.dtype}")
                #print(f"Fluorescent shape: {fluorescent.shape} and dtype {fluorescent.dtype}")

                img_array = self.create_overlay(dpc, fluorescent)
                #print(f"Loading fov {fov_id} with shape {img_array.shape} and dtype {img_array.dtype}")
                self.fov_image_cache[fov_id] = img_array
                # delete the oldest image
                if len(self.fov_image_cache) > self.max_cache_size:
                    oldest_fov = next(iter(self.fov_image_cache))
                    del self.fov_image_cache[oldest_fov]
            except FileNotFoundError:
                self.logger.error(f"FOV {fov_id} not found on disk")
                return

        self.selected_fov_id = fov_id
        self.display_current_fov()
        self.update_positive_images(fov_id)
            

    def update_positive_images(self, fov_id):
        if fov_id in self.fov_image_data:
            coordinates = self.fov_coordinates_data.get(fov_id, [None] * len(self.fov_image_data[fov_id]))
            self.positive_images_widget.update_images(
                self.fov_image_data[fov_id], 
                fov_id,
                coordinates
            )
            # Connect the click signal if not already connected
            try:
                self.positive_images_widget.image_clicked.disconnect()
            except:
                pass
            self.positive_images_widget.image_clicked.connect(self.on_positive_image_clicked)
        else:
            print(f"No positive images for FOV {fov_id}")

    def on_positive_image_clicked(self, coordinates):
        """Handle when a positive image is clicked to show its bounding box"""
        if coordinates is None:
            # Hide any existing bounding box
            self.bbox_item.hide()
            return
        
        # Show a bounding box around the spot in the FOV image
        try:
            # Coordinates are typically [x, y, radius] or [x, y]
            x, y = coordinates[0], coordinates[1]
            r = 15  # Fixed radius to ensure 31x31 box (matches cropped images)
            
            # Update the bounding box position and size
            self.bbox_item.setPos(x - r, y - r)
            self.bbox_item.setSize((2*r, 2*r))
            self.bbox_item.show()
            
            # Adjust view to center on the spot
            self.fov_image_view.view.setRange(
                xRange=(x - 2*r, x + 2*r), 
                yRange=(y - 2*r, y + 2*r),
                padding=0.5
            )
        except Exception as e:
            self.logger.error(f"Error displaying bounding box: {e}")
            self.bbox_item.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(200)

    def create_overlay(self, dpc_image, fluorescent_image):
        # stack fluorescent and DPC images to 4xHxW
        dpc_image = dpc_image.astype(np.float16) / 255.0 if dpc_image.dtype == np.uint8 else dpc_image
        fluorescent_image = fluorescent_image.astype(np.float16) / 255.0 if fluorescent_image.dtype == np.uint8 else fluorescent_image
        # then direcly call numpy2png
        img = np.stack([fluorescent_image[:,:,0], fluorescent_image[:,:,1], fluorescent_image[:,:,2], dpc_image], axis=0)
        #print(f"Overlay image shape: {img.shape} and dtype {img.dtype}")
        img =  numpy2png(img,resize_factor=None)
        
        return img

    def update_rbc_count(self, fov_id, count):
        self.fov_data[fov_id]['rbc_count'] = count
        row = self.find_fov_row(fov_id)
        if row is not None:
            self.fov_table.setItem(row, 1, QTableWidgetItem(str(count)))
        self.update_stats()

        self.latest_fov_time = time.time()

    def find_fov_row(self, fov_id):
        for row in range(self.fov_table.rowCount()):
            if self.fov_table.item(row, 0).text() == fov_id:
                return row
        return None
    def update_malaria_positives(self, fov_id, count):
        self.fov_data[fov_id]['malaria_positives'] = count
        row = self.find_fov_row(fov_id)
        if row is not None:
            self.fov_table.setItem(row, 2, QTableWidgetItem(str(count)))
        self.update_stats()

    def update_stats(self):
        total_rbc = sum(data['rbc_count'] for data in self.fov_data.values())
        total_positives = self.virtual_image_list.model.rowCount()
        # round to two decimal places   
        parasite_per_ul = round(total_positives  * (5000000 / (total_rbc + 1)), 2)
        parasitemia_percentage = round(total_positives / (total_rbc + 1) * 100, 2)
        self.stats_label.setText(f"FoVs: {len(self.fov_data)} | RBCs Count: {total_rbc:,} | Positives: {total_positives:,} | Parasites / μl: {int(parasite_per_ul):,} | Parasitemia: {parasitemia_percentage:.2f}%")
        self.stats_label_small.setText(f"FoVs: {len(self.fov_data)} | RBCs: {total_rbc:,} | Parasites / μl: {int(parasite_per_ul):,}")

    def start_analysis(self):
        self.patient_id = self.patient_id_input.text().strip()
        directory = self.directory_input.text().strip()

        if not self.patient_id:
            QMessageBox.warning(self, "Input Error", "Please enter a Patient ID before starting the analysis.")
            return
                        
        if not directory:
            QMessageBox.warning(self, "Input Error", "Please enter or select a directory for saving data.")
            return
        
        if self.loading_position_button.text() == "To Scanning Position":
            QMessageBox.warning(self, "Input Error", "Please return to the loading position before starting the analysis.")
            return

        # set the patient id in the shared config
        # add a timestamp to the patient id
        self.patient_id = f"{self.patient_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.shared_config.patient_id.value = self.patient_id
        # Create patient directory
        patient_directory = os.path.join(directory, self.patient_id)
        try:
            os.makedirs(patient_directory, exist_ok=False)
        except FileExistsError:
            QMessageBox.warning(self, "Patient ID Exists", f"A directory for patient ID '{self.patient_id}' already exists. Please use a different ID.")
            return
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to create patient directory: {e}")
            return
        self.shared_config.set_path(patient_directory)

        self.shared_config.set_log_file(os.path.join(directory, f"{self.patient_id}"))
        self.logger = self.shared_config.setup_process_logger()
        self.logger.info(f"Starting analysis for patient {self.patient_id}")

        self.auto_focus_dialog = AutoFocusDialog(self)
    
        # Start the timer before showing the dialog
        self.auto_focus_timer = QTimer(self)
        self.auto_focus_timer.timeout.connect(self.check_auto_focus_status)
        self.auto_focus_timer.start(500)  # Check every 500 ms

        self.auto_focus_dialog.show()
        QApplication.processEvents()  


        self.shared_config.save_bf_images.value = self.bf_image_check.isChecked()
        self.shared_config.save_fluo_images.value = self.fluo_image_check.isChecked()
        self.shared_config.save_spot_images.value = self.positives_images_check.isChecked()
        self.shared_config.save_dpc_image.value = self.dpc_image_check.isChecked()
        self.shared_config.nx.value = self.x_input.value()
        self.shared_config.ny.value = self.y_input.value()
        

        self.patient_id_label.setText(f"Patient ID: {self.patient_id}")
        self.start_event.set()  # Signal the main process to start
        self.tab_widget.setCurrentIndex(1)  # Switch to FOVs List tab
        self.start_button.setEnabled(False)
        self.start_button.setText("Scanning in progress")

    def check_auto_focus_status(self):
        #print(f"Checking auto-focus status. Indicator: {self.shared_config.auto_focus_indicator.value}")
        QApplication.processEvents()  # Force processing of events
        if self.shared_config.auto_focus_indicator.value:
            print("Auto-focus complete. Closing dialog.")
            self.auto_focus_timer.stop()
            self.auto_focus_dialog.close()
            self.auto_focus_dialog = None
            QApplication.processEvents()  

    def check_calibration_status(self):
        #print(f"Checking auto-focus status. Indicator: {self.shared_config.auto_focus_indicator.value}")
        QApplication.processEvents()  # Force processing of events
        if not self.shared_config.is_auto_focus_calibration.value:
            print("Auto-focus calibration complete. Closing dialog.")
            self.calibration_timer.stop()
            self.calibration_dialog.close()
            self.calibration_dialog = None
            self.live_view_timer.stop()
            QApplication.processEvents()  

    def load_patient(self):
        # Clear existing data
        self.new_patient()  # This will clear all the existing data
        directory = QFileDialog.getExistingDirectory(self, "Select Patient Directory")
        if directory:
            patient_id = os.path.basename(directory)
            print(f"Loading patient {patient_id} from directory {directory}")
            self.load_patient_data(patient_id, directory)

    def load_patient_data(self, patient_id, directory):
        self.patient_id = patient_id
        self.patient_id_label.setText(f"Patient ID: {self.patient_id}")
        self.shared_config.set_path(directory)

        # Load saved data
        self.load_saved_data(directory)

        # Switch to the FOVs List tab
        self.tab_widget.setCurrentIndex(1)

    def load_saved_data(self, directory):
        # Load stats

        # Load FOV data
        fovs = [f.split("_dpc")[0] for f in os.listdir(directory) if f.endswith("_dpc.npy") or f.endswith("_dpc.bmp")]
        # sort the fovs by arithmetic order
        fovs.sort(key=lambda x: int(x.split("_")[-1]))
        for fov_id in fovs:
            self.update_fov_list(fov_id)
            
            # Load cropped images, scores, and coordinates if available
            cropped_path = os.path.join(directory, f"{fov_id}_cropped.npy")
            scores_path = os.path.join(directory, f"{fov_id}_scores.npy")
            coordinates_path = os.path.join(directory, f"{fov_id}_filtered_spots.npy")
            
            if os.path.exists(cropped_path) and os.path.exists(scores_path):
                cropped_images = np.load(cropped_path)
                scores = np.load(scores_path)
                
                # Also load coordinates if available
                coordinates = None
                if os.path.exists(coordinates_path):
                    coordinates = np.load(coordinates_path)
                    self.logger.info(f"Loaded coordinates for FOV {fov_id}: {len(coordinates)} spots")
                else:
                    self.logger.info(f"No coordinates found for FOV {fov_id}")
                
                # Update with images, scores, and coordinates
                self.update_cropped_images(fov_id, cropped_images, scores, coordinates)

        self.update_all_fov_images()

        self.load_fov_cache(fovs[-1])

        try:
            with open(os.path.join(directory, "stats.txt"), "r") as f:
                stats = f.read()
                self.stats_label.setText(stats)
                self.update_stats_small_label(stats)

                # load the csv for the rbc_counts
                with open(os.path.join(directory, "rbc_counts.csv"), "r") as f:
                    for line in f:
                        fov_id, rbc_count = line.strip().split(",")
                        self.update_rbc_count(fov_id, int(rbc_count))

                # update the stats
                self.update_stats()
        except FileNotFoundError:
            print("Stats file not found")

    def update_stats_small_label(self, stats):
        # Extract relevant information from stats and update stats_label_small
        stats_parts = stats.split("|")
        fovs = stats_parts[0].strip()
        rbcs = stats_parts[1].strip()
        parasites = stats_parts[3].strip()
        self.stats_label_small.setText(f"{fovs} | {rbcs} | {parasites}")

class AutoFocusDialog(QDialog):
    def __init__(self, parent=None,title="Auto-focus",message="Auto-focusing in progress. Please wait..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.label = QLabel(message)
        layout.addWidget(self.label)

    def closeEvent(self, event):
        print("Dialog close event triggered")
        event.accept()

class UIThread(QThread):
    update_fov = pyqtSignal(str)
    update_images = pyqtSignal(str, np.ndarray, np.ndarray)
    update_rbc = pyqtSignal(str, int)
    update_fov_image = pyqtSignal(str, np.ndarray, np.ndarray)
    update_coordinates = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)  # New signal for coordinates

    def __init__(self, input_queue, output, shared_memory_final, shared_memory_classification, 
                 shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                 shared_memory_timing, final_lock, timing_lock, window, shared_config):
        super().__init__()
        self.input_queue = input_queue
        self.output = output    
        self.shared_memory_final = shared_memory_final
        self.shared_memory_classification = shared_memory_classification
        self.shared_memory_segmentation = shared_memory_segmentation
        self.shared_memory_acquisition = shared_memory_acquisition
        self.shared_memory_dpc = shared_memory_dpc
        self.shared_memory_timing = shared_memory_timing
        self.final_lock = final_lock
        self.timing_lock = timing_lock
        self.processed_fovs = set()
        self.window = window
        self.logger = shared_config.setup_process_logger()

    def run(self):
        while True:
            try:
                fov_id = self.input_queue.get(timeout=0.1)
                self.log_time(fov_id, "UI Process", "start")

                with self.final_lock:
                    if fov_id in self.shared_memory_final and not self.shared_memory_final[fov_id]['displayed']:
                        self.process_fov(fov_id)
                        self.log_time(fov_id, "UI Process", "end")

                        temp_dict = self.shared_memory_final[fov_id]
                        temp_dict['displayed'] = True
                        self.shared_memory_final[fov_id] = temp_dict    
                        self.processed_fovs.add(fov_id)
                        if self.shared_memory_final[fov_id]['saved']:
                            self.output.put(fov_id)

            except Empty:
                pass

    def process_fov(self, fov_id):
        self.update_fov.emit(fov_id)
        
        # Emit full FOV images
        acquisition_data = self.shared_memory_acquisition.get(fov_id, {})
        dpc_data = self.shared_memory_dpc.get(fov_id, {})
        dpc_image = dpc_data.get('dpc_image', np.array([]))
        fluorescent_image = acquisition_data.get('fluorescent', np.array([]))

        if dpc_image.size > 0 and fluorescent_image.size > 0:
            self.update_fov_image.emit(fov_id, dpc_image, fluorescent_image)
        else:
            self.logger.error(f"Missing DPC or fluorescent image for FOV {fov_id}")

        classification_data = self.shared_memory_classification.get(fov_id, {})
        images = classification_data.get('cropped_images', np.array([]))
        scores = classification_data.get('scores', np.array([]))
        
        # Get coordinates (filtered_spots) from classification data
        filtered_spots = classification_data.get('filtered_spots', np.array([]))
        
        if len(images) > 0 and len(scores) > 0:
            if len(filtered_spots) > 0:
                # Pass images, scores, and coordinates
                self.update_coordinates.emit(fov_id, images, scores, filtered_spots)
            else:
                # Fall back to just images and scores if no coordinates
                self.update_images.emit(fov_id, images, scores)
        else:
            self.logger.error(f"No images or scores for FOV {fov_id}")
        
        segmentation_data = self.shared_memory_segmentation.get(fov_id, {})
        rbc_count = segmentation_data.get('n_cells', 0)
        self.update_rbc.emit(fov_id, rbc_count)
    
    def log_time(self,fov_id: str, process_name: str, event: str):
        import time
        with self.timing_lock:
            if fov_id not in self.shared_memory_timing:
                self.shared_memory_timing[fov_id] = {}
            if process_name not in self.shared_memory_timing[fov_id]:
                self.shared_memory_timing[fov_id][process_name] = {}

            temp_dict = self.shared_memory_timing[fov_id]
            temp_process_dict = temp_dict.get(process_name, {})
            temp_process_dict[event] = time.time()
            temp_dict[process_name] = temp_process_dict
            self.shared_memory_timing[fov_id] = temp_dict

    def shutdown(self):
        self.shutdown_signal.emit()
        QApplication.quit()  # This will close all windows

    def closeEvent(self, event):
        # This method is called when the window is about to be closed
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.shutdown()
            event.accept()
        else:
            event.ignore()


def ui_process(input_queue, output, shared_memory_final, shared_memory_classification, 
               shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
               shared_memory_timing, final_lock, timing_lock, start_event, shutdown_event, shared_config):

    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI(start_event, shared_config)
    
    ui_thread = UIThread(input_queue, output, shared_memory_final, shared_memory_classification, 
                         shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                         shared_memory_timing, final_lock, timing_lock, window, shared_config)
    ui_thread.update_fov.connect(window.update_fov_list)
    ui_thread.update_images.connect(window.update_cropped_images)
    ui_thread.update_coordinates.connect(lambda fov_id, images, scores, coords: window.update_cropped_images(fov_id, images, scores, coords))
    ui_thread.update_rbc.connect(window.update_rbc_count)
    ui_thread.update_fov_image.connect(window.update_fov_image)
    
    def handle_shutdown():
        shutdown_event.set()
        app.quit()
    
    window.shutdown_signal.connect(handle_shutdown)
    
    ui_thread.start()
    
    window.show()
    app.exec_()
    shutdown_event.set()  # Ensure shutdown_event is set when app closes

def start_ui(input_queue, output, shared_memory_final, shared_memory_classification, shared_memory_segmentation, 
             shared_memory_acquisition, shared_memory_dpc, shared_memory_timing, final_lock, timing_lock,start_event):
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI(start_event)
    
    ui_thread = UIThread(input_queue, output, shared_memory_final, shared_memory_classification, shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, shared_memory_timing, final_lock, timing_lock, window)
    ui_thread.update_fov.connect(window.update_fov_list)
    ui_thread.update_images.connect(window.update_cropped_images)
    ui_thread.update_rbc.connect(window.update_rbc_count)
    ui_thread.update_fov_image.connect(window.update_fov_image)
    
    def shutdown():
        app.quit()
    
    ui_thread.connect_shutdown(shutdown)

    ui_thread.start()
    
    window.show()
    sys.exit(app.exec_())