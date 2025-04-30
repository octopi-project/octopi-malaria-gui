import sys
import numpy as np
import threading
from queue import Empty
from utils import numpy2png_ui as numpy2png
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QStyleFactory, QFileDialog,
    QComboBox, QCheckBox, QGroupBox, QGridLayout,QSpinBox, QFrame, QDialog, QDoubleSpinBox
)
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg
from widgets import VirtualImageListWidget, ExpandableImageWidget

import time, os

from utils import SharedConfig

import cv2

MINIMUM_SCORE_THRESHOLD = 0.5  # Adjust this value as needed

class CustomROI(pg.ROI):
    """Custom ROI class with click handling and state management"""
    def __init__(self, pos, size, parent=None, index=None, **kwargs):
        super().__init__(pos, size, **kwargs)
        self.parent = parent
        self.index = index
        self.setAcceptHoverEvents(True)
        
        # Get pen styles from parent if available, otherwise use defaults
        if parent is not None and hasattr(parent, 'normal_bbox_pen'):
            self.normal_pen = parent.normal_bbox_pen
            self.selected_pen = parent.selected_bbox_pen
            self.hover_pen = parent.hover_bbox_pen
        else:
            # Define pen styles for different states as fallback
            self.normal_pen = pg.mkPen('r', width=1)  # Red, thin pen for normal state
            self.selected_pen = pg.mkPen('y', width=5)  # Yellow, thick pen for selected state
            self.hover_pen = pg.mkPen('r', width=2)  # Red, slightly thicker pen for hover state
        
        # Set initial state
        self.setPen(self.normal_pen)
        self.hoverPen = self.hover_pen
        
    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            print(f"ROI clicked directly: index={self.index}")
            if self.parent is not None:
                self.parent.on_bbox_clicked(self.index)
        else:
            super().mouseClickEvent(ev)
            
    def set_state(self, state):
        """Set the visual state of the bounding box"""
        if state == 'normal':
            self.setPen(self.normal_pen)
        elif state == 'selected':
            self.setPen(self.selected_pen)
        elif state == 'hover':
            self.setPen(self.hover_pen)

class ImageAnalysisUI(QMainWindow):
    shutdown_signal = pyqtSignal()
    
    def load_styles(self, filename):
        """Load CSS styles from an external file"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Style file {filename} not found")
            return ""

    def update_button_style(self, button):
        """Helper method to recalculate button styles"""
        button.style().unpolish(button)
        button.style().polish(button)

    @property
    def report_data_cache(self):
        """Lazy-loaded property for report data cache"""
        if not hasattr(self, '_report_data_cache'):
            self._report_data_cache = None
        return self._report_data_cache
        
    @report_data_cache.setter
    def report_data_cache(self, value):
        self._report_data_cache = value

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
        
        # Load styles from external file
        self.setStyleSheet(self.load_styles("styles.css"))
        
        self.image_lock = threading.Lock()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.patient_id = ""
        
        # Initialize bounding box related attributes
        self.bbox_items = []
        self.selected_bbox_index = None
        self.normal_bbox_pen = pg.mkPen('r', width=1)  # Red, thin pen for normal state
        self.selected_bbox_pen = pg.mkPen('y', width=5)  # Yellow, thick pen for selected state
        self.hover_bbox_pen = pg.mkPen('r', width=2)  # Red, slightly thicker pen for hover state
        
        self.setup_ui()

        self.image_cache = {}
        # Current FOV images stored directly instead of in caches
        self.current_overlay_image = None
        self.current_dpc_image = None 
        self.current_fluorescent_image = None
        self.current_segmentation_image = None
        
        self.fov_data = {}
        self.selected_fov_id = None
        self.current_view_mode = "Overlay"  # Default view mode

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
        self.patient_id_label.setObjectName("patientIdLabel")
        self.main_layout.addWidget(self.patient_id_label)

        # Top layout with shutdown button
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.new_patient_button = QPushButton("New Patient")
        self.new_patient_button.clicked.connect(self.new_patient)
        self.new_patient_button.setObjectName("newPatientButton")
        top_layout.addWidget(self.new_patient_button)

        # Add this after the "New Patient" button
        self.load_patient_button = QPushButton("Load Patient")
        self.load_patient_button.clicked.connect(self.load_patient)
        self.load_patient_button.setObjectName("loadPatientButton")
        top_layout.addWidget(self.load_patient_button)

        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.shutdown)
        self.shutdown_button.setObjectName("shutdownButton")
        top_layout.addWidget(self.shutdown_button)
        self.main_layout.addLayout(top_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Connect tab changed signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Start Tab
        start_tab = QWidget()

        # Card frame
        card = QFrame(self)
        card.setObjectName("card")
        card.setFixedSize(400, 500)
        card_layout = QVBoxLayout(card)

        # Welcome label
        welcome_label = QLabel("Welcome to Octopi")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setObjectName("welcomeLabel")
        card_layout.addWidget(welcome_label)

        # Patient ID input
        patient_id_layout = QVBoxLayout()
        patient_id_label = QLabel("Patient ID:")
        patient_id_label.setStyleSheet("""
            margin-bottom: 0px; 
        """)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter Patient ID")
        self.patient_id_input.setObjectName("patientIdInput")
        patient_id_layout.addWidget(patient_id_label)
        patient_id_layout.addWidget(self.patient_id_input)
        card_layout.addLayout(patient_id_layout)

        # To Loading Position button
        self.loading_position_button = QPushButton("To Loading Position")
        self.loading_position_button.setObjectName("loadingPositionButton")
        card_layout.addWidget(self.loading_position_button)
        self.loading_position_button.clicked.connect(self.move_to_loading_position)

        # Start Scanning button
        self.start_button = QPushButton("Start Scanning")
        self.start_button.setObjectName("startButton")
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
        
        # Add FOV column title
        fov_title = QLabel("Field of View")
        fov_title.setAlignment(Qt.AlignCenter)
        fov_title.setProperty("class", "columnTitle")
        left_layout.addWidget(fov_title)

        # Add view mode selector
        view_mode_layout = QHBoxLayout()
        view_mode_layout.addWidget(QLabel("Channels:"))
        self.view_mode_selector = QComboBox()
        self.view_mode_selector.addItems(["Overlay", "DPC", "Fluorescent", "Segmentation"])
        self.view_mode_selector.setCurrentText("Overlay")
        self.view_mode_selector.currentTextChanged.connect(self.switch_view_mode)
        view_mode_layout.addWidget(self.view_mode_selector)
        view_mode_layout.addStretch()
        left_layout.addLayout(view_mode_layout)

        self.fov_image_view = pg.ImageView()
        self.setup_fov_image_view(self.fov_image_view)
        left_layout.addWidget(self.fov_image_view)

        splitter.addWidget(fov_image_widget)

        # Middle column: Positive spots display
        positive_spots_widget = QWidget()
        middle_layout = QVBoxLayout(positive_spots_widget)

        # Add Spot images column title
        spots_title = QLabel("Positive Spots")
        spots_title.setAlignment(Qt.AlignCenter)
        spots_title.setProperty("class", "columnTitle")
        middle_layout.addWidget(spots_title)

        # Add sorting controls for positive spots
        spots_sort_layout = QHBoxLayout()
        spots_sort_layout.addWidget(QLabel("Sort by score:"))
        self.spots_sort_combo = QComboBox()
        self.spots_sort_combo.addItems(["No sorting", "Highest to lowest", "Lowest to highest"])
        self.spots_sort_combo.currentIndexChanged.connect(self.sort_positive_spots)
        spots_sort_layout.addWidget(self.spots_sort_combo)
        spots_sort_layout.addStretch(1)
        middle_layout.addLayout(spots_sort_layout)

        self.positive_images_widget = ExpandableImageWidget()
        middle_layout.addWidget(self.positive_images_widget)

        splitter.addWidget(positive_spots_widget)

        # Right column: FOV list
        fov_list_widget = QWidget()
        right_layout = QVBoxLayout(fov_list_widget)

        # Add FOV list column title
        fov_list_title = QLabel("FOV List")
        fov_list_title.setAlignment(Qt.AlignCenter)
        fov_list_title.setProperty("class", "columnTitle")
        right_layout.addWidget(fov_list_title)

        # Add threshold slider for FOV view
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.fov_threshold_spinbox = QDoubleSpinBox()
        self.fov_threshold_spinbox.setRange(0.0, 1.0)
        self.fov_threshold_spinbox.setSingleStep(0.01)
        self.fov_threshold_spinbox.setValue(MINIMUM_SCORE_THRESHOLD)
        self.fov_threshold_spinbox.setDecimals(2)
        self.fov_threshold_spinbox.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.fov_threshold_spinbox)
        self.fov_threshold_value = QLabel(f"{MINIMUM_SCORE_THRESHOLD:.2f}")
        right_layout.addLayout(threshold_layout)

        # Add a new label for average processing time
        self.avg_processing_time_label = QLabel("Avg Processing Time: N/A")
        self.avg_processing_time_label.setObjectName("avgProcessingTimeLabel")
        right_layout.addWidget(self.avg_processing_time_label)

        # Timer to update average processing time
        self.update_avg_timer = QTimer(self)
        self.update_avg_timer.timeout.connect(self.update_avg_processing_time)
        self.update_avg_timer.start(1000) 

        self.stats_label_small = QLabel("FoVs: 0 | RBCs: 0 | Parasites / μl: 0")
        self.stats_label_small.setObjectName("statsLabelSmall")
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
        self.stats_label.setObjectName("statsLabel")
        self.cropped_layout.addWidget(self.stats_label)

        # Add sorting controls
        sort_controls_layout = QHBoxLayout()
        sort_controls_layout.addWidget(QLabel("Sort by score:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["No sorting", "Highest to lowest", "Lowest to highest"])
        self.sort_combo.currentIndexChanged.connect(self.sort_report_images)
        sort_controls_layout.addWidget(self.sort_combo)
        sort_controls_layout.addStretch(1)
        self.cropped_layout.addLayout(sort_controls_layout)

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
        self.live_button.setObjectName("liveButton")
        left_layout.addWidget(self.live_button)

        # Live position display
        self.live_position_label = QLabel("X: 0, Y: 0, Z: 0")
        self.live_position_label.setObjectName("livePositionLabel")
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
        """Central method for updating threshold across the application"""
        global MINIMUM_SCORE_THRESHOLD
        MINIMUM_SCORE_THRESHOLD = value
        
        # Keep all UI elements displaying threshold in sync
        # Update FOV slider if it exists and has a different value
        if hasattr(self, 'fov_threshold_spinbox') and self.fov_threshold_spinbox.value() != value:
            self.fov_threshold_spinbox.setValue(value)
            
        # Update settings tab threshold input if it exists and has a different value
        if hasattr(self, 'threshold_input') and self.threshold_input.value() != value:
            self.threshold_input.setValue(value)
            
        # Update threshold display value if it exists
        if hasattr(self, 'fov_threshold_value'):
            self.fov_threshold_value.setText(f"{value:.2f}")
            
        # Update FOV list counts and positive images if data exists
        if hasattr(self, 'fov_data') and self.fov_data:
            # Update counts for each FOV
            for fov_id in list(self.fov_data.keys()):
                self.recalculate_fov_positives(fov_id)
            
            # Update displayed positive images if there's a selected FOV
            if self.selected_fov_id:
                self.update_positive_images(self.selected_fov_id)
                # Refresh the bounding boxes to match the new threshold
                self.display_all_bounding_boxes()
            
            # Clear the report cache since threshold has changed
            self.report_data_cache = None

    def recalculate_fov_positives(self, fov_id):
        """Recalculate positives for an FOV based on current threshold"""
        # Load the scores for the FOV
        path = self.shared_config.get_path()
        scores_path = os.path.join(path, f"{fov_id}_scores.npy")
        
        if os.path.exists(scores_path):
            try:
                scores = np.load(scores_path)
                # Count positives based on current threshold
                malaria_positives = sum(1 for score in scores if score >= MINIMUM_SCORE_THRESHOLD)
                # Update the count in the FOV data
                self.update_malaria_positives(fov_id, malaria_positives)
            except Exception as e:
                self.logger.error(f"Error recalculating positives for FOV {fov_id}: {e}")

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

    def setup_fov_image_view(self, image_view):
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.ui.histogram.hide()
        image_view.view.setMouseEnabled(x=True, y=True)
        image_view.view.setBackgroundColor((255, 255, 255))
        # Initialize bounding box related attributes in a consistent way
        # (we've already created bbox_items in __init__, this is just to be safe)
        if not hasattr(self, 'bbox_items'):
            self.bbox_items = []
        if not hasattr(self, 'selected_bbox_index'):
            self.selected_bbox_index = None

    def load_channels(self):
        try:
            tree = ET.parse('config/channel_configurations.xml')
            root = tree.getroot()
            channels = [mode.get('Name') for mode in root.findall('mode')]
            self.shared_config.set_channels_list(channels)
            self.channel_combo.addItems(channels)
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML: {e}")
        except FileNotFoundError:
            self.logger.error("config/channel_configurations.xml file not found")

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
        self.live_button.setProperty("active", True)
        # Force style recalculation
        self.update_button_style(self.live_button)
        self.live_view_timer.start()
        self.shared_config.is_live_view_active.value = True

    def stop_live_view(self):
        self.live_button.setText("LIVE")
        self.live_button.setProperty("active", False)
        # Force style recalculation
        self.update_button_style(self.live_button)
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
                    self.loading_position_button.setProperty("state", "loading")
                    # Force style recalculation
                    self.update_button_style(self.loading_position_button)
        else:
            with self.shared_config.position_lock:
                if not self.shared_config.to_loading.value:
                    self.shared_config.set_to_scanning()
                    self.loading_position_button.setText("To Loading Position")
                    self.loading_position_button.setProperty("state", "")
                    # Force style recalculation
                    self.update_button_style(self.loading_position_button)

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
        self.current_overlay_image = None
        self.current_dpc_image = None
        self.current_fluorescent_image = None
        self.current_segmentation_image = None

        self.fov_data.clear()
        self.fov_image_data.clear()

        # Clear all bounding boxes
        self.clear_all_bounding_boxes()

        # Reset UI elements
        self.fov_table.setRowCount(0)
        self.virtual_image_list.clear()
        self.fov_image_view.clear()
        self.patient_id_label.setText("")
        self.positive_images_widget.image_list.clear()
        self.stats_label.setText("FoVs: 0 | RBCs count: 0 | Positives: 0 | Parasites / μl: 0 | Parasitemia: 0%")
        self.stats_label_small.setText("FoVs: 0 | RBCs: 0 | Parasites / μl: 0")
        
        # Reset other variables
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
        """Modified to update only the FOV-specific info, not accumulate images in the report"""
        with self.image_lock:
            # Count malaria positives but don't accumulate images
            malaria_positives = sum(1 for score in scores if score >= MINIMUM_SCORE_THRESHOLD)
            self.update_malaria_positives(fov_id, malaria_positives)
            
            # Filter images and coordinates that meet threshold criteria
            filtered_indices = [i for i, score in enumerate(scores) if score >= MINIMUM_SCORE_THRESHOLD]
            
            # Process and store only images that meet threshold
            updated_images = []
            updated_coords = []
            
            for idx in filtered_indices:
                img = images[idx]
                score = scores[idx]
                overlay_img = numpy2png(img, resize_factor=None)
                
                if overlay_img is not None:
                    qimg = self.create_qimage(overlay_img)
                    updated_images.append((qimg, score))
                    
                    # Get coordinate if available
                    coord = coordinates[idx] if coordinates is not None and idx < len(coordinates) else None
                    updated_coords.append(coord)
            
            self.fov_image_data[fov_id] = updated_images
            self.fov_coordinates_data[fov_id] = updated_coords

        # Update only the positive images widget if this is the selected FOV
        if fov_id == self.selected_fov_id:
            self.update_positive_images(fov_id)

    def update_all_fov_images(self):
        # Don't update the virtual image list here
        # The report will be generated on-demand when clicking the tab
        self.update_stats()
    
    def create_qimage(self, overlay_img):
        """Convert numpy array to QImage with caching based on array data"""
        # Hash the image data for caching
        img_hash = hash(overlay_img.tobytes())
        
        # Check cache first
        if hasattr(self, '_qimage_cache') and img_hash in self._qimage_cache:
            return self._qimage_cache[img_hash]
            
        # Create new QImage if not in cache
        height, width, channel = overlay_img.shape
        bytes_per_line = 3 * width
        qimg = QImage(overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create cache dictionary if it doesn't exist
        if not hasattr(self, '_qimage_cache'):
            self._qimage_cache = {}
            
        # Cache the image (limit cache size to 100 images)
        if len(self._qimage_cache) > 100:
            # Clear oldest entries if cache gets too big
            self._qimage_cache = {}
        self._qimage_cache[img_hash] = qimg
        
        return qimg

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
        # Clear all bounding boxes
        self.clear_all_bounding_boxes()
        
        # Store each image type directly
        
        # Store DPC image
        self.current_dpc_image = dpc_image.copy()
        
        # Store fluorescent image
        self.current_fluorescent_image = fluorescent_image.copy()
        
        # Create and store overlay image
        self.current_overlay_image = self.create_overlay(dpc_image, fluorescent_image)
        
        self.selected_fov_id = fov_id
        self.display_current_fov()

    def display_current_fov(self):
        if not self.selected_fov_id:
            self.logger.error("No FOV selected")
            return
            
        # Choose the appropriate image based on view mode
        if self.current_view_mode == "Overlay" and self.current_overlay_image is not None:
            self.fov_image_view.setImage(self.current_overlay_image, autoLevels=False, levels=(0, 255))
            
        elif self.current_view_mode == "DPC" and self.current_dpc_image is not None:
            # Scale DPC image for display
            dpc_display = (self.current_dpc_image * 255).astype(np.uint8)
            self.fov_image_view.setImage(dpc_display, autoLevels=False)
            
        elif self.current_view_mode == "Fluorescent" and self.current_fluorescent_image is not None:
            # Fix the channel order for fluorescent images - swap R and B channels
            # This matches the channel order used in numpy2png_ui (where channel order is [2,1,0])
            fluo_img = self.current_fluorescent_image.copy()
            if fluo_img.shape[2] == 3:  # Make sure it's a 3-channel image
                fluo_img = fluo_img[:, :, [2, 1, 0]]  # Swap R and B channels
            self.fov_image_view.setImage(fluo_img, autoLevels=False)
            
        elif self.current_view_mode == "Segmentation" and self.current_segmentation_image is not None:
            self.fov_image_view.setImage(self.current_segmentation_image, autoLevels=False)
            
        else:
            # Fall back to overlay if selected mode is not available
            if self.current_overlay_image is not None:
                self.fov_image_view.setImage(self.current_overlay_image, autoLevels=False, levels=(0, 255))
            else:
                self.logger.error(f"No {self.current_view_mode} image available for FOV {self.selected_fov_id}")
                return

        # Highlight the selected row in the table
        row = self.find_fov_row(self.selected_fov_id)
        if row is not None:
            self.fov_table.selectRow(row)

    def fov_table_item_clicked(self, item):
        fov_id = self.fov_table.item(item.row(), 0).text()
        self.load_fov_cache(fov_id)

    def load_fov_cache(self, fov_id):
        # Clear all existing bounding boxes
        self.clear_all_bounding_boxes()
        
        # Reset selected FOV ID
        self.selected_fov_id = fov_id
        
        try:
            # Load DPC and fluorescent images
            if self.shared_config.SAVE_NPY.value:
                dpc = np.load(f"{self.shared_config.get_path()}/{fov_id}_dpc.npy")
                fluorescent = np.load(f"{self.shared_config.get_path()}/{fov_id}_fluorescent.npy")
            else:
                dpc = cv2.imread(f"{self.shared_config.get_path()}/{fov_id}_dpc.bmp", cv2.IMREAD_GRAYSCALE)
                fluorescent = cv2.imread(f"{self.shared_config.get_path()}/{fov_id}_fluorescent.bmp")

            # Convert grayscale DPC to float
            if dpc.dtype == np.uint8:
                dpc = dpc.astype(np.float16) / 255.0
            
            # Store separate images
            self.current_dpc_image = dpc
            self.current_fluorescent_image = fluorescent
            
            # Try to load segmentation map if available
            try:
                seg_path = os.path.join(self.shared_config.get_path(), f"{fov_id}_segmentation_map.bmp")
                if os.path.exists(seg_path):
                    seg_map = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    self.current_segmentation_image = seg_map
                else:
                    self.current_segmentation_image = None
            except Exception as e:
                self.logger.error(f"Error loading segmentation map: {e}")
                self.current_segmentation_image = None
            
            # Create overlay image
            img_array = self.create_overlay(dpc, fluorescent)
            self.current_overlay_image = img_array
            
        except FileNotFoundError:
            self.logger.error(f"FOV {fov_id} not found on disk")
            return

        self.display_current_fov()
        self.update_positive_images(fov_id)
        
        # After updating positive images, display all bounding boxes
        self.display_all_bounding_boxes()

    def update_positive_images(self, fov_id):
        """Update the positive images display for the selected FOV based on current threshold"""
        # Clear existing images
        self.positive_images_widget.image_list.clear()
        
        try:
            # Load the raw data to filter with current threshold
            path = self.shared_config.get_path()
            cropped_path = os.path.join(path, f"{fov_id}_cropped.npy")
            scores_path = os.path.join(path, f"{fov_id}_scores.npy")
            coordinates_path = os.path.join(path, f"{fov_id}_filtered_spots.npy")
            
            if os.path.exists(cropped_path) and os.path.exists(scores_path):
                cropped_images = np.load(cropped_path)
                scores = np.load(scores_path)
                
                # Load coordinates if available
                coordinates = None
                if os.path.exists(coordinates_path):
                    coordinates = np.load(coordinates_path)
                
                # Filter by current threshold
                images_to_display = []
                coords_to_display = []
                
                for i, (img, score) in enumerate(zip(cropped_images, scores)):
                    if score >= MINIMUM_SCORE_THRESHOLD:
                        overlay_img = numpy2png(img, resize_factor=None)
                        if overlay_img is not None:
                            qimg = self.create_qimage(overlay_img)
                            images_to_display.append((qimg, score))
                            
                            # Add coordinate if available
                            if coordinates is not None and i < len(coordinates):
                                coords_to_display.append(coordinates[i])
                            else:
                                coords_to_display.append(None)
                
                # Cache the filtered data for sorting
                self.current_positive_images = {
                    'images': images_to_display,
                    'coordinates': coords_to_display
                }
                
                # Apply sorting based on current selection
                self.apply_positive_images_sort(self.spots_sort_combo.currentIndex())
                
            else:
                self.logger.info(f"No positive images for FOV {fov_id}")
                self.current_positive_images = None
        except Exception as e:
            self.logger.error(f"Error updating positive images for FOV {fov_id}: {e}")
            self.current_positive_images = None

    def apply_positive_images_sort(self, sort_mode=0):
        """Apply sorting to the positive images display without reloading data"""
        if not hasattr(self, 'current_positive_images') or self.current_positive_images is None:
            return
            
        # Get cached data
        images = self.current_positive_images['images']
        coordinates = self.current_positive_images['coordinates']
        
        if not images:
            return
            
        # Sort images if requested
        if sort_mode == 1:  # Highest to lowest
            # Sort by score in descending order
            sorted_indices = [i for i, _ in sorted(enumerate(images), 
                                                 key=lambda x: x[1][1], reverse=True)]
        elif sort_mode == 2:  # Lowest to highest
            # Sort by score in ascending order
            sorted_indices = [i for i, _ in sorted(enumerate(images), 
                                                 key=lambda x: x[1][1], reverse=False)]
        else:  # No sorting
            sorted_indices = list(range(len(images)))
        
        # Create sorted lists
        sorted_images = [images[i] for i in sorted_indices]
        sorted_coords = [coordinates[i] for i in sorted_indices]
        
        # Update display
        self.positive_images_widget.image_list.clear()
        self.positive_images_widget.update_images(sorted_images, self.selected_fov_id, sorted_coords)
        
        # Connect click signal
        try:
            self.positive_images_widget.image_clicked.disconnect()
        except:
            pass
        self.positive_images_widget.image_clicked.connect(self.on_positive_image_clicked)

    def sort_positive_spots(self, index):
        """Handle sorting change for positive spots display"""
        self.apply_positive_images_sort(sort_mode=index)

    def on_positive_image_clicked(self, coordinates):
        """Handle when a positive image is clicked to show its bounding box"""
        if coordinates is None:
            return
        
        # Show a bounding box around the spot in the FOV image
        try:
            # Coordinates are typically [x, y, radius] or [x, y]
            x, y = coordinates[0], coordinates[1]
            r = 15  # Fixed radius to ensure 31x31 box (matches cropped images)
            
            # Highlight the selected bounding box
            self.highlight_selected_bbox(coordinates)
            
            # Adjust view to center on the spot
            self.fov_image_view.view.setRange(
                xRange=(x - 2*r, x + 2*r), 
                yRange=(y - 2*r, y + 2*r),
                padding=0.5
            )
        except Exception as e:
            self.logger.error(f"Error displaying bounding box: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(200)

    def create_overlay(self, dpc_image, fluorescent_image):
        # Ensure input images are properly normalized float arrays
        dpc = dpc_image.astype(np.float32) if dpc_image.dtype != np.float32 and dpc_image.dtype != np.float16 else dpc_image
        if dpc.dtype == np.uint8:
            dpc = dpc / 255.0
            
        fluo = fluorescent_image
        if fluo.dtype == np.uint8:
            fluo = fluo.astype(np.float32) / 255.0
            
        # Stack images for numpy2png function
        img = np.stack([fluo[:,:,0], fluo[:,:,1], fluo[:,:,2], dpc], axis=0)
        return numpy2png(img, resize_factor=None)

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
        total_positives = sum(data['malaria_positives'] for data in self.fov_data.values())
        # round to two decimal places   
        parasite_per_ul = round(total_positives * (5000000 / (total_rbc + 1)), 2)
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
        # Clear all bounding boxes
        self.clear_all_bounding_boxes()
        
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
                
                # Make sure the threshold sliders are up to date
                self.update_threshold(MINIMUM_SCORE_THRESHOLD)
        except FileNotFoundError:
            print("Stats file not found")

    def update_stats_small_label(self, stats):
        # Extract relevant information from stats and update stats_label_small
        stats_parts = stats.split("|")
        fovs = stats_parts[0].strip()
        rbcs = stats_parts[1].strip()
        parasites = stats_parts[3].strip()
        self.stats_label_small.setText(f"{fovs} | {rbcs} | {parasites}")

    def switch_view_mode(self, mode):
        self.current_view_mode = mode
        self.display_current_fov()

    def on_tab_changed(self, index):
        # Clear bounding boxes when switching away from FOV tab
        if self.tab_widget.tabText(index) != "FOVs List":
            self.clear_all_bounding_boxes()
            
        if self.tab_widget.tabText(index) == "Malaria Detection Report":
            # Only generate report if needed (first time or after new data)
            if not hasattr(self, 'report_data_cache') or self.report_data_cache is None:
                self.generate_report(sort_mode=self.sort_combo.currentIndex())
            else:
                # Just re-sort existing data
                self.apply_sort_to_cached_report(self.sort_combo.currentIndex())

    def generate_report(self, sort_mode=0):
        """Generate the malaria detection report on-demand by loading images from disk"""
        # Clear existing data
        self.virtual_image_list.clear()
        
        path = self.shared_config.get_path()
        if not path or not os.path.exists(path):
            return
            
        # Reset counters for recalculation
        total_positives = 0
        total_rbc = sum(data['rbc_count'] for data in self.fov_data.values())
        
        # Accumulate all images before sorting
        all_images = []
        all_coordinates = []
        all_fov_ids = []
        
        # Process each FOV
        for fov_id in self.fov_data.keys():
            # Load cropped images, scores, and coordinates
            cropped_path = os.path.join(path, f"{fov_id}_cropped.npy")
            scores_path = os.path.join(path, f"{fov_id}_scores.npy")
            coordinates_path = os.path.join(path, f"{fov_id}_filtered_spots.npy")
            
            if os.path.exists(cropped_path) and os.path.exists(scores_path):
                try:
                    cropped_images = np.load(cropped_path)
                    scores = np.load(scores_path)
                    
                    # Load coordinates if available
                    coordinates = None
                    if os.path.exists(coordinates_path):
                        coordinates = np.load(coordinates_path)
                    
                    # Process only images that meet the current threshold
                    for i, (img, score) in enumerate(zip(cropped_images, scores)):
                        if score >= MINIMUM_SCORE_THRESHOLD:
                            total_positives += 1
                            # Convert image to QImage
                            overlay_img = numpy2png(img, resize_factor=None)
                            if overlay_img is not None:
                                qimg = self.create_qimage(overlay_img)
                                
                                # Add to accumulation lists
                                all_images.append((qimg, score))
                                all_fov_ids.append(fov_id)
                                
                                # Add coordinate if available
                                if coordinates is not None and i < len(coordinates):
                                    all_coordinates.append(coordinates[i])
                                else:
                                    all_coordinates.append(None)
                    
                except Exception as e:
                    self.logger.error(f"Error loading data for FOV {fov_id}: {e}")
        
        # Cache the loaded data
        self.report_data_cache = {
            'images': all_images,
            'coordinates': all_coordinates,
            'fov_ids': all_fov_ids,
            'total_positives': total_positives,
            'total_rbc': total_rbc
        }
        
        # Apply sorting and display
        self.apply_sort_to_cached_report(sort_mode)
    
    def apply_sort_to_cached_report(self, sort_mode=0):
        """Apply sorting to cached report data without reloading from disk"""
        if not hasattr(self, 'report_data_cache') or self.report_data_cache is None:
            # No cached data, generate full report
            self.generate_report(sort_mode)
            return
            
        # Clear the display
        self.virtual_image_list.clear()
        
        # Get cached data
        all_images = self.report_data_cache['images']
        all_coordinates = self.report_data_cache['coordinates']
        all_fov_ids = self.report_data_cache['fov_ids']
        total_positives = self.report_data_cache['total_positives']
        total_rbc = self.report_data_cache['total_rbc']
        
        # Sort images if requested
        if sort_mode == 1:  # Highest to lowest
            # Sort by score in descending order
            sorted_indices = [i for i, _ in sorted(enumerate(all_images), 
                                               key=lambda x: x[1][1], reverse=True)]
        elif sort_mode == 2:  # Lowest to highest
            # Sort by score in ascending order
            sorted_indices = [i for i, _ in sorted(enumerate(all_images), 
                                               key=lambda x: x[1][1], reverse=False)]
        else:  # No sorting (keep original FOV order)
            sorted_indices = list(range(len(all_images)))
        
        # Add sorted images to the virtual list, grouped by FOV
        current_fov = None
        current_images = []
        current_coords = []
        
        for idx in sorted_indices:
            fov_id = all_fov_ids[idx]
            image = all_images[idx]
            coord = all_coordinates[idx]
            
            if sort_mode == 0:
                # In unsorted mode, group by FOV ID
                if current_fov != fov_id:
                    # Add the previous group if it exists
                    if current_fov and current_images:
                        self.virtual_image_list.update_images(current_images, current_fov, current_coords)
                        current_images = []
                        current_coords = []
                    current_fov = fov_id
                
                current_images.append(image)
                current_coords.append(coord)
            else:
                # In sorted mode, add each image individually
                self.virtual_image_list.update_images([image], fov_id, [coord])
        
        # Add the last group if in unsorted mode
        if sort_mode == 0 and current_fov and current_images:
            self.virtual_image_list.update_images(current_images, current_fov, current_coords)
        
        # Update stats with cached values
        parasite_per_ul = round(total_positives * (5000000 / (total_rbc + 1)), 2)
        parasitemia_percentage = round(total_positives / (total_rbc + 1) * 100, 2)
        
        self.stats_label.setText(f"FoVs: {len(self.fov_data)} | Total RBC Count: {total_rbc:,} | Total Malaria Positives: {total_positives:,} | Parasites / μl: {int(parasite_per_ul):,} | Parasitemia: {parasitemia_percentage:.2f}%")
        self.stats_label_small.setText(f"FoVs: {len(self.fov_data)} | RBCs: {total_rbc:,} | Parasites / μl: {int(parasite_per_ul):,}")

    def sort_report_images(self, index):
        # Apply sorting using cached data instead of regenerating
        self.apply_sort_to_cached_report(sort_mode=index)

    def display_all_bounding_boxes(self):
        """Display bounding boxes for all positive spots in the current FOV"""
        # Clear any existing boxes first
        self.clear_all_bounding_boxes()
        
        # Check if we have coordinates for the current FOV
        if not hasattr(self, 'current_positive_images') or self.current_positive_images is None:
            return
            
        coordinates = self.current_positive_images.get('coordinates', [])
        if not coordinates:
            return
        
        print(f"Creating {len(coordinates)} bounding boxes")
        
        # Create a bounding box for each valid coordinate
        r = 15  # Fixed radius for all boxes
        self.bbox_items = []
        
        for i, coord in enumerate(coordinates):
            if coord is not None:
                x, y = coord[0], coord[1]
                
                # Create a new ROI for this spot
                bbox = CustomROI((x - r, y - r), (2*r, 2*r), 
                              parent=self, 
                              index=i)
                
                self.fov_image_view.view.addItem(bbox)
                self.bbox_items.append(bbox)
                bbox.set_state('normal')

    def highlight_selected_bbox(self, selected_coordinates):
        """Highlight the selected bounding box and reset others"""
        # Find index of bounding box with these coordinates
        selected_index = None
        if hasattr(self, 'current_positive_images') and self.current_positive_images is not None:
            coordinates = self.current_positive_images.get('coordinates', [])
            for i, coord in enumerate(coordinates):
                if coord is not None and len(coord) >= 2 and len(selected_coordinates) >= 2:
                    if coord[0] == selected_coordinates[0] and coord[1] == selected_coordinates[1]:
                        selected_index = i
                        break
        
        # If we've found the index, use the centralized selection method
        if selected_index is not None:
            print(f"Selecting bounding box from coordinates: index={selected_index}")
            self.select_bounding_box(selected_index, from_spot_click=True)
    
    def on_bbox_clicked(self, roi_index):
        """Handle clicks on bounding boxes in the FOV view"""
        try:
            print(f"Bounding box clicked: index={roi_index}")
            # Use the centralized selection method
            self.select_bounding_box(roi_index, from_bbox_click=True)
        except Exception as e:
            self.logger.error(f"Error handling bbox click: {e}")
            print(f"Error handling bbox click: {e}")
    
    def select_bounding_box(self, index, from_spot_click=False, from_bbox_click=False):
        """Centralized method to handle bounding box selection from any source"""
        # Reset only the previously selected box if it exists and is different
        if self.selected_bbox_index is not None and self.selected_bbox_index != index:
            if 0 <= self.selected_bbox_index < len(self.bbox_items):
                self.bbox_items[self.selected_bbox_index].set_state('normal')
        
        # Highlight the selected box if it exists
        if 0 <= index < len(self.bbox_items):
            self.bbox_items[index].set_state('selected')
            self.selected_bbox_index = index
            
            # Update spot list selection if click came from bounding box
            if from_bbox_click:
                # Set flag to prevent infinite loop
                self._bbox_click_triggered = True
                
                # Find and select spot in list
                if hasattr(self, 'current_positive_images') and self.current_positive_images is not None:
                    coordinates = self.current_positive_images.get('coordinates', [])
                    if index < len(coordinates):
                        selected_coord = coordinates[index]
                        if selected_coord is not None:
                            # Select the corresponding item in the positive images list
                            self.select_positive_image_by_index(index)
                            print(f"Selected positive image index: {index}")
                
                # Reset flag
                self._bbox_click_triggered = False
            
            # If click came from spot list and we're not in an infinite loop
            if from_spot_click and (not hasattr(self, '_bbox_click_triggered') or not self._bbox_click_triggered):
                pass  # No additional action needed for spot list clicks

    def clear_all_bounding_boxes(self):
        """Clear all bounding boxes"""
        for bbox in self.bbox_items:
            self.fov_image_view.view.removeItem(bbox)
        self.bbox_items = []
        self.selected_bbox_index = None
    
    def select_positive_image_by_index(self, index):
        """Select the spot image at the given index in the positive images widget"""
        try:
            # Get the list view from the positive images widget
            list_view = self.positive_images_widget.image_list.list_view
            
            # Create a model index for the item
            model_index = list_view.model().index(index, 0)
            
            # Select the item
            list_view.setCurrentIndex(model_index)
            list_view.scrollTo(model_index)
            
            # Make sure the list view has focus so the selection is visible
            list_view.setFocus()
        except Exception as e:
            self.logger.error(f"Error selecting positive image: {e}")

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
                
                # Quick check without lock to see if we should process this FOV
                if fov_id not in self.processed_fovs:
                    self.log_time(fov_id, "UI Process", "start")
                    
                    # Acquire lock only when necessary
                    process_fov = False
                    with self.final_lock:
                        if fov_id in self.shared_memory_final and not self.shared_memory_final[fov_id]['displayed']:
                            process_fov = True
                    
                    if process_fov:
                        # Process FOV outside the lock
                        self.process_fov(fov_id)
                        self.log_time(fov_id, "UI Process", "end")
                        
                        # Acquire lock again to update shared memory
                        with self.final_lock:
                            if fov_id in self.shared_memory_final:
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
        segmentation_data = self.shared_memory_segmentation.get(fov_id, {})
        
        dpc_image = dpc_data.get('dpc_image', np.array([]))
        fluorescent_image = acquisition_data.get('fluorescent', np.array([]))
        segmentation_map = segmentation_data.get('segmentation_map', np.array([]))

        if dpc_image.size > 0 and fluorescent_image.size > 0:
            # First update with the FOV image data
            self.update_fov_image.emit(fov_id, dpc_image, fluorescent_image)
            
            # If segmentation map is available, set it directly
            if segmentation_map.size > 0:
                self.window.current_segmentation_image = segmentation_map.copy()
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