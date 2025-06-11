from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListView
from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle, QHBoxLayout, QPushButton

class ImageItem:
    def __init__(self, image, score, fov_id, coordinates=None, annotation_type='model'):
        self.image = image
        self.score = score
        self.fov_id = fov_id
        self.coordinates = coordinates  # Store coordinates for each image
        self.annotation_type = annotation_type  # Track if this is model prediction or manual annotation

class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.threshold = 0.5  # Default threshold, will be updated from UI

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            item = self.items[index.row()]
            score = item.score
            
            # Check if this item has annotation type information
            annotation_type = getattr(item, 'annotation_type', 'model')
            
            if annotation_type == 'manual_positive':
                return "M-Pos"
            elif annotation_type == 'manual_negative':
                return "M-Neg"
            elif annotation_type == 'manual_unsure':
                return "M-Unsure"
            else:  # annotation_type == 'model' or not set
                if score >= self.threshold:
                    return f"Pos: {score:.2f}"
                else:
                    return f"Neg: {score:.2f}"
        elif role == Qt.DecorationRole:
            return self.items[index.row()].image
        elif role == Qt.UserRole:  # Custom role for coordinates
            return self.items[index.row()].coordinates
        elif role == Qt.UserRole + 1:  # Custom role for score
            return self.items[index.row()].score
        elif role == Qt.UserRole + 2:  # Custom role for annotation type
            return getattr(self.items[index.row()], 'annotation_type', 'model')

    def addItem(self, image, score, fov_id, coordinates=None, annotation_type='model'):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.items.append(ImageItem(image, score, fov_id, coordinates, annotation_type))
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()
        
    def setThreshold(self, threshold):
        self.threshold = threshold
        # Notify view that data has changed to trigger repaint
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, 0))

class ImageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_size = QSize(150, 190)  # Adjusted size to accommodate FOV ID

    def paint(self, painter, option, index):
        image = index.data(Qt.DecorationRole)
        text = index.data(Qt.DisplayRole)
        score = index.data(Qt.UserRole + 1)
        annotation_type = index.data(Qt.UserRole + 2)
        model = index.model()
        threshold = getattr(model, 'threshold', 0.5)  # Get threshold from model or use default
        
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Check if item is selected to add highlighting
        if option.state & QStyle.State_Selected:
            # Draw selection background
            selection_color = QColor("#3498DB")  # Blue background for selection
            selection_color.setAlpha(40)  # Semi-transparent
            painter.fillRect(option.rect, selection_color)
            
            # Draw border for selected items
            pen = QPen(QColor("#3498DB"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))
        
        # Draw image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_rect = QRect(option.rect.x() + 5, option.rect.y() + 5, 140, 140)
        
        # Draw normal image
        painter.drawPixmap(image_rect, scaled_pixmap)

        # Draw text (centered)
        if option.state & QStyle.State_Selected:
            painter.setPen(QColor("#2C3E50"))  # Darker text for selected items
        else:
            # Set text color based on annotation type and score
            if annotation_type == 'manual_positive':
                painter.setPen(QColor("#E74C3C"))  # Red for manual positive
            elif annotation_type == 'manual_negative':
                painter.setPen(QColor("#3498DB"))  # Blue for manual negative
            elif annotation_type == 'manual_unsure':
                painter.setPen(QColor("#95A5A6"))  # Grey for manual unsure
            else:  # annotation_type == 'model' or not set
                # Use original threshold-based coloring for model predictions
                if score >= threshold:
                    painter.setPen(QColor("#E74C3C"))  # Red text for above threshold
                else:
                    painter.setPen(QColor("#3498DB"))  # Blue text for below threshold
                
        text_rect = QRect(option.rect.x(), option.rect.y() + 150, 150, 40)
        painter.drawText(text_rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index):
        return self.item_size

class ReportImageDelegate(ImageDelegate):
    """A specialized image delegate that uses a consistent text color regardless of score."""
    
    def paint(self, painter, option, index):
        image = index.data(Qt.DecorationRole)
        text = index.data(Qt.DisplayRole)
        score = index.data(Qt.UserRole + 1)
        annotation_type = index.data(Qt.UserRole + 2)
        
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Check if item is selected to add highlighting
        if option.state & QStyle.State_Selected:
            # Draw selection background
            selection_color = QColor("#3498DB")  # Blue background for selection
            selection_color.setAlpha(40)  # Semi-transparent
            painter.fillRect(option.rect, selection_color)
            
            # Draw border for selected items
            pen = QPen(QColor("#3498DB"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))
        
        # Draw image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_rect = QRect(option.rect.x() + 5, option.rect.y() + 5, 140, 140)
        
        # Draw normal image
        painter.drawPixmap(image_rect, scaled_pixmap)

        # Draw text (centered) with consistent color
        if option.state & QStyle.State_Selected:
            painter.setPen(QColor("#2C3E50"))  # Darker text for selected items
        else:
            # Use a consistent text color for all scores (dark gray)
            painter.setPen(QColor("#2C3E50"))
                
        text_rect = QRect(option.rect.x(), option.rect.y() + 150, 150, 40)
        painter.drawText(text_rect, Qt.AlignCenter, text)

        painter.restore()

class VirtualImageListWidget(QWidget):
    image_clicked = pyqtSignal(object)  # Signal for when an image is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.list_view = QListView()
        self.model = ImageListModel()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(ImageDelegate())
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setSpacing(10)
        self.list_view.clicked.connect(self._on_image_clicked)
        
        # Enable selection and set selection behavior
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.setSelectionBehavior(QListView.SelectItems)
        
        # Set some style properties for better visual feedback
        self.list_view.setStyleSheet("""
            QListView {
                background-color: white;
                outline: none;
            }
            QListView::item:hover {
                background-color: rgba(52, 152, 219, 0.1);
            }
        """)
        
        self.layout.addWidget(self.list_view)

    def clear(self):
        self.model.clear()

    def update_images(self, images, fov_id, coordinates=None, annotation_types=None):

        for i, (image, score) in enumerate(images):
            coords = coordinates[i] if coordinates is not None and i < len(coordinates) else None
            annotation_type = annotation_types[i] if annotation_types is not None and i < len(annotation_types) else 'model'
            
            self.model.addItem(image, score, fov_id, coords, annotation_type)
    
    def _on_image_clicked(self, index):
        # Emit signal with coordinates when an image is clicked
        coordinates = index.data(Qt.UserRole)
        if coordinates is not None:
            self.image_clicked.emit(coordinates)
            
    def set_threshold(self, threshold):
        """Update the threshold value used for coloring spots"""
        self.model.setThreshold(threshold)

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class ExpandableImageWidget(QWidget):
    image_clicked = pyqtSignal(object)  # Signal for when an image is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Image list
        self.image_list = VirtualImageListWidget()
        self.image_list.image_clicked.connect(self._on_image_clicked)
        self.layout.addWidget(self.image_list)

        # Default state is shown
        self.image_list.show()

    def update_images(self, images, fov_id, coordinates=None, annotation_types=None):
        self.image_list.clear()
        self.image_list.update_images(images, fov_id, coordinates, annotation_types)

    def _on_image_clicked(self, coordinates):
        # Forward the signal
        self.image_clicked.emit(coordinates)
        
    def set_threshold(self, threshold):
        """Update the threshold value used for coloring spots"""
        self.image_list.set_threshold(threshold)

# Add new AnnotationListWidget for handling annotations
class AnnotationFileItem:
    def __init__(self, filename, version, timestamp, fov_id):
        self.filename = filename
        self.version = version
        self.timestamp = timestamp
        self.fov_id = fov_id
        self.display_name = f"Version: {version} - {timestamp}"

class AnnotationListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role):
        if not index.isValid():
            return None

        item = self.items[index.row()]
        
        if role == Qt.DisplayRole:
            return item.display_name
        elif role == Qt.BackgroundRole:
            return QColor(220, 220, 220, 100)  # Light gray for all items
        elif role == Qt.UserRole:  # Custom role for the annotation file data
            return item

    def addItem(self, filename, version, timestamp, fov_id):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.items.append(AnnotationFileItem(filename, version, timestamp, fov_id))
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()

class AnnotationListWidget(QWidget):
    annotation_file_selected = pyqtSignal(object)  # Signal when annotation file is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Annotation list
        self.list_view = QListView()
        self.model = AnnotationListModel()
        self.list_view.setModel(self.model)
        
        # Set selection mode
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.setSelectionBehavior(QListView.SelectItems)
        
        # Connect signals
        self.list_view.clicked.connect(self._on_annotation_file_selected)
        
        # Set a fixed height that's not too tall
        self.list_view.setMaximumHeight(150)
        
        self.layout.addWidget(self.list_view)
        
        # Current FOV ID
        self.current_fov_id = None

    def update_annotation_files(self, files_info, fov_id):
        """Update the list with available annotation files"""
        self.model.clear()
        self.current_fov_id = fov_id
        
        for file_info in files_info:
            self.model.addItem(
                file_info['filename'],
                file_info['version'],
                file_info['timestamp'],
                fov_id
            )
    
    def clear(self):
        """Clear all annotation files"""
        self.model.clear()
        self.current_fov_id = None
    
    def _on_annotation_file_selected(self, index):
        """Handle annotation file selection"""
        item = index.data(Qt.UserRole)
        if item:
            self.annotation_file_selected.emit(item)

class ExpandableAnnotationWidget(QWidget):
    annotation_file_selected = pyqtSignal(object)  # Signal when annotation file is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Toggle button
        self.toggle_button = QPushButton("Show Annotation Versions")
        self.toggle_button.clicked.connect(self._toggle_visibility)
        self.layout.addWidget(self.toggle_button)

        # Annotation list
        self.annotation_list = AnnotationListWidget()
        self.annotation_list.annotation_file_selected.connect(self._on_annotation_file_selected)
        self.layout.addWidget(self.annotation_list)

        # Default state is hidden
        self.annotation_list.hide()
        self.is_expanded = False

    def _toggle_visibility(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.annotation_list.show()
            self.toggle_button.setText("Hide Annotation Versions")
        else:
            self.annotation_list.hide()
            self.toggle_button.setText("Show Annotation Versions")

    def update_annotation_files(self, files_info, fov_id):
        self.annotation_list.update_annotation_files(files_info, fov_id)

    def clear(self):
        self.annotation_list.clear()

    def _on_annotation_file_selected(self, item):
        # Forward the signal
        self.annotation_file_selected.emit(item)