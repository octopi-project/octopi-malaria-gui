from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListView
from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle

class ImageItem:
    def __init__(self, image, score, fov_id, coordinates=None):
        self.image = image
        self.score = score
        self.fov_id = fov_id
        self.coordinates = coordinates  # Store coordinates for each image

class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return f"Score: {self.items[index.row()].score:.2f}"
        elif role == Qt.DecorationRole:
            return self.items[index.row()].image
        elif role == Qt.UserRole:  # Custom role for coordinates
            return self.items[index.row()].coordinates

    def addItem(self, image, score, fov_id, coordinates=None):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.items.append(ImageItem(image, score, fov_id, coordinates))
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()

class ImageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_size = QSize(150, 190)  # Adjusted size to accommodate FOV ID

    def paint(self, painter, option, index):
        image = index.data(Qt.DecorationRole)
        text = index.data(Qt.DisplayRole)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Check if item is selected to add highlighting
        if option.state & QStyle.State_Selected:
            # Draw selection background
            selection_color = QColor("#3498DB")  # Blue background for selection
            selection_color.setAlpha(40)  # Semi-transparent
            painter.fillRect(option.rect, selection_color)
            
            # Draw border
            pen = QPen(QColor("#3498DB"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))

        # Draw image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_rect = QRect(option.rect.x() + 5, option.rect.y() + 5, 140, 140)
        painter.drawPixmap(image_rect, scaled_pixmap)

        # Draw text (centered)
        if option.state & QStyle.State_Selected:
            painter.setPen(QColor("#2C3E50"))  # Darker text for selected items
        else:
            painter.setPen(QColor("#34495E"))  # Normal text color
        text_rect = QRect(option.rect.x(), option.rect.y() + 150, 150, 40)
        painter.drawText(text_rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index):
        return self.item_size

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

    def update_images(self, images, fov_id, coordinates=None):
        # If coordinates are provided, they should be a list matching the images
        for i, (image, score) in enumerate(images):
            coords = coordinates[i] if coordinates is not None and i < len(coordinates) else None
            index = self.model.rowCount()
            self.model.addItem(image, score, fov_id, coords)
    
    def _on_image_clicked(self, index):
        # Emit signal with coordinates when an image is clicked
        coordinates = index.data(Qt.UserRole)
        if coordinates is not None:
            self.image_clicked.emit(coordinates)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
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

    def update_images(self, images, fov_id, coordinates=None):
        self.image_list.clear()
        self.image_list.update_images(images, fov_id, coordinates)

    def _on_image_clicked(self, coordinates):
        # Forward the signal
        self.image_clicked.emit(coordinates)