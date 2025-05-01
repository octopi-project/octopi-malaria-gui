from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListView
from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle

class ImageItem:
    def __init__(self, image, score, fov_id, coordinates=None, class_name=None):
        self.image = image
        self.score = score
        self.fov_id = fov_id
        self.coordinates = coordinates  # Store coordinates for each image
        self.class_name = class_name  # Store class for styling

class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.annotation_mode = False

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            item = self.items[index.row()]
            if self.annotation_mode:
                # In annotation mode, show only class
                return f"{item.class_name}"
            else:
                # In normal mode, show class and score
                if item.class_name:
                    return f"{item.class_name}: {item.score:.2f}"
                else:
                    return f"Score: {item.score:.2f}"
        elif role == Qt.DecorationRole:
            return self.items[index.row()].image
        elif role == Qt.UserRole:  # Custom role for coordinates
            return self.items[index.row()].coordinates
        elif role == Qt.UserRole + 1:  # Custom role for class
            return self.items[index.row()].class_name

    def addItem(self, image, score, fov_id, coordinates=None, class_name=None):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.items.append(ImageItem(image, score, fov_id, coordinates, class_name))
        self.endInsertRows()

    def setAnnotationMode(self, enabled):
        """Set whether the model is in annotation mode"""
        if self.annotation_mode != enabled:
            self.annotation_mode = enabled
            self.layoutChanged.emit()

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
        class_name = index.data(Qt.UserRole + 1)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Set border color based on class
        border_color = QColor("#3498DB")  # Default blue
        if class_name == "Parasite":
            border_color = QColor("#e74c3c")  # Red for parasites
        elif class_name == "Negative":
            border_color = QColor("#3498DB")  # Blue for non-parasites

        # Check if item is selected to add highlighting
        if option.state & QStyle.State_Selected:
            # Draw selection background
            selection_color = border_color.lighter(150)
            selection_color.setAlpha(40)  # Semi-transparent
            painter.fillRect(option.rect, selection_color)
            
            # Draw border
            pen = QPen(border_color)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))
        elif option.state & QStyle.State_MouseOver:
            # Hover effect
            hover_color = border_color.lighter(170)
            hover_color.setAlpha(30)  # More transparent
            painter.fillRect(option.rect, hover_color)
            
            # Draw lighter border
            pen = QPen(border_color.lighter(130))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))

        # Draw image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_rect = QRect(option.rect.x() + 5, option.rect.y() + 5, 140, 140)
        painter.drawPixmap(image_rect, scaled_pixmap)

        # Draw text with class-specific styling
        if class_name == "Parasite":
            painter.setPen(QColor("#e74c3c"))  # Red for parasites
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
        elif class_name == "Negative":
            painter.setPen(QColor("#3498DB"))  # Blue for non-parasites
        else:
            painter.setPen(QColor("#34495E"))  # Default color
        
        text_rect = QRect(option.rect.x(), option.rect.y() + 150, 150, 40)
        painter.drawText(text_rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index):
        return self.item_size

class VirtualImageListWidget(QWidget):
    image_clicked = pyqtSignal(object)  # Signal for when an image is clicked (coordinates or index)
    
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

    def setAnnotationMode(self, enabled):
        """Set whether the widget is in annotation mode"""
        self.model.setAnnotationMode(enabled)

    def clear(self):
        self.model.clear()

    def update_images(self, images, fov_id, coordinates=None):
        # If coordinates are provided, they should be a list matching the images
        for i, (image, score) in enumerate(images):
            coords = coordinates[i] if coordinates is not None and i < len(coordinates) else None
            index = self.model.rowCount()
            self.model.addItem(image, score, fov_id, coords)
            
    def update_images_with_classes(self, images, fov_id, coordinates=None, classes=None):
        # If coordinates and classes are provided, they should be lists matching the images
        for i, (image, score) in enumerate(images):
            coords = coordinates[i] if coordinates is not None and i < len(coordinates) else None
            class_name = classes[i] if classes is not None and i < len(classes) else None
            index = self.model.rowCount()
            self.model.addItem(image, score, fov_id, coords, class_name)
    
    def _on_image_clicked(self, index):
        # Emit signal with coordinates when an image is clicked
        coordinates = index.data(Qt.UserRole)
        if coordinates is not None:
            self.image_clicked.emit(coordinates)

    def update_annotation_image(self, index, image, class_name, score=0.5):
        """Update a specific image in the list (used for annotations)"""
        model = self.model
        if index < 0 or index >= model.rowCount():
            # Add new image if it doesn't exist
            model.addItem(image, score, "annotation", None, class_name)
        else:
            # Update existing image
            if index < len(model.items):
                model.beginResetModel()
                model.items[index].image = image
                model.items[index].class_name = class_name
                model.endResetModel()
        
    def remove_annotation_image(self, index):
        """Remove a specific image from the list"""
        model = self.model
        if 0 <= index < len(model.items):
            model.beginResetModel()
            del model.items[index]
            model.endResetModel()

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
        
    def setAnnotationMode(self, enabled):
        """Set whether the widget is in annotation mode"""
        self.image_list.setAnnotationMode(enabled)

    def update_images(self, images, fov_id, coordinates=None):
        self.image_list.clear()
        self.image_list.update_images(images, fov_id, coordinates)
        
    def update_images_with_classes(self, images, fov_id, coordinates=None, classes=None):
        self.image_list.clear()
        self.image_list.update_images_with_classes(images, fov_id, coordinates, classes)
        
    def update_annotation_image(self, index, image, class_name, score=0.5):
        """Update a specific image in the list (used for annotations)"""
        self.image_list.update_annotation_image(index, image, class_name, score)
        
    def remove_annotation_image(self, index):
        """Remove a specific image from the list"""
        self.image_list.remove_annotation_image(index)

    def _on_image_clicked(self, coordinates):
        # Forward the signal
        self.image_clicked.emit(coordinates)