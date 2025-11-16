"""
Manual review UI for annotation quality control.

Provides a PyQt5-based GUI for reviewing and editing zero-shot annotations.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QTextEdit, QProgressBar,
    QSplitter, QScrollArea, QGridLayout, QMessageBox, QFileDialog,
    QShortcut, QGroupBox
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QBrush
)

from .review_session import ReviewSession, ReviewStatus


class TileViewer(QLabel):
    """Widget for displaying a tile with mask overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")

        # Data
        self.tile_image: Optional[np.ndarray] = None
        self.mask_image: Optional[np.ndarray] = None
        self.overlay_alpha = 0.5
        self.show_mask = True

        # Editing
        self.editing_enabled = False
        self.brush_size = 10
        self.draw_mode = True  # True = draw, False = erase
        self.is_drawing = False
        self.last_point = None

    def load_tile(self, tile_path: str):
        """Load tile image."""
        try:
            with rasterio.open(tile_path) as src:
                # Read first 3 bands as RGB (or repeat if single band)
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                else:
                    # Single band - use as grayscale
                    band = src.read(1)
                    r = g = b = band

                # Normalize to 0-255
                def normalize(arr):
                    arr = arr.astype(np.float32)
                    if arr.max() > 255:
                        # Likely 0-10000 range (Sentinel-2)
                        arr = np.clip(arr / 10000.0 * 255, 0, 255)
                    return arr.astype(np.uint8)

                r = normalize(r)
                g = normalize(g)
                b = normalize(b)

                # Stack to RGB
                self.tile_image = np.stack([r, g, b], axis=-1)

        except Exception as e:
            print(f"Error loading tile: {e}")
            self.tile_image = None

    def load_mask(self, mask_path: str):
        """Load mask image."""
        try:
            if mask_path.endswith('.npz'):
                # Load from .npz format
                data = np.load(mask_path)
                self.mask_image = data['mask']
                if self.mask_image.ndim == 3:
                    self.mask_image = self.mask_image[0]
            else:
                # Load from GeoTIFF
                with rasterio.open(mask_path) as src:
                    self.mask_image = src.read(1)

            # Ensure binary
            self.mask_image = (self.mask_image > 0).astype(np.uint8)

        except Exception as e:
            print(f"Error loading mask: {e}")
            self.mask_image = None

    def get_mask(self) -> Optional[np.ndarray]:
        """Get current mask."""
        return self.mask_image

    def set_overlay_alpha(self, alpha: float):
        """Set mask overlay alpha (0-1)."""
        self.overlay_alpha = alpha
        self.update_display()

    def set_show_mask(self, show: bool):
        """Toggle mask visibility."""
        self.show_mask = show
        self.update_display()

    def update_display(self):
        """Update the displayed image."""
        if self.tile_image is None:
            self.clear()
            return

        # Start with tile image
        display = self.tile_image.copy()

        # Overlay mask if available
        if self.mask_image is not None and self.show_mask:
            # Create colored mask overlay (green for mask)
            mask_overlay = np.zeros_like(display)
            mask_overlay[self.mask_image > 0] = [0, 255, 0]  # Green

            # Blend
            display = (
                display * (1 - self.overlay_alpha) +
                mask_overlay * self.overlay_alpha
            ).astype(np.uint8)

        # Convert to QPixmap
        height, width, channel = display.shape
        bytes_per_line = 3 * width
        q_image = QImage(display.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit widget
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        """Handle mouse press for drawing."""
        if self.editing_enabled and self.mask_image is not None:
            self.is_drawing = True
            self.last_point = self._map_to_image(event.pos())
            self._draw_at_point(self.last_point)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing."""
        if self.is_drawing and self.editing_enabled:
            current_point = self._map_to_image(event.pos())
            if current_point and self.last_point:
                self._draw_line(self.last_point, current_point)
            self.last_point = current_point

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.is_drawing = False
        self.last_point = None

    def _map_to_image(self, pos):
        """Map widget position to image coordinates."""
        if self.pixmap() is None or self.tile_image is None:
            return None

        pixmap = self.pixmap()
        px_width = pixmap.width()
        px_height = pixmap.height()

        # Get position relative to pixmap
        widget_width = self.width()
        widget_height = self.height()

        x_offset = (widget_width - px_width) // 2
        y_offset = (widget_height - px_height) // 2

        x = pos.x() - x_offset
        y = pos.y() - y_offset

        if 0 <= x < px_width and 0 <= y < px_height:
            # Scale to image coordinates
            img_height, img_width = self.tile_image.shape[:2]
            img_x = int(x * img_width / px_width)
            img_y = int(y * img_height / px_height)
            return (img_x, img_y)

        return None

    def _draw_at_point(self, point):
        """Draw at a specific point."""
        if point is None or self.mask_image is None:
            return

        x, y = point
        h, w = self.mask_image.shape

        # Create circular brush
        r = self.brush_size // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.mask_image[ny, nx] = 1 if self.draw_mode else 0

        self.update_display()

    def _draw_line(self, p1, p2):
        """Draw a line between two points."""
        if p1 is None or p2 is None:
            return

        # Bresenham's line algorithm
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            self._draw_at_point((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy


class ReviewerWindow(QMainWindow):
    """Main window for annotation review."""

    def __init__(self, session: ReviewSession):
        super().__init__()
        self.session = session
        self.setWindowTitle("Annotation Reviewer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentral(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel: viewer
        viewer_panel = self._create_viewer_panel()
        main_layout.addWidget(viewer_panel, stretch=3)

        # Right panel: controls
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Setup keyboard shortcuts
        self._setup_shortcuts()

        # Load first tile
        self.load_current_tile()

        # Update stats
        self.update_statistics()

    def _create_viewer_panel(self):
        """Create the tile viewer panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Tile viewer
        self.viewer = TileViewer()
        layout.addWidget(self.viewer)

        # View controls
        controls = QHBoxLayout()

        # Overlay alpha
        controls.addWidget(QLabel("Overlay:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(
            lambda v: self.viewer.set_overlay_alpha(v / 100.0)
        )
        controls.addWidget(self.alpha_slider)

        # Show mask toggle
        self.show_mask_btn = QPushButton("Hide Mask")
        self.show_mask_btn.setCheckable(True)
        self.show_mask_btn.toggled.connect(self._toggle_mask)
        controls.addWidget(self.show_mask_btn)

        layout.addLayout(controls)

        # Navigation
        nav = QHBoxLayout()

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_tile)
        nav.addWidget(self.prev_btn)

        self.tile_label = QLabel("Tile 0/0")
        self.tile_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.tile_label)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_tile)
        nav.addWidget(self.next_btn)

        layout.addLayout(nav)

        return panel

    def _create_control_panel(self):
        """Create the control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Review actions
        actions_group = QGroupBox("Review Actions")
        actions_layout = QVBoxLayout()
        actions_group.setLayout(actions_layout)

        self.accept_btn = QPushButton("✓ Accept (A)")
        self.accept_btn.clicked.connect(self.accept_tile)
        self.accept_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        actions_layout.addWidget(self.accept_btn)

        self.reject_btn = QPushButton("✗ Reject (R)")
        self.reject_btn.clicked.connect(self.reject_tile)
        self.reject_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        actions_layout.addWidget(self.reject_btn)

        self.skip_btn = QPushButton("⏭ Skip (S)")
        self.skip_btn.clicked.connect(self.skip_tile)
        actions_layout.addWidget(self.skip_btn)

        layout.addWidget(actions_group)

        # Editing tools
        edit_group = QGroupBox("Editing Tools")
        edit_layout = QVBoxLayout()
        edit_group.setLayout(edit_layout)

        self.edit_toggle_btn = QPushButton("Enable Editing")
        self.edit_toggle_btn.setCheckable(True)
        self.edit_toggle_btn.toggled.connect(self._toggle_editing)
        edit_layout.addWidget(self.edit_toggle_btn)

        # Draw/Erase mode
        mode_layout = QHBoxLayout()
        self.draw_btn = QPushButton("Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setChecked(True)
        self.draw_btn.clicked.connect(lambda: self._set_draw_mode(True))
        mode_layout.addWidget(self.draw_btn)

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(lambda: self._set_draw_mode(False))
        mode_layout.addWidget(self.erase_btn)
        edit_layout.addLayout(mode_layout)

        # Brush size
        edit_layout.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self._set_brush_size)
        edit_layout.addWidget(self.brush_slider)
        self.brush_label = QLabel("10 px")
        edit_layout.addWidget(self.brush_label)

        # Save edits
        self.save_edit_btn = QPushButton("Save Edits")
        self.save_edit_btn.clicked.connect(self.save_edits)
        self.save_edit_btn.setEnabled(False)
        edit_layout.addWidget(self.save_edit_btn)

        layout.addWidget(edit_group)

        # Notes
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout()
        notes_group.setLayout(notes_layout)

        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("Add notes about this tile...")
        self.notes_text.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_text)

        layout.addWidget(notes_group)

        # Statistics
        stats_group = QGroupBox("Progress")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.progress_bar = QProgressBar()
        stats_layout.addWidget(self.progress_bar)

        self.stats_label = QLabel("Reviewed: 0/0")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        return panel

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("A"), self, self.accept_tile)
        QShortcut(QKeySequence("R"), self, self.reject_tile)
        QShortcut(QKeySequence("S"), self, self.skip_tile)
        QShortcut(QKeySequence("Left"), self, self.previous_tile)
        QShortcut(QKeySequence("Right"), self, self.next_tile)
        QShortcut(QKeySequence("E"), self, lambda: self.edit_toggle_btn.toggle())

    def load_current_tile(self):
        """Load the current tile and mask."""
        review = self.session.get_current_review()
        if review is None:
            return

        # Load tile
        self.viewer.load_tile(review.tile_path)

        # Load mask if exists
        if review.mask_path:
            self.viewer.load_mask(review.mask_path)
        else:
            # Create empty mask
            if self.viewer.tile_image is not None:
                h, w = self.viewer.tile_image.shape[:2]
                self.viewer.mask_image = np.zeros((h, w), dtype=np.uint8)

        self.viewer.update_display()

        # Update tile label
        tile_ids = self.session.get_tile_ids()
        current_idx = self.session.current_index
        self.tile_label.setText(f"Tile {current_idx + 1}/{len(tile_ids)}")
        self.setWindowTitle(f"Annotation Reviewer - {review.tile_id}")

        # Load notes
        self.notes_text.setText(review.notes)

    def accept_tile(self):
        """Accept current tile."""
        notes = self.notes_text.toPlainText()
        self.session.accept_current(notes)
        self.update_statistics()
        self.load_current_tile()

    def reject_tile(self):
        """Reject current tile."""
        notes = self.notes_text.toPlainText()
        self.session.reject_current(notes)
        self.update_statistics()
        self.load_current_tile()

    def skip_tile(self):
        """Skip current tile."""
        self.session.skip_current()
        self.update_statistics()
        self.load_current_tile()

    def previous_tile(self):
        """Go to previous tile."""
        self.session.previous_tile()
        self.load_current_tile()

    def next_tile(self):
        """Go to next tile."""
        self.session.next_tile()
        self.load_current_tile()

    def _toggle_mask(self, checked):
        """Toggle mask visibility."""
        self.viewer.set_show_mask(not checked)
        self.show_mask_btn.setText("Show Mask" if checked else "Hide Mask")

    def _toggle_editing(self, checked):
        """Toggle editing mode."""
        self.viewer.editing_enabled = checked
        self.save_edit_btn.setEnabled(checked)
        self.draw_btn.setEnabled(checked)
        self.erase_btn.setEnabled(checked)
        self.brush_slider.setEnabled(checked)

    def _set_draw_mode(self, is_draw):
        """Set draw/erase mode."""
        self.viewer.draw_mode = is_draw
        self.draw_btn.setChecked(is_draw)
        self.erase_btn.setChecked(not is_draw)

    def _set_brush_size(self, size):
        """Set brush size."""
        self.viewer.brush_size = size
        self.brush_label.setText(f"{size} px")

    def save_edits(self):
        """Save edited mask."""
        review = self.session.get_current_review()
        if review is None:
            return

        mask = self.viewer.get_mask()
        if mask is not None:
            self.session.save_edited_mask(review.tile_id, mask)
            QMessageBox.information(self, "Saved", "Mask edits saved successfully!")

    def update_statistics(self):
        """Update statistics display."""
        stats = self.session.get_statistics()
        counts = stats["counts"]
        progress = stats["progress_percent"]

        self.progress_bar.setValue(int(progress))
        self.stats_label.setText(
            f"Reviewed: {counts['accepted'] + counts['rejected']}/{counts['total']}\n"
            f"Accepted: {counts['accepted']}\n"
            f"Rejected: {counts['rejected']}\n"
            f"Edited: {counts['edited']}"
        )

    def export_results(self):
        """Export review results."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            str(self.session.session_dir / "review_results.json"),
            "JSON Files (*.json)"
        )

        if filename:
            self.session.export_results(Path(filename))
            QMessageBox.information(self, "Exported", f"Results exported to {filename}")


def launch_reviewer(session: ReviewSession):
    """
    Launch the reviewer UI.

    Args:
        session: ReviewSession to review
    """
    import os

    # Check if we're in WSL and provide helpful error message
    is_wsl = 'WSL' in os.environ.get('WSL_DISTRO_NAME', '') or \
             'microsoft' in os.uname().release.lower()

    # Check if DISPLAY is set
    if is_wsl and not os.environ.get('DISPLAY'):
        print("\n" + "=" * 70)
        print("ERROR: No display server detected (WSL environment)")
        print("=" * 70)
        print("\nYou're running in WSL, but no X server is configured.")
        print("\nTo use the GUI reviewer in WSL, you need to:")
        print("\n1. Install an X server on Windows:")
        print("   - VcXsrv: https://sourceforge.net/projects/vcxsrv/")
        print("   - X410: https://x410.dev/")
        print("   - Or use WSLg (Windows 11 with WSL2)")
        print("\n2. Set the DISPLAY variable in your ~/.bashrc:")
        print("   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0")
        print("   Or for WSLg: export DISPLAY=:0")
        print("\n3. Restart your terminal and try again")
        print("\nAlternatively, you can review annotations programmatically:")
        print("   from river_segmentation.annotation import ReviewSession")
        print("   session = ReviewSession(...)")
        print("   # Use session.accept_current(), session.reject_current(), etc.")
        print("=" * 70)
        sys.exit(1)

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Failed to initialize Qt application")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nThis usually means:")
        if is_wsl:
            print("- Your X server isn't running")
            print("- The DISPLAY variable is set incorrectly")
            print(f"  Current DISPLAY: {os.environ.get('DISPLAY', 'not set')}")
            print("\nTry running: export DISPLAY=:0")
        else:
            print("- Qt platform plugin couldn't be loaded")
            print("- Missing system dependencies")
            print("\nTry: pip install --force-reinstall PyQt5")
        print("=" * 70)
        sys.exit(1)

    # Set application style
    app.setStyle("Fusion")

    # Dark theme
    palette = app.palette()
    from PyQt5.QtGui import QPalette
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = ReviewerWindow(session)
    window.show()

    sys.exit(app.exec_())
