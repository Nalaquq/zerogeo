"""
Web-based annotation reviewer using Flask.

Provides a modern, browser-based UI for reviewing annotations
that works across all platforms including WSL.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
from PIL import Image
from flask import Flask, render_template, jsonify, request, send_file, session as flask_session

from .review_session import ReviewSession


class ReviewerWebApp:
    """Flask web application for annotation review."""

    def __init__(self, review_session: ReviewSession, host='127.0.0.1', port=5000):
        """
        Initialize Flask app.

        Args:
            review_session: ReviewSession instance
            host: Host address to bind to
            port: Port number
        """
        self.session = review_session
        self.host = host
        self.port = port

        # Create Flask app
        self.app = Flask(__name__,
                         template_folder=str(Path(__file__).parent / 'templates'),
                         static_folder=str(Path(__file__).parent / 'static'))
        self.app.secret_key = os.urandom(24)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route('/')
        def index():
            """Render main page."""
            return render_template('reviewer.html')

        @self.app.route('/api/status')
        def get_status():
            """Get current review status and statistics."""
            stats = self.session.get_statistics()
            review = self.session.get_current_review()

            return jsonify({
                'success': True,
                'current_tile': review.tile_id if review else None,
                'current_index': self.session.current_index,
                'statistics': stats
            })

        @self.app.route('/api/tile/current')
        def get_current_tile():
            """Get current tile data."""
            review = self.session.get_current_review()
            if not review:
                return jsonify({'success': False, 'error': 'No tile available'}), 404

            return jsonify({
                'success': True,
                'tile_id': review.tile_id,
                'index': self.session.current_index,
                'total': len(self.session.get_tile_ids()),
                'status': review.status.value,
                'notes': review.notes,
                'has_mask': review.mask_path is not None
            })

        @self.app.route('/api/image/tile')
        def get_tile_image():
            """Serve current tile image as PNG."""
            review = self.session.get_current_review()
            if not review:
                return jsonify({'error': 'No tile available'}), 404

            try:
                # Load and convert image
                img_array = self._load_tile_image(review.tile_path)

                # Convert to PIL Image and return as PNG
                img = Image.fromarray(img_array)
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)

                return send_file(img_io, mimetype='image/png')
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/image/mask')
        def get_mask_image():
            """Serve current mask as PNG overlay."""
            review = self.session.get_current_review()
            if not review or not review.mask_path:
                # Return transparent image if no mask
                img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')

            try:
                # Load mask
                mask_array = self._load_mask_image(review.mask_path)

                # Create colored overlay (cyan with transparency)
                overlay = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
                overlay[mask_array > 0] = [0, 255, 255, 128]  # Cyan with 50% opacity

                # Convert to PIL Image
                img = Image.fromarray(overlay, 'RGBA')
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)

                return send_file(img_io, mimetype='image/png')
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/mask/classes')
        def get_mask_classes():
            """Get class information for current mask."""
            review = self.session.get_current_review()
            if not review or not review.mask_path:
                return jsonify({
                    'success': True,
                    'classes': [],
                    'has_classes': False
                })

            try:
                mask_data = self._load_mask_data(review.mask_path)

                classes = []
                for i, (label, score) in enumerate(zip(mask_data['labels'], mask_data['scores'])):
                    mask_area = int(np.sum(mask_data['masks'][i]))
                    # Handle both scalar and array scores
                    if np.isscalar(score):
                        score_val = float(score)
                    elif isinstance(score, np.ndarray):
                        # Take mean if array (e.g., box confidence scores)
                        score_val = float(np.mean(score))
                    else:
                        score_val = 1.0
                    classes.append({
                        'id': i,
                        'label': label,
                        'score': score_val,
                        'area': mask_area,
                        'color': self._get_class_color(label)
                    })

                return jsonify({
                    'success': True,
                    'classes': classes,
                    'has_classes': len(classes) > 0
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/image/mask_multiclass')
        def get_mask_multiclass():
            """Serve current mask with multi-class colors."""
            review = self.session.get_current_review()

            # Get enabled classes from query params (comma-separated)
            enabled_classes = request.args.get('enabled', '')
            enabled_ids = set()
            if enabled_classes:
                try:
                    enabled_ids = set(int(x) for x in enabled_classes.split(','))
                except:
                    pass

            if not review or not review.mask_path:
                # Return transparent image if no mask
                img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')

            try:
                mask_data = self._load_mask_data(review.mask_path)

                # Create colored overlay
                h, w = mask_data['masks'][0].shape if len(mask_data['masks']) > 0 else (512, 512)
                overlay = np.zeros((h, w, 4), dtype=np.uint8)

                # Overlay each class with its color
                for i, (mask, label) in enumerate(zip(mask_data['masks'], mask_data['labels'])):
                    # Skip if not enabled
                    if enabled_ids and i not in enabled_ids:
                        continue

                    color = self._get_class_color(label)
                    overlay[mask > 0] = color

                # Convert to PIL Image
                img = Image.fromarray(overlay, 'RGBA')
                img_io = io.BytesIO()
                img.save(img_io, 'PNG')
                img_io.seek(0)

                return send_file(img_io, mimetype='image/png')
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/mask/delete', methods=['POST'])
        def delete_mask_instance():
            """Delete a specific mask instance."""
            data = request.get_json() or {}
            mask_id = data.get('mask_id')

            if mask_id is None:
                return jsonify({'success': False, 'error': 'mask_id required'}), 400

            review = self.session.get_current_review()
            if not review or not review.mask_path:
                return jsonify({'success': False, 'error': 'No mask available'}), 404

            try:
                mask_data = self._load_mask_data(review.mask_path)

                if mask_id < 0 or mask_id >= len(mask_data['masks']):
                    return jsonify({'success': False, 'error': 'Invalid mask_id'}), 400

                # Remove the mask
                masks = np.delete(mask_data['masks'], mask_id, axis=0)
                labels = [l for i, l in enumerate(mask_data['labels']) if i != mask_id]
                boxes = np.delete(mask_data['boxes'], mask_id, axis=0) if len(mask_data['boxes']) > 0 else np.array([])
                scores = np.delete(mask_data['scores'], mask_id)

                # Save updated masks
                self._save_mask_data(review.tile_id, masks, labels, boxes, scores)

                return jsonify({
                    'success': True,
                    'message': f'Deleted mask {mask_id}'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/mask/reassign', methods=['POST'])
        def reassign_mask_class():
            """Reassign a mask to a different class."""
            data = request.get_json() or {}
            mask_id = data.get('mask_id')
            new_label = data.get('new_label')

            if mask_id is None or not new_label:
                return jsonify({'success': False, 'error': 'mask_id and new_label required'}), 400

            review = self.session.get_current_review()
            if not review or not review.mask_path:
                return jsonify({'success': False, 'error': 'No mask available'}), 404

            try:
                mask_data = self._load_mask_data(review.mask_path)

                if mask_id < 0 or mask_id >= len(mask_data['masks']):
                    return jsonify({'success': False, 'error': 'Invalid mask_id'}), 400

                # Update label
                labels = list(mask_data['labels'])
                labels[mask_id] = new_label

                # Save updated masks
                self._save_mask_data(
                    review.tile_id,
                    mask_data['masks'],
                    labels,
                    mask_data['boxes'],
                    mask_data['scores']
                )

                return jsonify({
                    'success': True,
                    'message': f'Reassigned mask {mask_id} to {new_label}'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/mask/data')
        def get_mask_data():
            """Get raw mask data for editing."""
            review = self.session.get_current_review()
            if not review or not review.mask_path:
                return jsonify({
                    'success': False,
                    'error': 'No mask available'
                }), 404

            try:
                mask_data = self._load_mask_data(review.mask_path)

                # Convert masks to list format for JSON
                masks_list = []
                for mask in mask_data['masks']:
                    masks_list.append(mask.flatten().tolist())

                return jsonify({
                    'success': True,
                    'mask_data': {
                        'masks': masks_list,
                        'labels': mask_data['labels'],
                        'width': mask_data['masks'][0].shape[1] if len(mask_data['masks']) > 0 else 512,
                        'height': mask_data['masks'][0].shape[0] if len(mask_data['masks']) > 0 else 512,
                    }
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/mask/save_edits', methods=['POST'])
        def save_mask_edits():
            """Save edited mask data."""
            data = request.get_json() or {}
            mask_data_json = data.get('mask_data')

            if not mask_data_json:
                return jsonify({'success': False, 'error': 'mask_data required'}), 400

            review = self.session.get_current_review()
            if not review:
                return jsonify({'success': False, 'error': 'No tile available'}), 404

            try:
                # Reconstruct mask arrays from JSON
                masks_list = []
                width = mask_data_json.get('width', 512)
                height = mask_data_json.get('height', 512)

                for mask_flat in mask_data_json['masks']:
                    mask = np.array(mask_flat, dtype=np.uint8).reshape(height, width)
                    masks_list.append(mask)

                masks = np.array(masks_list)
                labels = mask_data_json['labels']

                # Load original data to get boxes and scores
                if review.mask_path:
                    original_data = self._load_mask_data(review.mask_path)
                    boxes = original_data.get('boxes', np.array([]))
                    scores = original_data.get('scores', np.ones(len(masks)))
                else:
                    boxes = np.array([])
                    scores = np.ones(len(masks))

                # Save updated masks
                self._save_mask_data(
                    review.tile_id,
                    masks,
                    labels,
                    boxes,
                    scores
                )

                return jsonify({
                    'success': True,
                    'message': 'Edits saved successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/tile/accept', methods=['POST'])
        def accept_tile():
            """Accept current tile."""
            data = request.get_json() or {}
            notes = data.get('notes', '')

            # Get current tile before accepting (moves to next)
            current_review = self.session.get_current_review()
            if current_review:
                tile_id = current_review.tile_id
                # Convert any edited .npz to .tif before accepting
                self._convert_npz_to_tif(tile_id)

            self.session.accept_current(notes)

            return jsonify({
                'success': True,
                'message': 'Tile accepted'
            })

        @self.app.route('/api/tile/reject', methods=['POST'])
        def reject_tile():
            """Reject current tile."""
            data = request.get_json() or {}
            notes = data.get('notes', '')

            # Get current tile before rejecting (moves to next)
            current_review = self.session.get_current_review()
            if current_review:
                tile_id = current_review.tile_id
                # Convert any edited .npz to .tif before rejecting
                self._convert_npz_to_tif(tile_id)

            self.session.reject_current(notes)

            return jsonify({
                'success': True,
                'message': 'Tile rejected'
            })

        @self.app.route('/api/tile/skip', methods=['POST'])
        def skip_tile():
            """Skip current tile."""
            self.session.skip_current()

            return jsonify({
                'success': True,
                'message': 'Tile skipped'
            })

        @self.app.route('/api/tile/navigate', methods=['POST'])
        def navigate_tile():
            """Navigate to next/previous tile."""
            data = request.get_json() or {}
            direction = data.get('direction', 'next')

            if direction == 'next':
                self.session.next_tile()
            elif direction == 'previous':
                self.session.previous_tile()
            else:
                return jsonify({'success': False, 'error': 'Invalid direction'}), 400

            return jsonify({
                'success': True,
                'message': f'Moved to {direction} tile'
            })

        @self.app.route('/api/tile/update_notes', methods=['POST'])
        def update_notes():
            """Update notes for current tile."""
            data = request.get_json() or {}
            notes = data.get('notes', '')

            review = self.session.get_current_review()
            if review:
                review.notes = notes
                self.session.save_state()

                return jsonify({
                    'success': True,
                    'message': 'Notes updated'
                })

            return jsonify({'success': False, 'error': 'No tile available'}), 404

        @self.app.route('/api/export', methods=['POST'])
        def export_results():
            """Export reviewed annotations."""
            data = request.get_json() or {}
            output_path = data.get('output_path')

            if not output_path:
                return jsonify({'success': False, 'error': 'output_path required'}), 400

            try:
                self.session.export_results(Path(output_path))
                return jsonify({
                    'success': True,
                    'message': f'Results exported to {output_path}'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

    def _load_tile_image(self, tile_path: str) -> np.ndarray:
        """
        Load tile image as RGB array.

        Args:
            tile_path: Path to tile file

        Returns:
            RGB image array (H, W, 3) uint8
        """
        with rasterio.open(tile_path) as src:
            # Read data
            data = src.read()  # (bands, height, width)

            # Handle different band counts
            if data.shape[0] >= 3:
                # Use first 3 bands as RGB
                r, g, b = data[0], data[1], data[2]
            else:
                # Single band - use as grayscale
                r = g = b = data[0]

            # Normalize to 0-255
            def normalize(arr):
                arr = arr.astype(np.float32)
                arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
                if arr_max > arr_min:
                    arr = ((arr - arr_min) / (arr_max - arr_min) * 255)
                else:
                    arr = np.zeros_like(arr)
                return arr.astype(np.uint8)

            r = normalize(r)
            g = normalize(g)
            b = normalize(b)

            # Stack to RGB
            return np.stack([r, g, b], axis=-1)

    def _load_mask_image(self, mask_path: str) -> np.ndarray:
        """
        Load mask image.

        Args:
            mask_path: Path to mask file

        Returns:
            Binary mask array (H, W) uint8
        """
        if mask_path.endswith('.npz'):
            data = np.load(mask_path)
            mask = data['mask']
            if mask.ndim == 3:
                mask = mask[0]
        else:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

        return (mask > 0).astype(np.uint8)

    def _get_class_color(self, label: str) -> list:
        """
        Get RGBA color for a class label.

        Args:
            label: Class label (e.g., 'river', 'stream')

        Returns:
            RGBA color as [R, G, B, A]
        """
        # Predefined colors for common classes (RGBA format with alpha=180)
        color_map = {
            'river': [30, 144, 255, 180],       # Dodger blue
            'stream': [0, 191, 255, 180],       # Deep sky blue
            'creek': [135, 206, 250, 180],      # Light sky blue
            'waterway': [0, 206, 209, 180],     # Dark turquoise
            'lake': [64, 224, 208, 180],        # Turquoise
            'pond': [127, 255, 212, 180],       # Aquamarine
            'coastline': [70, 130, 180, 180],   # Steel blue
            'shore': [100, 149, 237, 180],      # Cornflower blue
            'merged': [0, 255, 255, 128],       # Cyan (default)
            'unknown': [128, 128, 128, 128],    # Gray
        }

        # Return predefined color or generate from hash
        if label.lower() in color_map:
            return color_map[label.lower()]

        # Generate color from label hash
        import hashlib
        h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        return [r, g, b, 180]

    def _save_mask_data(self, tile_id: str, masks: np.ndarray, labels: list,
                        boxes: np.ndarray, scores: np.ndarray):
        """
        Save updated mask data.

        Args:
            tile_id: Tile identifier
            masks: Mask array (N, H, W)
            labels: List of labels
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
        """
        review = self.session.get_review(tile_id)
        if not review:
            raise ValueError(f"Unknown tile: {tile_id}")

        # Save to reviewed_masks directory
        output_path = self.session.reviewed_masks_dir / f"{tile_id}_mask.npz"

        np.savez_compressed(
            output_path,
            masks=masks,
            labels=np.array(labels, dtype=object),
            boxes=boxes,
            scores=scores
        )

        # Update review
        review.mask_path = str(output_path)
        review.edited = True
        self.session.save_session()

    def _convert_npz_to_tif(self, tile_id: str):
        """
        Convert edited .npz mask to .tif format.

        Args:
            tile_id: Tile identifier
        """
        review = self.session.get_review(tile_id)
        if not review:
            return

        # Check if there's an edited .npz file
        npz_path = self.session.reviewed_masks_dir / f"{tile_id}_mask.npz"
        if not npz_path.exists():
            return

        # Load the .npz data
        data = np.load(npz_path, allow_pickle=True)
        masks = data['masks']
        labels = data['labels']

        # Merge all masks into single binary mask for .tif output
        if len(masks) > 0:
            merged_mask = np.any(masks, axis=0).astype(np.uint8)
        else:
            merged_mask = np.zeros((512, 512), dtype=np.uint8)

        # Save as GeoTIFF
        tif_path = self.session.reviewed_masks_dir / f"{tile_id}_mask.tif"

        try:
            # Try to preserve georeferencing from original tile
            with rasterio.open(review.tile_path) as src:
                profile = src.profile.copy()
                profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'compress': 'lzw',
                })

                with rasterio.open(tif_path, 'w', **profile) as dst:
                    dst.write(merged_mask, 1)
        except Exception as e:
            # Fallback: save without georeferencing
            print(f"Warning: Could not preserve georeferencing for {tile_id}: {e}")

            height, width = merged_mask.shape
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
            }

            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(merged_mask, 1)

    def _load_mask_data(self, mask_path: str) -> dict:
        """
        Load multi-class mask data with labels.

        Args:
            mask_path: Path to mask file

        Returns:
            Dictionary with masks, labels, boxes, and scores
        """
        if mask_path.endswith('.npz'):
            # Load from npz with full metadata
            data = np.load(mask_path, allow_pickle=True)

            # Extract masks
            masks = data.get('mask', data.get('masks', None))
            if masks is None:
                return {'masks': np.array([]), 'labels': [], 'boxes': np.array([]), 'scores': np.array([])}

            # Ensure masks is 3D (N, H, W)
            # Squeeze out any singleton dimensions (e.g., (N, 1, H, W) -> (N, H, W))
            while masks.ndim > 3:
                masks = np.squeeze(masks, axis=1)

            if masks.ndim == 2:
                # Single merged mask - no class info preserved
                masks = masks[np.newaxis, :, :]
                labels = ['unknown']
                boxes = np.array([])
                scores = np.array([1.0])
            elif masks.ndim == 3:
                # Multi-class masks
                labels = data.get('labels', [f'class_{i}' for i in range(len(masks))])
                # Handle numpy arrays of objects (pickle format)
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                boxes = data.get('boxes', np.array([]))
                scores = data.get('scores', np.ones(len(masks)))
            else:
                return {'masks': np.array([]), 'labels': [], 'boxes': np.array([]), 'scores': np.array([])}

            return {
                'masks': (masks > 0).astype(np.uint8),
                'labels': labels,
                'boxes': boxes,
                'scores': scores
            }
        else:
            # Load from GeoTIFF (single merged mask, no class info)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            return {
                'masks': ((mask > 0).astype(np.uint8))[np.newaxis, :, :],
                'labels': ['merged'],
                'boxes': np.array([]),
                'scores': np.array([1.0])
            }

    def run(self, debug=False):
        """
        Start the Flask development server.

        Args:
            debug: Enable debug mode
        """
        # Find available port if the default is in use
        import socket
        original_port = self.port

        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.host, port))
                    return False
                except OSError:
                    return True

        # Try to find an available port
        if is_port_in_use(self.port):
            print(f"\nPort {self.port} is already in use, finding available port...")
            for port in range(5000, 5100):
                if not is_port_in_use(port):
                    self.port = port
                    break

            if is_port_in_use(self.port):
                print(f"Error: Could not find an available port in range 5000-5100")
                return

        print("\n" + "=" * 70)
        print("ANNOTATION REVIEWER - WEB INTERFACE")
        print("=" * 70)
        print(f"\nServer starting at: http://{self.host}:{self.port}")
        if self.port != original_port:
            print(f"(Note: Using port {self.port} instead of {original_port})")
        print("\nKeyboard shortcuts:")
        print("  A - Accept tile")
        print("  R - Reject tile")
        print("  S - Skip tile")
        print("  Left/Right arrows - Navigate tiles")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70)
        print()

        self.app.run(host=self.host, port=self.port, debug=debug)


def launch_web_reviewer(session: ReviewSession, host='127.0.0.1', port=5000, debug=False):
    """
    Launch web-based reviewer.

    Args:
        session: ReviewSession to review
        host: Host address (default: 127.0.0.1)
        port: Port number (default: 5000)
        debug: Enable debug mode
    """
    app = ReviewerWebApp(session, host=host, port=port)
    app.run(debug=debug)
