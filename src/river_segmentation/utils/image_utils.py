"""
Image loading and processing utilities.

Supports various image formats including:
- GeoTIFF (georeferenced)
- Standard formats (JPEG, PNG, BMP, etc.)
- Multi-band images (RGB, RGBA, multispectral)
- Various data types (uint8, uint16, float32, etc.)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageInfo:
    """Container for image metadata."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        bands: int,
        dtype: str,
        is_georeferenced: bool = False,
        crs: Optional[str] = None,
        transform: Optional[Any] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Initialize image info.

        Args:
            path: Path to image file
            width: Image width in pixels
            height: Image height in pixels
            bands: Number of bands/channels
            dtype: Data type (e.g., 'uint8', 'float32')
            is_georeferenced: Whether image has geospatial metadata
            crs: Coordinate reference system (if georeferenced)
            transform: Affine transform (if georeferenced)
            bounds: Geographic bounds (if georeferenced)
        """
        self.path = Path(path)
        self.width = width
        self.height = height
        self.bands = bands
        self.dtype = dtype
        self.is_georeferenced = is_georeferenced
        self.crs = crs
        self.transform = transform
        self.bounds = bounds

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return image shape as (height, width, bands)."""
        return (self.height, self.width, self.bands)

    @property
    def size_mb(self) -> float:
        """Estimate image size in MB."""
        bytes_per_pixel = {
            'uint8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float32': 4,
            'float64': 8,
        }
        byte_size = self.width * self.height * self.bands * bytes_per_pixel.get(self.dtype, 4)
        return byte_size / (1024 * 1024)

    def __repr__(self) -> str:
        geo_str = f", CRS={self.crs}" if self.is_georeferenced else ""
        return (
            f"ImageInfo({self.width}x{self.height}x{self.bands}, "
            f"dtype={self.dtype}, size={self.size_mb:.2f}MB{geo_str})"
        )


def get_image_info(image_path: Path, logger: Optional[logging.Logger] = None) -> ImageInfo:
    """
    Get metadata about an image without loading the full array.

    Args:
        image_path: Path to image file
        logger: Optional logger for diagnostic messages

    Returns:
        ImageInfo object with image metadata

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try rasterio first (best for GeoTIFF)
    if HAS_RASTERIO and image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with rasterio.open(image_path) as src:
                info = ImageInfo(
                    path=image_path,
                    width=src.width,
                    height=src.height,
                    bands=src.count,
                    dtype=str(src.dtypes[0]),
                    is_georeferenced=src.crs is not None,
                    crs=str(src.crs) if src.crs else None,
                    transform=src.transform,
                    bounds=src.bounds,
                )

                if logger:
                    logger.info(f"Image info: {info}")
                    if info.is_georeferenced:
                        logger.debug(f"  CRS: {info.crs}")
                        logger.debug(f"  Bounds: {info.bounds}")

                return info
        except Exception as e:
            if logger:
                logger.warning(f"Rasterio failed to read {image_path}: {e}, trying other methods")

    # Try PIL for standard formats
    if HAS_PIL:
        try:
            with PILImage.open(image_path) as img:
                mode_to_bands = {
                    'L': 1,  # Grayscale
                    'RGB': 3,  # RGB
                    'RGBA': 4,  # RGBA
                    'CMYK': 4,  # CMYK
                }

                bands = mode_to_bands.get(img.mode, 3)

                info = ImageInfo(
                    path=image_path,
                    width=img.width,
                    height=img.height,
                    bands=bands,
                    dtype='uint8',  # PIL images are typically uint8
                    is_georeferenced=False,
                )

                if logger:
                    logger.info(f"Image info: {info}")

                return info
        except Exception as e:
            if logger:
                logger.warning(f"PIL failed to read {image_path}: {e}")

    # Try OpenCV as last resort
    if HAS_OPENCV:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                height, width = img.shape[:2]
                bands = img.shape[2] if img.ndim == 3 else 1

                info = ImageInfo(
                    path=image_path,
                    width=width,
                    height=height,
                    bands=bands,
                    dtype=str(img.dtype),
                    is_georeferenced=False,
                )

                if logger:
                    logger.info(f"Image info: {info}")

                return info
        except Exception as e:
            if logger:
                logger.warning(f"OpenCV failed to read {image_path}: {e}")

    raise ValueError(
        f"Could not read image: {image_path}\n"
        f"Supported formats: GeoTIFF (.tif/.tiff), JPEG (.jpg/.jpeg), PNG (.png), BMP (.bmp)\n"
        f"Available libraries: rasterio={HAS_RASTERIO}, PIL={HAS_PIL}, opencv={HAS_OPENCV}"
    )


def load_image_rgb(
    image_path: Path,
    logger: Optional[logging.Logger] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Load image in RGB format for model inference.

    Supports:
    - GeoTIFF (float32, uint16, uint8, multi-band)
    - JPEG, PNG, BMP, and other standard formats
    - Automatic band selection and conversion
    - Data type normalization to uint8

    Args:
        image_path: Path to image file
        logger: Optional logger for diagnostic messages
        normalize: Whether to normalize data to 0-255 range

    Returns:
        Image array (H, W, 3) in RGB format (uint8)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or converted
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if logger:
        logger.debug(f"Loading image: {image_path}")

    # Try rasterio first for TIFF/GeoTIFF (better format support)
    if HAS_RASTERIO and image_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with rasterio.open(image_path) as src:
                # Read all bands
                data = src.read()  # Shape: (bands, height, width)

                if logger:
                    logger.debug(f"  Rasterio read: shape={data.shape}, dtype={data.dtype}")
                    logger.debug(f"  Bands: {src.count}, CRS: {src.crs}")

                # Transpose to (height, width, bands)
                if data.ndim == 3:
                    data = np.transpose(data, (1, 2, 0))
                else:
                    # Single band
                    data = data[0]  # Remove band dimension
                    if logger:
                        logger.debug(f"  Single band image, converting to RGB")
                    data = np.stack([data, data, data], axis=-1)

                # Handle different data types
                if data.dtype in [np.float32, np.float64]:
                    if normalize:
                        # Normalize float data to 0-255 range
                        data_min = np.nanmin(data)
                        data_max = np.nanmax(data)

                        if logger:
                            logger.debug(f"  Float data range: [{data_min:.3f}, {data_max:.3f}]")

                        if data_max > data_min:
                            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                        else:
                            data = np.zeros_like(data, dtype=np.uint8)
                    else:
                        data = data.astype(np.uint8)

                elif data.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    if logger:
                        logger.debug(f"  Converting uint16 to uint8")
                    data = (data / 256).astype(np.uint8)

                elif data.dtype != np.uint8:
                    if logger:
                        logger.debug(f"  Converting {data.dtype} to uint8")
                    data = data.astype(np.uint8)

                # Handle different band counts
                if data.shape[-1] == 1:
                    # Grayscale -> RGB
                    data = np.repeat(data, 3, axis=-1)
                elif data.shape[-1] == 2:
                    # 2-band -> RGB (duplicate first band)
                    data = np.dstack([data[..., 0], data[..., 0], data[..., 1]])
                elif data.shape[-1] == 3:
                    # Already RGB
                    pass
                elif data.shape[-1] == 4:
                    # RGBA or multispectral -> RGB
                    if logger:
                        logger.debug(f"  4-band image, using first 3 bands as RGB")
                    data = data[..., :3]
                elif data.shape[-1] > 4:
                    # Multispectral -> RGB (use first 3 bands)
                    if logger:
                        logger.warning(
                            f"  Multispectral image with {data.shape[-1]} bands, "
                            f"using bands 1-3 as RGB"
                        )
                    data = data[..., :3]
                else:
                    raise ValueError(f"Unexpected number of channels: {data.shape[-1]}")

                if logger:
                    logger.debug(f"  Final shape: {data.shape}, dtype: {data.dtype}")

                return data

        except Exception as e:
            if logger:
                logger.warning(f"Rasterio failed ({e}), trying other methods...")

    # Fallback: OpenCV for standard formats (JPEG, PNG, etc.)
    if HAS_OPENCV:
        try:
            # Load with OpenCV (BGR)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV failed to load image")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if logger:
                logger.debug(f"  Loaded with OpenCV: shape={image.shape}, dtype={image.dtype}")

            return image

        except Exception as e:
            if logger:
                logger.warning(f"OpenCV failed ({e}), trying PIL...")

    # Last resort: PIL
    if HAS_PIL:
        try:
            with PILImage.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    if logger:
                        logger.debug(f"  Converting {img.mode} to RGB")
                    img = img.convert('RGB')

                image = np.array(img)

                if logger:
                    logger.debug(f"  Loaded with PIL: shape={image.shape}, dtype={image.dtype}")

                return image

        except Exception as e:
            if logger:
                logger.error(f"PIL failed ({e})")

    raise ValueError(
        f"Could not load image as RGB: {image_path}\n"
        f"Supported formats: GeoTIFF, JPEG, PNG, BMP\n"
        f"Available libraries: rasterio={HAS_RASTERIO}, opencv={HAS_OPENCV}, PIL={HAS_PIL}\n"
        f"Install missing libraries with: pip install rasterio opencv-python pillow"
    )


def validate_image_for_annotation(
    image_path: Path,
    max_size_mb: float = 1000.0,
    logger: Optional[logging.Logger] = None
) -> ImageInfo:
    """
    Validate that an image is suitable for annotation.

    Args:
        image_path: Path to image file
        max_size_mb: Maximum allowed image size in MB
        logger: Optional logger for diagnostic messages

    Returns:
        ImageInfo object if validation passes

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image is invalid or too large
    """
    # Get image info
    info = get_image_info(image_path, logger=logger)

    # Check size
    if info.size_mb > max_size_mb:
        raise ValueError(
            f"Image too large: {info.size_mb:.2f} MB (max: {max_size_mb} MB)\n"
            f"Consider tiling the image first or increasing max_size_mb parameter"
        )

    # Check dimensions
    if info.width < 32 or info.height < 32:
        raise ValueError(
            f"Image too small: {info.width}x{info.height} (minimum: 32x32)\n"
            f"The annotation models require at least 32x32 pixel images"
        )

    # Check for extremely large dimensions
    if info.width > 100000 or info.height > 100000:
        raise ValueError(
            f"Image extremely large: {info.width}x{info.height}\n"
            f"Consider tiling the image first for better performance"
        )

    if logger:
        logger.info(f"Image validation passed: {info}")

    return info
