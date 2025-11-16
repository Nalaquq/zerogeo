"""
Logging utilities for CIVIC annotation pipeline.

Provides centralized logging configuration with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with file and optional console output.

    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to save log files (if None, only console logging)
        log_file: Log filename (if None, auto-generated from timestamp)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(message)s'
    )

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name.replace('.', '_')}_{timestamp}.log"

        log_path = log_dir / log_file

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_pipeline_logger(
    log_dir: Path,
    pipeline_name: str = "civic_pipeline",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for the main CIVIC pipeline.

    Args:
        log_dir: Directory to save log files
        pipeline_name: Name of the pipeline run
        level: Logging level

    Returns:
        Configured logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{pipeline_name}_{timestamp}.log"

    logger = setup_logger(
        name="civic",
        log_dir=log_dir,
        log_file=log_file,
        level=level,
        console=True
    )

    # Log header
    logger.info("=" * 80)
    logger.info(f"CIVIC ANNOTATION PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with traceback.

    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context about where the exception occurred
    """
    import traceback

    if context:
        logger.error(f"Exception in {context}:")
    else:
        logger.error("Exception occurred:")

    logger.error(f"  Type: {type(exception).__name__}")
    logger.error(f"  Message: {str(exception)}")
    logger.error("  Traceback:")

    # Log each line of traceback
    for line in traceback.format_tb(exception.__traceback__):
        for tb_line in line.strip().split('\n'):
            logger.error(f"    {tb_line}")


def log_system_info(logger: logging.Logger):
    """
    Log system information for debugging.

    Args:
        logger: Logger instance
    """
    import platform
    import psutil

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # Check for CUDA
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("  PyTorch not available")


def log_config(logger: logging.Logger, config: dict):
    """
    Log configuration settings.

    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration:")

    def log_dict(d: dict, indent: int = 2):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{' ' * indent}{key}:")
                log_dict(value, indent + 2)
            else:
                logger.info(f"{' ' * indent}{key}: {value}")

    log_dict(config)
