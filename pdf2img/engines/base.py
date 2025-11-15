"""
Base classes for OCR engine plugin system

This module provides the foundation for the OCR engine plugin pattern,
making it trivial to add new OCR engines without code duplication.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


@dataclass
class OCRResult:
    """Standardized result from OCR extraction

    Attributes:
        engine: Name of the engine used
        text: Extracted text
        processing_time: Time taken in seconds
        model_size: Size of the model (e.g., "8B", "580M")
        ram_usage_gb: Estimated RAM usage in GB
        success: Whether extraction succeeded
        error: Error message if failed
        metadata: Engine-specific metadata
    """

    engine: str
    text: str
    processing_time: float
    model_size: str
    ram_usage_gb: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class OCREngine(ABC):
    """Base class for all OCR engines

    To add a new engine:
    1. Subclass OCREngine
    2. Set class attributes (name, model_size, etc.)
    3. Implement extract() method
    4. Register with @register_engine decorator

    Example:
        @register_engine
        class MyEngine(OCREngine):
            name = "my-engine"
            model_size = "1B"
            ram_usage_gb = 2.0

            def extract(self, image_path: Path, **kwargs) -> OCRResult:
                # Your implementation
                pass
    """

    # Override these in subclasses
    name: str = "override_me"
    model_size: str = "unknown"
    ram_usage_gb: float = 0.0
    description: str = ""

    def __init__(self, **config):
        """Initialize engine with optional configuration

        Args:
            **config: Engine-specific configuration options
        """
        self.config = config
        self._model = None  # Lazy loading

    @abstractmethod
    def extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            **kwargs: Engine-specific options

        Returns:
            OCRResult with extracted text and metadata
        """
        pass

    def _safe_extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Wrapper that catches exceptions and returns OCRResult

        This is used internally to ensure engines always return
        a valid OCRResult even if they fail.
        """
        start_time = time.time()

        try:
            return self.extract(image_path, **kwargs)
        except Exception as e:
            logger.error(f"{self.name} extraction failed: {e}")
            return OCRResult(
                engine=self.name,
                text="",
                processing_time=time.time() - start_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.name}' "
            f"size={self.model_size} "
            f"ram={self.ram_usage_gb}GB>"
        )


# Engine registry
_ENGINE_REGISTRY: Dict[str, type[OCREngine]] = {}


def register_engine(engine_class: type[OCREngine]) -> type[OCREngine]:
    """Decorator to register an engine in the global registry

    Usage:
        @register_engine
        class MyEngine(OCREngine):
            name = "my-engine"
            ...

    Args:
        engine_class: The engine class to register

    Returns:
        The same class (for decorator compatibility)
    """
    if not issubclass(engine_class, OCREngine):
        raise TypeError(f"{engine_class} must inherit from OCREngine")

    engine_name = engine_class.name

    if engine_name == "override_me":
        raise ValueError(f"{engine_class.__name__} must set a unique 'name' class attribute")

    if engine_name in _ENGINE_REGISTRY:
        logger.warning(
            f"Engine '{engine_name}' already registered, overwriting "
            f"with {engine_class.__name__}"
        )

    _ENGINE_REGISTRY[engine_name] = engine_class
    logger.debug(f"Registered OCR engine: {engine_name}")

    return engine_class


def get_engine(name: str, **config) -> OCREngine:
    """Get an engine instance by name

    Args:
        name: Engine name (e.g., 'minicpm', 'got-ocr')
        **config: Configuration to pass to engine constructor

    Returns:
        Initialized engine instance

    Raises:
        ValueError: If engine not found
    """
    if name not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine: '{name}'. Available engines: {available}")

    engine_class = _ENGINE_REGISTRY[name]
    return engine_class(**config)


def list_engines() -> list[str]:
    """List all registered engine names

    Returns:
        List of engine names
    """
    return sorted(_ENGINE_REGISTRY.keys())


def get_engine_info(name: str) -> dict:
    """Get information about an engine

    Args:
        name: Engine name

    Returns:
        Dictionary with engine metadata

    Raises:
        ValueError: If engine not found
    """
    if name not in _ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: '{name}'")

    engine_class = _ENGINE_REGISTRY[name]

    return {
        "name": engine_class.name,
        "model_size": engine_class.model_size,
        "ram_usage_gb": engine_class.ram_usage_gb,
        "description": engine_class.description,
        "class": engine_class.__name__,
    }


def list_all_engine_info() -> list[dict]:
    """Get information about all registered engines

    Returns:
        List of dictionaries with engine metadata
    """
    return [get_engine_info(name) for name in list_engines()]
