"""
OCR Engine Plugin System

This module provides a plugin-based architecture for OCR engines,
making it easy to add and test new OCR libraries.

Quick Start:
    # List available engines
    from pdf2img.engines import list_engines
    print(list_engines())  # ['minicpm', 'got-ocr', 'surya', ...]

    # Get an engine
    from pdf2img.engines import get_engine
    engine = get_engine('minicpm')

    # Extract text
    result = engine.extract('flyer.png')
    print(result.text)

Adding a New Engine:
    1. Create a new file in pdf2img/engines/ (e.g., myengine.py)
    2. Subclass OCREngine and set attributes
    3. Implement extract() method
    4. Add @register_engine decorator
    5. Import it in this file

    Example:
        # pdf2img/engines/myengine.py
        from pdf2img.engines import OCREngine, OCRResult, register_engine

        @register_engine
        class MyEngine(OCREngine):
            name = "my-engine"
            model_size = "1B"
            ram_usage_gb = 2.0

            def extract(self, image_path, **kwargs):
                # Your code here
                return OCRResult(...)

        # pdf2img/engines/__init__.py
        from .myengine import MyEngine  # Auto-registers!
"""

from .base import (
    OCREngine,
    OCRResult,
    get_engine,
    get_engine_info,
    list_all_engine_info,
    list_engines,
    register_engine,
)

# Auto-import all engines (triggers registration via @register_engine decorator)
# Just add a new import line when you create a new engine!
try:
    from .minicpm import MiniCPMEngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .got_ocr import GOTOCREngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .phi3 import Phi3VisionEngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .surya import SuryaEngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .qwen2vl import Qwen2VLEngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .paddleocr import PaddleOCREngine  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from .claude_api import ClaudeAPIEngine  # noqa: F401
except ImportError:
    pass  # Optional dependency


__all__ = [
    "OCREngine",
    "OCRResult",
    "register_engine",
    "get_engine",
    "list_engines",
    "get_engine_info",
    "list_all_engine_info",
]
