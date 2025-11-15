"""
Tesseract OCR Engine

Traditional OCR engine, industry standard, works on all platforms.
ARM-optimized for Apple Silicon since v5.0.
"""

import time
from pathlib import Path

from loguru import logger
from PIL import Image

from .base import OCREngine, OCRResult, register_engine


@register_engine
class TesseractEngine(OCREngine):
    """Tesseract OCR - Industry standard OCR engine

    Traditional OCR engine that works on all platforms.
    ARM-optimized for Apple Silicon in v5.0+.

    Performance:
        - Accuracy: 80-90%
        - Speed: 1-3s per page
        - RAM: ~200MB
        - Platform: All (Linux, macOS, Windows)

    Installation:
        - macOS: brew install tesseract && pip install pytesseract
        - Linux: apt-get install tesseract-ocr && pip install pytesseract
        - Windows: Download installer + pip install pytesseract

    Example:
        engine = TesseractEngine()
        result = engine.extract('flyer.png')
    """

    name = "tesseract"
    model_size = "N/A"
    ram_usage_gb = 0.2
    description = "Tesseract OCR - Traditional, reliable, works on all platforms"

    def __init__(self, lang: str = "eng", config: str = "", **kwargs):
        """Initialize Tesseract engine

        Args:
            lang: Language code (default: 'eng')
            config: Tesseract config string (default: '')
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        self.lang = lang
        self.tesseract_config = config
        self._pytesseract = None

    def _load_model(self):
        """Lazy load pytesseract library"""
        if self._pytesseract is not None:
            return

        try:
            import pytesseract

            self._pytesseract = pytesseract

            # Check if tesseract is installed
            try:
                version = self._pytesseract.get_tesseract_version()
                logger.info(f"Tesseract OCR v{version} ready")
            except Exception as e:
                raise RuntimeError(
                    "Tesseract not installed. Install with:\n"
                    "  macOS: brew install tesseract\n"
                    "  Linux: apt-get install tesseract-ocr\n"
                    "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
                ) from e

        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract\n"
                "Also install tesseract binary (see above)."
            )

    def extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            **kwargs: Additional options (override lang, config)

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()

        try:
            # Load model (lazy loading)
            self._load_model()

            # Load image
            image = Image.open(image_path)

            # Get parameters
            lang = kwargs.get("lang", self.lang)
            config = kwargs.get("config", self.tesseract_config)

            # Run OCR
            logger.info(f"Running Tesseract OCR on {image_path.name} (lang={lang})...")

            text = self._pytesseract.image_to_string(image, lang=lang, config=config)

            processing_time = time.time() - start_time

            logger.success(
                f"Tesseract OCR completed in {processing_time:.2f}s " f"({len(text)} chars)"
            )

            return OCRResult(
                engine=self.name,
                text=text,
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                metadata={
                    "lang": lang,
                    "config": config,
                    "version": str(self._pytesseract.get_tesseract_version()),
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Tesseract OCR failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
