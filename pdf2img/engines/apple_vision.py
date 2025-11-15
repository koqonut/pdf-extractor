"""
Apple Vision Framework Engine

Uses Apple's native Vision Framework for OCR on macOS.
Best performance and accuracy on Apple Silicon (M1/M2/M3).
"""

import time
from pathlib import Path

from loguru import logger

from .base import OCREngine, OCRResult, register_engine


@register_engine
class AppleVisionEngine(OCREngine):
    """Apple Vision Framework - Native macOS OCR

    REQUIRES macOS - Uses Apple's Vision Framework via ocrmac wrapper.
    Best choice for Apple Silicon (M1/M2/M3) with excellent accuracy and speed.

    Performance:
        - Accuracy: 90-95%
        - Speed: 0.2-0.5s per page (fastest!)
        - RAM: ~500MB
        - Platform: macOS only

    Example:
        engine = AppleVisionEngine()
        result = engine.extract('flyer.png')
    """

    name = "apple-vision"
    model_size = "N/A"
    ram_usage_gb = 0.5
    description = "Apple Vision Framework - Native macOS OCR, fastest and most accurate"

    def __init__(self, **config):
        """Initialize Apple Vision engine

        Args:
            **config: Additional configuration options
        """
        super().__init__(**config)
        self._ocrmac = None

    def _check_platform(self):
        """Check if running on macOS"""
        import platform

        if platform.system() != "Darwin":
            raise RuntimeError(
                "Apple Vision Framework requires macOS. "
                "For other platforms, use Tesseract, Surya, or EasyOCR."
            )

    def _load_model(self):
        """Lazy load ocrmac library"""
        if self._ocrmac is not None:
            return

        self._check_platform()

        try:
            from ocrmac.ocrmac import OCR

            self._ocrmac = OCR
            logger.info("Apple Vision Framework ready (native macOS OCR)")
        except ImportError:
            raise ImportError(
                "ocrmac not installed. Install with: pip install ocrmac\n"
                "Note: Only works on macOS."
            )

    def extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            **kwargs: Additional options (ignored)

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()

        try:
            # Load model (lazy loading)
            self._load_model()

            # Run OCR
            logger.info(f"Running Apple Vision OCR on {image_path.name}...")

            # Create OCR object and recognize text
            # Returns list of tuples: (text, confidence, bbox)
            ocr = self._ocrmac(str(image_path), framework="vision")
            annotations = ocr.recognize()

            # Extract text from annotations
            if annotations:
                # Join all text pieces with newlines
                text = "\n".join([item[0] for item in annotations])
            else:
                text = ""

            processing_time = time.time() - start_time

            logger.success(
                f"Apple Vision OCR completed in {processing_time:.2f}s "
                f"({len(text)} chars, {len(annotations)} detections)"
            )

            return OCRResult(
                engine=self.name,
                text=text,
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                metadata={
                    "platform": "macOS",
                    "framework": "Apple Vision Framework",
                    "num_detections": len(annotations),
                    "avg_confidence": (
                        sum(item[1] for item in annotations) / len(annotations)
                        if annotations
                        else 0.0
                    ),
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Apple Vision OCR failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
