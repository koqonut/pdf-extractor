"""
Surya OCR Engine

Modern transformer-based OCR with excellent accuracy (90+ languages).
Uses 2025 predictor-based API for optimal performance.
"""

import time
from pathlib import Path

from loguru import logger
from PIL import Image

from .base import OCREngine, OCRResult, register_engine


@register_engine
class SuryaEngine(OCREngine):
    """Surya - Modern transformer OCR (90+ languages)

    Uses 2025 predictor-based API for optimal performance.

    Performance:
        - Accuracy: 90-93%
        - Speed: 2-4s per page
        - RAM: ~2GB
        - Model size: ~400MB download
        - Languages: 90+ languages supported

    Example:
        engine = SuryaEngine()
        result = engine.extract('flyer.png')
    """

    name = "surya"
    model_size = "400MB"
    ram_usage_gb = 2.0
    description = "Surya - Modern OCR (90+ languages), works on CPU/MPS/CUDA"

    def _load_model(self):
        """Lazy load model (2025 API)"""
        if self._model is not None:
            return

        logger.info("Loading Surya OCR...")

        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        # Load predictors (new 2025 API)
        self._foundation_predictor = FoundationPredictor()
        self._recognition_predictor = RecognitionPredictor(self._foundation_predictor)
        self._detection_predictor = DetectionPredictor()

        # Mark as loaded
        self._model = True

        logger.success("Surya OCR loaded successfully (2025 API)")

    def extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            **kwargs: Additional options

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()

        try:
            # Load model (lazy loading)
            self._load_model()

            # Load image
            image = Image.open(image_path)

            # Run OCR (2025 API - uses predictor classes)
            logger.info(f"Running Surya OCR extraction on {image_path.name}...")

            # Surya expects a list of images
            predictions = self._recognition_predictor(
                [image], det_predictor=self._detection_predictor
            )

            # Extract text from predictions
            text_lines = []
            for pred in predictions[0].text_lines:
                text_lines.append(pred.text)

            text = "\n".join(text_lines)

            processing_time = time.time() - start_time

            logger.success(
                f"Surya OCR extraction completed in {processing_time:.2f}s " f"({len(text)} chars)"
            )

            return OCRResult(
                engine=self.name,
                text=text,
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                metadata={"lines": len(text_lines)},
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Surya OCR extraction failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
