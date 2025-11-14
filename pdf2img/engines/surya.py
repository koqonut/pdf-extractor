"""
Surya OCR Engine

Modern transformer-based OCR with excellent accuracy.
One of the best 2024 OCR engines.
"""

import time
from pathlib import Path

from loguru import logger
from PIL import Image

from .base import OCREngine, OCRResult, register_engine


@register_engine
class SuryaEngine(OCREngine):
    """Surya - Modern transformer OCR

    Performance:
        - Accuracy: 90-93%
        - Speed: 2-4s per page
        - RAM: ~2GB
        - Model size: ~400MB download

    Example:
        engine = SuryaEngine()
        result = engine.extract('flyer.png')
    """

    name = "surya"
    model_size = "400MB"
    ram_usage_gb = 2.0
    description = "Surya - Modern transformer OCR, 90-93% accuracy"

    def _load_model(self):
        """Lazy load model"""
        if self._model is not None:
            return

        logger.info("Loading Surya OCR...")

        from surya.ocr import run_ocr
        from surya.model.detection import segformer
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor

        # Load models
        self._det_model = segformer.load_model()
        self._rec_model = load_rec_model()
        self._processor = load_processor()

        # Store run_ocr function
        self._run_ocr = run_ocr

        logger.success("Surya OCR loaded successfully")

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

            # Run OCR
            logger.info(f"Running Surya OCR extraction on {image_path.name}...")

            # Surya expects a list of images
            predictions = self._run_ocr(
                [image],
                [["en"]],  # Language
                self._det_model,
                self._rec_model,
                self._processor
            )

            # Extract text from predictions
            text_lines = []
            for pred in predictions[0].text_lines:
                text_lines.append(pred.text)

            text = "\n".join(text_lines)

            processing_time = time.time() - start_time

            logger.success(
                f"Surya OCR extraction completed in {processing_time:.2f}s "
                f"({len(text)} chars)"
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
