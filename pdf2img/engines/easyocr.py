"""
EasyOCR Engine

Modern deep learning OCR with 80+ language support.
MPS support for Apple Silicon since Sept 2023.
"""

import time
from pathlib import Path

from loguru import logger

from .base import OCREngine, OCRResult, register_engine


@register_engine
class EasyOCREngine(OCREngine):
    """EasyOCR - Modern deep learning OCR with multi-language support

    Modern OCR engine with 80+ languages and GPU acceleration.
    MPS support for Apple Silicon added in Sept 2023.

    Performance:
        - Accuracy: 85-95%
        - Speed: 2-5s per page
        - RAM: ~1-2GB
        - Platform: All (CUDA, MPS, CPU)

    Example:
        engine = EasyOCREngine(languages=['en'], use_gpu=True)
        result = engine.extract('flyer.png')
    """

    name = "easyocr"
    model_size = "~200MB"
    ram_usage_gb = 1.5
    description = "EasyOCR - Multi-language (80+), GPU support (CUDA/MPS)"

    def __init__(self, languages: list = None, use_gpu: bool = True, **config):
        """Initialize EasyOCR engine

        Args:
            languages: List of language codes (default: ['en'])
            use_gpu: Use GPU if available (CUDA or MPS)
            **config: Additional configuration options
        """
        super().__init__(**config)
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self._reader = None

    def _load_model(self):
        """Lazy load EasyOCR reader"""
        if self._reader is not None:
            return

        try:
            import easyocr

            logger.info(f"Loading EasyOCR ({', '.join(self.languages)}, GPU={self.use_gpu})...")

            # EasyOCR automatically detects CUDA/MPS/CPU
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)

            # Detect which device is being used
            import torch

            if self.use_gpu:
                if torch.cuda.is_available():
                    device = "CUDA"
                elif torch.backends.mps.is_available():
                    device = "MPS"
                else:
                    device = "CPU (GPU requested but not available)"
            else:
                device = "CPU"

            logger.success(f"EasyOCR loaded successfully on {device}")

        except ImportError:
            raise ImportError(
                "easyocr not installed. Install with: pip install easyocr\n"
                "Note: Will download language models on first use (~100-200MB per language)."
            )

    def extract(self, image_path: Path, detail: int = 0, **kwargs) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            detail: 0 = simple text, 1 = bounding boxes + text
            **kwargs: Additional EasyOCR options

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()

        try:
            # Load model (lazy loading)
            self._load_model()

            # Run OCR
            logger.info(f"Running EasyOCR on {image_path.name} ({', '.join(self.languages)})...")

            # readtext returns list of (bbox, text, confidence)
            results = self._reader.readtext(str(image_path), detail=detail, **kwargs)

            # Extract text (results format depends on detail parameter)
            if detail == 0:
                # Simple mode: results is list of strings
                text = "\n".join(results)
            else:
                # Detailed mode: results is list of (bbox, text, confidence)
                text = "\n".join([item[1] for item in results])

            processing_time = time.time() - start_time

            logger.success(f"EasyOCR completed in {processing_time:.2f}s ({len(text)} chars)")

            return OCRResult(
                engine=self.name,
                text=text,
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                metadata={
                    "languages": self.languages,
                    "gpu": self.use_gpu,
                    "detail": detail,
                    "num_detections": len(results),
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"EasyOCR failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
