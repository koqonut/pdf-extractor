"""
GOT-OCR 2.0 Engine

Lightweight and fast OCR with excellent accuracy.
Best for speed-critical applications.
"""

import time
from pathlib import Path

from loguru import logger
from PIL import Image

from .base import OCREngine, OCRResult, register_engine


@register_engine
class GOTOCREngine(OCREngine):
    """GOT-OCR 2.0 - Fastest OCR engine with excellent accuracy

    Performance:
        - Accuracy: 90-93%
        - Speed: 2-3s per page (fastest!)
        - RAM: ~2GB
        - Model size: 580M parameters

    Example:
        engine = GOTOCREngine()
        result = engine.extract('flyer.png', structured=True)
    """

    name = "got-ocr"
    model_size = "580M"
    ram_usage_gb = 2.0
    description = "GOT-OCR 2.0 - Lightweight, fast, handles tables"

    def _load_model(self):
        """Lazy load model"""
        if self._model is not None:
            return

        logger.info("Loading GOT-OCR 2.0...")

        from transformers import AutoModel, AutoTokenizer

        model_name = "stepfun-ai/GOT-OCR2_0"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
        )

        logger.success("GOT-OCR 2.0 loaded successfully")

    def extract(
        self,
        image_path: Path,
        structured: bool = False,
        **kwargs
    ) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            structured: If True, extract in structured format (markdown/JSON)
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
            logger.info(f"Running GOT-OCR extraction on {image_path.name}...")

            if structured:
                # Extract with structure (tables, formatting)
                text = self._model.chat(self._tokenizer, image, ocr_type="format")
            else:
                # Extract plain text
                text = self._model.chat(self._tokenizer, image, ocr_type="ocr")

            processing_time = time.time() - start_time

            logger.success(
                f"GOT-OCR extraction completed in {processing_time:.2f}s "
                f"({len(text)} chars)"
            )

            return OCRResult(
                engine=self.name,
                text=text,
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                metadata={"structured": structured},
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"GOT-OCR extraction failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
