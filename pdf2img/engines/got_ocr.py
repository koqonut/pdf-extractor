"""
GOT-OCR 2.0 Engine

Lightweight and fast OCR with excellent accuracy.
Best for speed-critical applications.
"""

import time
from pathlib import Path

from loguru import logger

from .base import OCREngine, OCRResult, register_engine


@register_engine
class GOTOCREngine(OCREngine):
    """GOT-OCR 2.0 - Fastest OCR engine with excellent accuracy

    REQUIRES CUDA (NVIDIA GPU) - Does NOT work on macOS or CPU-only systems.
    For macOS, use MiniCPM-V or Phi-3.5 Vision instead.

    Performance:
        - Accuracy: 90-93%
        - Speed: 2-3s per page (fastest!)
        - RAM: ~2GB
        - Model size: 580M parameters
        - Platform: CUDA only (no CPU/MPS support)

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

        import torch
        from transformers import AutoModel, AutoTokenizer

        # GOT-OCR ONLY works with CUDA - it has hardcoded CUDA calls in custom model code
        # See: https://huggingface.co/stepfun-ai/GOT-OCR2_0/discussions/4
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GOT-OCR requires CUDA and does not support CPU or MPS (Apple Silicon). "
                "Please use a different engine like 'minicpm' or 'phi3' on macOS. "
                "See: https://huggingface.co/stepfun-ai/GOT-OCR2_0/discussions/4"
            )

        device = "cuda"
        logger.info("Loading GOT-OCR 2.0 on CUDA...")

        model_name = "stepfun-ai/GOT-OCR2_0"

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        logger.success(f"GOT-OCR 2.0 loaded successfully on {device}")

    def extract(self, image_path: Path, structured: bool = False, **kwargs) -> OCRResult:
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

            # Run OCR
            logger.info(f"Running GOT-OCR extraction on {image_path.name}...")

            # GOT-OCR expects image path as string, not PIL Image object
            image_path_str = str(image_path)

            if structured:
                # Extract with structure (tables, formatting)
                text = self._model.chat(self._tokenizer, image_path_str, ocr_type="format")
            else:
                # Extract plain text
                text = self._model.chat(self._tokenizer, image_path_str, ocr_type="ocr")

            processing_time = time.time() - start_time

            logger.success(
                f"GOT-OCR extraction completed in {processing_time:.2f}s " f"({len(text)} chars)"
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
