"""
Phi-3.5 Vision Engine

Microsoft's small and efficient VLM, optimized for edge devices.
Perfect for M2 MacBook Air with limited RAM.
"""

import time
from pathlib import Path

from loguru import logger
from PIL import Image

from .base import OCREngine, OCRResult, register_engine


@register_engine
class Phi3VisionEngine(OCREngine):
    """Phi-3.5 Vision - Small, efficient VLM

    Performance:
        - Accuracy: 88-92%
        - Speed: 5-8s per page
        - RAM: 3-4GB (with 4-bit quantization)
        - Model size: 4.2B parameters
        - License: MIT (fully open)

    Example:
        engine = Phi3VisionEngine(use_4bit=True)
        result = engine.extract('flyer.png')
    """

    name = "phi3"
    model_size = "4.2B"
    ram_usage_gb = 3.5
    description = "Phi-3.5 Vision - Microsoft's small VLM, MIT license"

    def __init__(self, use_4bit: bool = True, **config):
        """Initialize Phi-3.5 Vision engine

        Args:
            use_4bit: Use 4-bit quantization (recommended for M2 Air)
            **config: Additional configuration options
        """
        super().__init__(**config)
        self.use_4bit = use_4bit
        self._processor = None

    def _load_model(self):
        """Lazy load model"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Determine device: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Disable 4-bit quantization on MPS (Apple Silicon doesn't support bitsandbytes)
        use_4bit = self.use_4bit
        if use_4bit and device == "mps":
            logger.warning(
                "4-bit quantization not supported on Apple Silicon (MPS). "
                "Falling back to 16-bit precision."
            )
            use_4bit = False

        logger.info(f"Loading Phi-3.5 Vision on {device.upper()} (4-bit={use_4bit})...")

        model_name = "microsoft/Phi-3.5-vision-instruct"

        # Phi-3.5 requires trust_remote_code=True
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, _attn_implementation="eager"
        )

        if use_4bit:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
                _attn_implementation="eager",  # Disable FlashAttention2 (not available on macOS)
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                _attn_implementation="eager",  # Disable FlashAttention2 (not available on macOS)
            ).to(device)

        logger.success(f"Phi-3.5 Vision loaded successfully on {device}")

    def extract(
        self, image_path: Path, extract_prices: bool = True, prompt: str = None, **kwargs
    ) -> OCRResult:
        """Extract text from image

        Args:
            image_path: Path to image file
            extract_prices: Focus on extracting prices and products
            prompt: Custom prompt (optional)
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

            # Prepare prompt
            if prompt is None:
                if extract_prices:
                    prompt = "<|image_1|>\nExtract all product names and their prices from this retail flyer. List each item with its price."
                else:
                    prompt = "<|image_1|>\nExtract all text visible in this image."

            # Run OCR
            logger.info(f"Running Phi-3.5 Vision extraction on {image_path.name}...")

            import torch

            inputs = self._processor(prompt, [image], return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                generate_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=False,
                )

            # Remove input tokens from output
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

            # Decode response
            response = self._processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            processing_time = time.time() - start_time

            logger.success(
                f"Phi-3.5 Vision extraction completed in {processing_time:.2f}s "
                f"({len(response)} chars)"
            )

            return OCRResult(
                engine=self.name,
                text=response,
                processing_time=processing_time,
                model_size=f"{self.model_size} ({'4-bit' if self.use_4bit else 'fp16'})",
                ram_usage_gb=self.ram_usage_gb if self.use_4bit else 8.5,
                metadata={
                    "4bit": self.use_4bit,
                    "prompt": prompt,
                    "extract_prices": extract_prices,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Phi-3.5 Vision extraction failed: {e}")

            return OCRResult(
                engine=self.name,
                text="",
                processing_time=processing_time,
                model_size=self.model_size,
                ram_usage_gb=self.ram_usage_gb,
                success=False,
                error=str(e),
            )
