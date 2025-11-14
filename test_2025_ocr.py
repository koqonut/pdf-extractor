#!/usr/bin/env python3
"""
Test script for 2025 Modern OCR & Vision Models
Optimized for M2 MacBook Air with 4-bit quantization support

Usage:
    # Test GOT-OCR 2.0 (lightweight, fast)
    python test_2025_ocr.py --image flyer.png --engine got

    # Test MiniCPM-V 2.6 (best OCR performance)
    python test_2025_ocr.py --image flyer.png --engine minicpm

    # Test Phi-3.5 Vision (small, efficient)
    python test_2025_ocr.py --image flyer.png --engine phi3

    # Test PaliGemma 2
    python test_2025_ocr.py --image flyer.png --engine paligemma --model-size 3b

    # Batch test multiple images
    python test_2025_ocr.py --batch data/images/*.png --engine minicpm

    # Compare all engines
    python test_2025_ocr.py --image flyer.png --compare-all
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from PIL import Image

app = typer.Typer()


@dataclass
class ExtractionResult:
    """Result from OCR extraction"""

    engine: str
    text: str
    processing_time: float
    model_size: str
    ram_usage_gb: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


# ============================================================================
# GOT-OCR 2.0 - Lightweight, Excellent OCR (580M params)
# ============================================================================


def test_got_ocr(image_path: Path, use_structured: bool = False) -> ExtractionResult:
    """
    Test GOT-OCR 2.0 for text extraction

    Args:
        image_path: Path to image file
        use_structured: If True, extract in structured format (markdown/JSON)

    Returns:
        ExtractionResult with extracted text
    """
    logger.info(f"Testing GOT-OCR 2.0 on {image_path}")
    start_time = time.time()

    try:
        from transformers import AutoModel, AutoTokenizer

        # Load model and tokenizer
        logger.info("Loading GOT-OCR 2.0 model (580M params)...")
        model_name = "stepfun-ai/GOT-OCR2_0"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
        )

        # Load image
        image = Image.open(image_path)

        # Perform OCR
        logger.info("Running OCR extraction...")
        if use_structured:
            # Extract in structured format (markdown/JSON)
            text = model.chat(tokenizer, image, ocr_type="format")
        else:
            # Extract plain text
            text = model.chat(tokenizer, image, ocr_type="ocr")

        processing_time = time.time() - start_time

        logger.success(f"GOT-OCR extraction completed in {processing_time:.2f}s")
        logger.info(f"Extracted {len(text)} characters")

        return ExtractionResult(
            engine="GOT-OCR 2.0",
            text=text,
            processing_time=processing_time,
            model_size="580M",
            ram_usage_gb=2.0,  # Approximate
        )

    except Exception as e:
        logger.error(f"GOT-OCR failed: {e}")
        return ExtractionResult(
            engine="GOT-OCR 2.0",
            text="",
            processing_time=time.time() - start_time,
            model_size="580M",
            success=False,
            error=str(e),
        )


# ============================================================================
# MiniCPM-V 2.6 - Top OCRBench Performer (8B params)
# ============================================================================


def test_minicpm(
    image_path: Path, use_4bit: bool = True, extract_structured: bool = False
) -> ExtractionResult:
    """
    Test MiniCPM-V 2.6 for text extraction

    Args:
        image_path: Path to image file
        use_4bit: Use 4-bit quantization for M2 Air (recommended)
        extract_structured: Extract as structured JSON

    Returns:
        ExtractionResult with extracted text
    """
    logger.info(f"Testing MiniCPM-V 2.6 on {image_path}")
    start_time = time.time()

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        # Load model with optional 4-bit quantization
        logger.info(
            f"Loading MiniCPM-V 2.6 (8B params, 4-bit={use_4bit})..."
        )
        model_name = "openbmb/MiniCPM-V-2_6"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if use_4bit:
            # 4-bit quantization for M2 Air (uses ~4-5GB RAM instead of ~16GB)
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
            ram_usage = 4.5
        else:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            ram_usage = 16.0

        # Load image
        image = Image.open(image_path)

        # Prepare prompt
        if extract_structured:
            prompt = """Extract all items and prices from this retail flyer.
Return as JSON with this structure:
{
  "items": [
    {"name": "Product Name", "price": "$X.XX", "unit": "each/lb/etc"}
  ]
}"""
        else:
            prompt = "Extract all text from this image, especially product names and prices."

        # Perform OCR
        logger.info("Running MiniCPM-V extraction...")

        # MiniCPM-V 2.6 chat interface
        msgs = [{"role": "user", "content": [image, prompt]}]

        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
        )

        processing_time = time.time() - start_time

        logger.success(f"MiniCPM-V extraction completed in {processing_time:.2f}s")
        logger.info(f"Extracted {len(response)} characters")

        return ExtractionResult(
            engine="MiniCPM-V 2.6",
            text=response,
            processing_time=processing_time,
            model_size="8B (4-bit)" if use_4bit else "8B",
            ram_usage_gb=ram_usage,
        )

    except Exception as e:
        logger.error(f"MiniCPM-V failed: {e}")
        return ExtractionResult(
            engine="MiniCPM-V 2.6",
            text="",
            processing_time=time.time() - start_time,
            model_size="8B",
            success=False,
            error=str(e),
        )


# ============================================================================
# Phi-3.5 Vision - Microsoft's Small VLM (4.2B params)
# ============================================================================


def test_phi3_vision(
    image_path: Path, use_4bit: bool = True, extract_prices: bool = True
) -> ExtractionResult:
    """
    Test Phi-3.5 Vision for text extraction

    Args:
        image_path: Path to image file
        use_4bit: Use 4-bit quantization for M2 Air (recommended)
        extract_prices: Focus on extracting prices and product names

    Returns:
        ExtractionResult with extracted text
    """
    logger.info(f"Testing Phi-3.5 Vision on {image_path}")
    start_time = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        # Load model with optional 4-bit quantization
        logger.info(f"Loading Phi-3.5 Vision (4.2B params, 4-bit={use_4bit})...")
        model_name = "microsoft/Phi-3.5-vision-instruct"

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if use_4bit:
            # 4-bit quantization for M2 Air
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
            ram_usage = 3.5
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            ram_usage = 8.5

        # Load image
        image = Image.open(image_path)

        # Prepare prompt
        if extract_prices:
            prompt = "<|image_1|>\nExtract all product names and their prices from this retail flyer. List each item with its price."
        else:
            prompt = "<|image_1|>\nExtract all text visible in this image."

        # Prepare inputs
        inputs = processor(prompt, [image], return_tensors="pt").to(model.device)

        # Generate response
        logger.info("Running Phi-3.5 Vision extraction...")

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
            )

        # Remove input tokens from output
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

        # Decode response
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        processing_time = time.time() - start_time

        logger.success(f"Phi-3.5 Vision extraction completed in {processing_time:.2f}s")
        logger.info(f"Extracted {len(response)} characters")

        return ExtractionResult(
            engine="Phi-3.5 Vision",
            text=response,
            processing_time=processing_time,
            model_size="4.2B (4-bit)" if use_4bit else "4.2B",
            ram_usage_gb=ram_usage,
        )

    except Exception as e:
        logger.error(f"Phi-3.5 Vision failed: {e}")
        return ExtractionResult(
            engine="Phi-3.5 Vision",
            text="",
            processing_time=time.time() - start_time,
            model_size="4.2B",
            success=False,
            error=str(e),
        )


# ============================================================================
# PaliGemma 2 - Google's VLM (3B/10B/28B variants)
# ============================================================================


def test_paligemma(
    image_path: Path, model_size: str = "3b", resolution: int = 448
) -> ExtractionResult:
    """
    Test PaliGemma 2 for text extraction

    Args:
        image_path: Path to image file
        model_size: Model size (3b, 10b, or 28b)
        resolution: Input resolution (224, 448, or 896)

    Returns:
        ExtractionResult with extracted text
    """
    logger.info(f"Testing PaliGemma 2 ({model_size}) on {image_path}")
    start_time = time.time()

    try:
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        import torch

        # Validate inputs
        if model_size not in ["3b", "10b", "28b"]:
            raise ValueError(f"Invalid model size: {model_size}. Use 3b, 10b, or 28b")
        if resolution not in [224, 448, 896]:
            raise ValueError(f"Invalid resolution: {resolution}. Use 224, 448, or 896")

        # Load model
        logger.info(f"Loading PaliGemma 2 ({model_size}, {resolution}px)...")
        model_name = f"google/paligemma-2-{model_size}-pt-{resolution}"

        processor = AutoProcessor.from_pretrained(model_name)

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Estimate RAM usage
        ram_map = {"3b": 2.5, "10b": 6.5, "28b": 14.0}
        ram_usage = ram_map[model_size]

        # Load image
        image = Image.open(image_path)

        # Prepare prompt
        prompt = "extract text from image"

        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            model.device
        )

        # Generate response
        logger.info("Running PaliGemma 2 extraction...")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=500)

        # Decode response
        response = processor.decode(output[0], skip_special_tokens=True)

        processing_time = time.time() - start_time

        logger.success(f"PaliGemma 2 extraction completed in {processing_time:.2f}s")
        logger.info(f"Extracted {len(response)} characters")

        return ExtractionResult(
            engine=f"PaliGemma 2 ({model_size})",
            text=response,
            processing_time=processing_time,
            model_size=model_size.upper(),
            ram_usage_gb=ram_usage,
        )

    except Exception as e:
        logger.error(f"PaliGemma 2 failed: {e}")
        return ExtractionResult(
            engine=f"PaliGemma 2 ({model_size})",
            text="",
            processing_time=time.time() - start_time,
            model_size=model_size.upper(),
            success=False,
            error=str(e),
        )


# ============================================================================
# Helper Functions
# ============================================================================


def print_result(result: ExtractionResult, save_output: bool = False):
    """Pretty print extraction result"""
    print("\n" + "=" * 80)
    print(f"Engine: {result.engine}")
    print(f"Model Size: {result.model_size}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    if result.ram_usage_gb:
        print(f"RAM Usage: ~{result.ram_usage_gb:.1f}GB")
    print("=" * 80)

    if result.success:
        print(f"\nExtracted Text ({len(result.text)} chars):")
        print("-" * 80)
        print(result.text)
        print("-" * 80)

        if save_output:
            output_file = Path(f"output_{result.engine.replace(' ', '_').lower()}.txt")
            output_file.write_text(result.text)
            logger.success(f"Saved output to {output_file}")
    else:
        print(f"\n❌ FAILED: {result.error}")


def compare_all_engines(image_path: Path) -> List[ExtractionResult]:
    """Run all 2025 engines and compare results"""
    logger.info(f"Running comparison on {image_path} with all 2025 engines...")

    results = []

    # Test GOT-OCR 2.0 (fastest, lightest)
    try:
        results.append(test_got_ocr(image_path))
    except Exception as e:
        logger.warning(f"GOT-OCR test failed: {e}")

    # Test Phi-3.5 Vision (small, efficient)
    try:
        results.append(test_phi3_vision(image_path, use_4bit=True))
    except Exception as e:
        logger.warning(f"Phi-3.5 Vision test failed: {e}")

    # Test MiniCPM-V 2.6 (best performance)
    try:
        results.append(test_minicpm(image_path, use_4bit=True))
    except Exception as e:
        logger.warning(f"MiniCPM-V test failed: {e}")

    # Test PaliGemma 2 3B (Google's offering)
    try:
        results.append(test_paligemma(image_path, model_size="3b"))
    except Exception as e:
        logger.warning(f"PaliGemma 2 test failed: {e}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY - 2025 OCR Engines")
    print("=" * 100)
    print(
        f"{'Engine':<25} {'Model Size':<12} {'Time (s)':<10} {'RAM (GB)':<10} {'Status':<10}"
    )
    print("-" * 100)

    for r in results:
        status = "✅ OK" if r.success else "❌ FAIL"
        ram = f"~{r.ram_usage_gb:.1f}" if r.ram_usage_gb else "N/A"
        print(
            f"{r.engine:<25} {r.model_size:<12} {r.processing_time:<10.2f} {ram:<10} {status:<10}"
        )

    print("=" * 100)

    return results


# ============================================================================
# CLI Commands
# ============================================================================


@app.command()
def test(
    image: Path = typer.Option(..., "--image", "-i", help="Path to image file"),
    engine: str = typer.Option(
        "minicpm",
        "--engine",
        "-e",
        help="Engine: got, minicpm, phi3, paligemma",
    ),
    model_size: str = typer.Option(
        "3b", "--model-size", "-s", help="Model size for PaliGemma (3b/10b/28b)"
    ),
    use_4bit: bool = typer.Option(
        True, "--4bit/--no-4bit", help="Use 4-bit quantization (M2 recommended)"
    ),
    save_output: bool = typer.Option(
        False, "--save/--no-save", help="Save output to file"
    ),
    compare_all: bool = typer.Option(
        False, "--compare-all", help="Test all engines and compare"
    ),
):
    """Test 2025 modern OCR engines on an image"""

    if not image.exists():
        logger.error(f"Image not found: {image}")
        raise typer.Exit(1)

    if compare_all:
        results = compare_all_engines(image)
        for result in results:
            print_result(result, save_output)
        return

    # Test single engine
    engine = engine.lower()

    if engine == "got":
        result = test_got_ocr(image)
    elif engine == "minicpm":
        result = test_minicpm(image, use_4bit=use_4bit)
    elif engine == "phi3":
        result = test_phi3_vision(image, use_4bit=use_4bit)
    elif engine == "paligemma":
        result = test_paligemma(image, model_size=model_size)
    else:
        logger.error(
            f"Unknown engine: {engine}. Use: got, minicpm, phi3, or paligemma"
        )
        raise typer.Exit(1)

    print_result(result, save_output)


@app.command()
def batch(
    images: List[Path] = typer.Argument(..., help="Image files to process"),
    engine: str = typer.Option(
        "minicpm", "--engine", "-e", help="Engine to use"
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
):
    """Batch process multiple images"""

    output_dir.mkdir(exist_ok=True)
    logger.info(f"Processing {len(images)} images with {engine}...")

    results = []

    for img in images:
        if not img.exists():
            logger.warning(f"Skipping missing file: {img}")
            continue

        logger.info(f"\nProcessing {img.name}...")

        # Test based on engine
        if engine == "got":
            result = test_got_ocr(img)
        elif engine == "minicpm":
            result = test_minicpm(img, use_4bit=True)
        elif engine == "phi3":
            result = test_phi3_vision(img, use_4bit=True)
        elif engine == "paligemma":
            result = test_paligemma(img, model_size="3b")
        else:
            logger.error(f"Unknown engine: {engine}")
            continue

        results.append(result)

        # Save output
        if result.success:
            output_file = output_dir / f"{img.stem}_extracted.txt"
            output_file.write_text(result.text)
            logger.success(f"Saved to {output_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images: {len(images)}")
    logger.info(f"Successful: {sum(1 for r in results if r.success)}")
    logger.info(f"Failed: {sum(1 for r in results if not r.success)}")
    logger.info(
        f"Average time: {sum(r.processing_time for r in results) / len(results):.2f}s"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
