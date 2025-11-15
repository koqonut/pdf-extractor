#!/usr/bin/env python3
"""
Local OCR Testing Suite - Compare Different OCR Engines

Tests multiple OCR approaches on M2 MacBook Air:
1. Tesseract (traditional)
2. EasyOCR (deep learning)
3. PaddleOCR (fast + accurate)
4. Apple Vision Framework (native macOS)

Usage:
    python test_local_ocr.py --image path/to/flyer.png
"""

import time
import re
import json
from pathlib import Path
from typing import List, Dict
import argparse
from dataclasses import dataclass, asdict


@dataclass
class OCRResult:
    engine: str
    processing_time: float
    raw_text: str
    items_found: List[Dict]
    prices_found: List[str]
    success: bool
    error: str = None


def extract_prices_from_text(text: str) -> List[str]:
    """Extract prices using regex patterns."""

    # Common price patterns
    patterns = [
        r"\$\s*\d+\.\d{2}",  # $5.99, $ 5.99
        r"\d+\.\d{2}\s*(?:ea|each|lb|kg)",  # 5.99 ea, 5.99 lb
        r"\d+\s*for\s*\$\s*\d+",  # 2 for $5
        r"\$\s*\d+",  # $5
        r"\d+¢",  # 99¢
    ]

    prices = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        prices.extend(matches)

    # Clean and deduplicate
    prices = list(set(prices))

    # Sort by value for consistency
    prices.sort()

    return prices


def pair_items_with_prices(text: str, prices: List[str]) -> List[Dict]:
    """Simple heuristic to pair product names with prices."""

    lines = text.split("\n")
    items = []

    for i, line in enumerate(lines):
        # Check if line contains a price
        for price in prices:
            if price in line:
                # Get context (previous and current line for product name)
                product_context = []
                if i > 0:
                    product_context.append(lines[i - 1].strip())
                product_context.append(line.replace(price, "").strip())

                product_name = " ".join(product_context).strip()

                # Clean price
                clean_price = re.sub(r"[^\d.]", "", price)

                if product_name and clean_price:
                    items.append(
                        {"name": product_name, "price": clean_price, "raw_text": line.strip()}
                    )

    return items


def test_tesseract(image_path: Path) -> OCRResult:
    """Test Tesseract OCR."""

    try:
        import pytesseract
        import cv2
        import numpy as np

        print("  📖 Running Tesseract OCR...")
        start = time.time()

        # Load and preprocess image
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocessing for better OCR
        # 1. Denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # 2. Thresholding
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 3. Deskewing (optional but helpful)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5:  # Only deskew if needed
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        # Run OCR
        text = pytesseract.image_to_string(thresh)

        end = time.time()

        # Extract prices and items
        prices = extract_prices_from_text(text)
        items = pair_items_with_prices(text, prices)

        return OCRResult(
            engine="Tesseract",
            processing_time=round(end - start, 2),
            raw_text=text,
            items_found=items,
            prices_found=prices,
            success=True,
        )

    except Exception as e:
        return OCRResult(
            engine="Tesseract",
            processing_time=0,
            raw_text="",
            items_found=[],
            prices_found=[],
            success=False,
            error=str(e),
        )


def test_easyocr(image_path: Path) -> OCRResult:
    """Test EasyOCR (deep learning based)."""

    try:
        import easyocr

        print("  📖 Running EasyOCR...")
        start = time.time()

        # Initialize reader (cached after first run)
        reader = easyocr.Reader(["en"], gpu=False)  # M2 uses CPU

        # Run OCR
        results = reader.readtext(str(image_path))

        # Combine text
        text = "\n".join([result[1] for result in results])

        end = time.time()

        # Extract prices and items
        prices = extract_prices_from_text(text)
        items = pair_items_with_prices(text, prices)

        return OCRResult(
            engine="EasyOCR",
            processing_time=round(end - start, 2),
            raw_text=text,
            items_found=items,
            prices_found=prices,
            success=True,
        )

    except Exception as e:
        return OCRResult(
            engine="EasyOCR",
            processing_time=0,
            raw_text="",
            items_found=[],
            prices_found=[],
            success=False,
            error=str(e),
        )


def test_paddleocr(image_path: Path) -> OCRResult:
    """Test PaddleOCR (fast and accurate)."""

    try:
        from paddleocr import PaddleOCR

        print("  📖 Running PaddleOCR...")
        start = time.time()

        # Initialize OCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)

        # Run OCR
        results = ocr.ocr(str(image_path), cls=True)

        # Extract text
        text_lines = []
        for line in results[0]:
            text_lines.append(line[1][0])

        text = "\n".join(text_lines)

        end = time.time()

        # Extract prices and items
        prices = extract_prices_from_text(text)
        items = pair_items_with_prices(text, prices)

        return OCRResult(
            engine="PaddleOCR",
            processing_time=round(end - start, 2),
            raw_text=text,
            items_found=items,
            prices_found=prices,
            success=True,
        )

    except Exception as e:
        return OCRResult(
            engine="PaddleOCR",
            processing_time=0,
            raw_text="",
            items_found=[],
            prices_found=[],
            success=False,
            error=str(e),
        )


def test_apple_vision(image_path: Path) -> OCRResult:
    """Test Apple Vision Framework (macOS only)."""

    try:
        import Vision
        from Quartz import CIImage
        from Foundation import NSURL

        print("  📖 Running Apple Vision Framework...")
        start = time.time()

        # Load image
        url = NSURL.fileURLWithPath_(str(image_path.absolute()))
        ci_image = CIImage.imageWithContentsOfURL_(url)

        # Create request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        # Create handler and perform
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
        success = handler.performRequests_error_([request], None)

        # Extract text
        text_lines = []
        if success[0]:
            observations = request.results()
            for observation in observations:
                text_lines.append(observation.text())

        text = "\n".join(text_lines)

        end = time.time()

        # Extract prices and items
        prices = extract_prices_from_text(text)
        items = pair_items_with_prices(text, prices)

        return OCRResult(
            engine="Apple Vision",
            processing_time=round(end - start, 2),
            raw_text=text,
            items_found=items,
            prices_found=prices,
            success=True,
        )

    except Exception as e:
        return OCRResult(
            engine="Apple Vision",
            processing_time=0,
            raw_text="",
            items_found=[],
            prices_found=[],
            success=False,
            error=str(e),
        )


def print_comparison(results: List[OCRResult]):
    """Print comparison of all OCR engines."""

    print("\n" + "=" * 80)
    print("📊 OCR ENGINE COMPARISON")
    print("=" * 80)

    # Create comparison table
    print(f"\n{'Engine':<20} {'Time':<10} {'Prices':<10} {'Items':<10} {'Status':<10}")
    print("-" * 80)

    for result in results:
        status = "✅ OK" if result.success else "❌ FAIL"
        print(
            f"{result.engine:<20} "
            f"{result.processing_time:>6.2f}s   "
            f"{len(result.prices_found):>5}      "
            f"{len(result.items_found):>5}      "
            f"{status}"
        )

    print("\n" + "=" * 80)
    print("📦 DETAILED RESULTS")
    print("=" * 80)

    for result in results:
        if not result.success:
            print(f"\n❌ {result.engine} failed: {result.error}")
            continue

        print(f"\n{'='*80}")
        print(f"🔍 {result.engine}")
        print(f"{'='*80}")

        print(f"\n⏱️  Processing Time: {result.processing_time}s")
        print(f"💰 Prices Found: {len(result.prices_found)}")
        print(f"📦 Items Found: {len(result.items_found)}")

        if result.prices_found:
            print("\n💵 Prices Extracted:")
            for i, price in enumerate(result.prices_found[:20], 1):
                print(f"   {i}. {price}")
            if len(result.prices_found) > 20:
                print(f"   ... and {len(result.prices_found) - 20} more")

        if result.items_found:
            print("\n🛒 Items Extracted:")
            for i, item in enumerate(result.items_found[:10], 1):
                print(f"   {i}. {item['name'][:50]:<50} → ${item['price']}")
            if len(result.items_found) > 10:
                print(f"   ... and {len(result.items_found) - 10} more")


def save_results(results: List[OCRResult], output_path: Path):
    """Save results to JSON."""

    output_data = {
        "results": [asdict(r) for r in results],
        "comparison": {
            "fastest": min(
                [r for r in results if r.success], key=lambda x: x.processing_time
            ).engine
            if any(r.success for r in results)
            else None,
            "most_prices": max(
                [r for r in results if r.success], key=lambda x: len(x.prices_found)
            ).engine
            if any(r.success for r in results)
            else None,
            "most_items": max(
                [r for r in results if r.success], key=lambda x: len(x.items_found)
            ).engine
            if any(r.success for r in results)
            else None,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test local OCR engines")
    parser.add_argument("--image", type=Path, required=True, help="Path to flyer image")
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["tesseract", "easyocr", "paddleocr", "apple"],
        help="Specific engines to test (default: all)",
    )
    parser.add_argument("--output", type=Path, help="Save results to JSON")

    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        return

    print("=" * 80)
    print("🧪 LOCAL OCR TESTING SUITE - M2 MacBook Air Optimized")
    print("=" * 80)
    print(f"\n📸 Testing image: {args.image}")

    results = []

    # Determine which engines to test
    engines_to_test = (
        args.engines if args.engines else ["tesseract", "easyocr", "paddleocr", "apple"]
    )

    if "tesseract" in engines_to_test:
        print("\n🔍 Testing Tesseract...")
        results.append(test_tesseract(args.image))

    if "easyocr" in engines_to_test:
        print("\n🔍 Testing EasyOCR...")
        results.append(test_easyocr(args.image))

    if "paddleocr" in engines_to_test:
        print("\n🔍 Testing PaddleOCR...")
        results.append(test_paddleocr(args.image))

    if "apple" in engines_to_test:
        print("\n🔍 Testing Apple Vision...")
        results.append(test_apple_vision(args.image))

    # Print comparison
    print_comparison(results)

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    successful = [r for r in results if r.success]
    if successful:
        fastest = min(successful, key=lambda x: x.processing_time)
        most_accurate = max(successful, key=lambda x: len(x.items_found))

        print(f"\n⚡ Fastest: {fastest.engine} ({fastest.processing_time}s)")
        print(
            f"🎯 Most Items Found: {most_accurate.engine} ({len(most_accurate.items_found)} items)"
        )

        print("\n💰 Cost Comparison (for 1000 flyers, 12 pages each):")
        print("   - Local OCR: $0 (free, runs locally)")
        print("   - Claude Vision API: ~$290")
        print("   - Savings: 100% ($290)")

        print("\n⚠️  Trade-offs:")
        print("   - Local: Free, private, but may have lower accuracy")
        print("   - Vision API: Costs money, but typically 95-98% accuracy")
        print("   - Recommendation: Use local for simple cases, API for complex/critical")

    # Save if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
