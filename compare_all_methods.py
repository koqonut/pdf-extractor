#!/usr/bin/env python3
"""
Compare All Extraction Methods Side-by-Side

Tests and compares:
1. Local OCR (Tesseract, EasyOCR, PaddleOCR, Apple Vision)
2. Cloud Vision API (Claude 3.5 Sonnet)

Provides direct comparison of accuracy, speed, and cost.

Usage:
    python compare_all_methods.py --image path/to/flyer.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

# Import test functions
try:
    from test_local_ocr import (
        test_tesseract, test_easyocr, test_paddleocr,
        test_apple_vision, OCRResult
    )
except ImportError:
    print("❌ Error: test_local_ocr.py not found")
    sys.exit(1)


def test_vision_api_simple(image_path: Path, api_key: str = None) -> Dict:
    """Simple wrapper for Vision API test."""

    try:
        import anthropic
        import base64
        import time
        import re

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()

        media_type = 'image/png' if image_path.suffix.lower() == '.png' else 'image/jpeg'

        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        prompt = """Extract ALL items and prices from this flyer. Return JSON only:
{"items": [{"name": "...", "price": "..."}]}"""

        start = time.time()

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        end = time.time()

        response_text = message.content[0].text

        # Clean markdown
        if response_text.strip().startswith("```"):
            lines = response_text.strip().split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith('json'):
                response_text = response_text[4:]

        try:
            data = json.loads(response_text.strip())
            items = data.get('items', [])
        except:
            items = []

        # Calculate cost
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = (input_tokens / 1_000_000) * 3.00 + (output_tokens / 1_000_000) * 15.00

        return {
            'success': True,
            'engine': 'Claude 3.5 Sonnet',
            'processing_time': round(end - start, 2),
            'items_found': items,
            'cost': round(cost, 4)
        }

    except Exception as e:
        return {
            'success': False,
            'engine': 'Claude 3.5 Sonnet',
            'error': str(e)
        }


def print_side_by_side_comparison(local_results: List[OCRResult], cloud_result: Dict):
    """Print comprehensive comparison."""

    print("\n" + "="*90)
    print(" " * 30 + "📊 COMPLETE COMPARISON")
    print("="*90)

    # Table header
    print(f"\n{'Method':<25} {'Time':<12} {'Items':<10} {'Cost/Page':<12} {'Status'}")
    print("-" * 90)

    # Local results
    for result in local_results:
        if result.success:
            status = "✅"
            print(f"{result.engine:<25} "
                  f"{result.processing_time:>6.2f}s      "
                  f"{len(result.items_found):>5}      "
                  f"{'$0.00':>10}  "
                  f"{status}")

    # Cloud result
    if cloud_result.get('success'):
        items_count = len(cloud_result.get('items_found', []))
        cost = cloud_result.get('cost', 0)
        time_taken = cloud_result.get('processing_time', 0)

        print(f"{cloud_result['engine']:<25} "
              f"{time_taken:>6.2f}s      "
              f"{items_count:>5}      "
              f"${cost:>9.4f}  "
              f"✅")
    else:
        print(f"{cloud_result['engine']:<25} "
              f"{'N/A':>6}       "
              f"{'N/A':>5}      "
              f"{'N/A':>10}  "
              f"❌ {cloud_result.get('error', 'Failed')[:20]}")

    # Cost comparison
    print("\n" + "="*90)
    print(" " * 30 + "💰 COST ANALYSIS")
    print("="*90)

    if cloud_result.get('success'):
        cost_per_page = cloud_result.get('cost', 0)

        print(f"\n{'Scenario':<40} {'Local':<15} {'Cloud':<15} {'Savings'}")
        print("-" * 90)

        scenarios = [
            ("Single flyer (12 pages)", 12),
            ("100 flyers", 1200),
            ("1,000 flyers", 12000),
        ]

        for scenario, pages in scenarios:
            cloud_cost = cost_per_page * pages
            savings = cloud_cost  # Since local is $0

            print(f"{scenario:<40} "
                  f"{'$0.00':>13}  "
                  f"${cloud_cost:>12.2f}  "
                  f"${savings:>10.2f}")

    # Accuracy comparison (if we have item counts)
    print("\n" + "="*90)
    print(" " * 30 + "🎯 ACCURACY INSIGHTS")
    print("="*90)

    successful_local = [r for r in local_results if r.success]

    if successful_local and cloud_result.get('success'):
        cloud_items = len(cloud_result.get('items_found', []))

        print(f"\n{'Method':<25} {'Items Found':<15} {'vs Claude'}")
        print("-" * 90)

        for result in successful_local:
            local_items = len(result.items_found)
            if cloud_items > 0:
                percentage = (local_items / cloud_items) * 100
                print(f"{result.engine:<25} {local_items:>8}        {percentage:>6.0f}%")

        print(f"{cloud_result['engine']:<25} {cloud_items:>8}        100% (baseline)")

    # Recommendations
    print("\n" + "="*90)
    print(" " * 30 + "💡 RECOMMENDATIONS")
    print("="*90)

    if successful_local:
        best_local = max(successful_local, key=lambda x: len(x.items_found))
        fastest_local = min(successful_local, key=lambda x: x.processing_time)

        print(f"\n🏆 Best Local Accuracy: {best_local.engine}")
        print(f"   - Found {len(best_local.items_found)} items")
        print(f"   - Processing time: {best_local.processing_time}s")
        print(f"   - Cost: $0")

        print(f"\n⚡ Fastest Local: {fastest_local.engine}")
        print(f"   - Processing time: {fastest_local.processing_time}s")
        print(f"   - Found {len(fastest_local.items_found)} items")

        if cloud_result.get('success'):
            cloud_items = len(cloud_result.get('items_found', []))
            best_local_items = len(best_local.items_found)

            if best_local_items >= cloud_items * 0.9:  # Within 90% of cloud
                print(f"\n✅ RECOMMENDATION: Use {best_local.engine} (local)")
                print(f"   Reason: Achieves {(best_local_items/cloud_items)*100:.0f}% of cloud accuracy at $0 cost")
            elif best_local_items >= cloud_items * 0.75:  # 75-90% of cloud
                print(f"\n⚠️  RECOMMENDATION: Hybrid approach")
                print(f"   - Use {best_local.engine} for initial extraction")
                print(f"   - Use Claude API for low-confidence cases")
                print(f"   - Estimated cost: ~30% of full cloud ($0-100 per 1000 flyers)")
            else:
                print(f"\n❌ RECOMMENDATION: Use Claude API (cloud)")
                print(f"   Reason: Local methods only achieve {(best_local_items/cloud_items)*100:.0f}% accuracy")
                print(f"   Cost is worth it for reliability")


def main():
    parser = argparse.ArgumentParser(description="Compare all extraction methods")
    parser.add_argument("--image", type=Path, required=True, help="Path to flyer image")
    parser.add_argument("--api-key", type=str, help="Anthropic API key for cloud test")
    parser.add_argument("--skip-cloud", action='store_true', help="Skip cloud API test")
    parser.add_argument("--local-engines", nargs='+',
                       choices=['tesseract', 'easyocr', 'paddleocr', 'apple'],
                       help="Specific local engines to test")

    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        return

    print("="*90)
    print(" " * 20 + "🧪 COMPLETE EXTRACTION METHOD COMPARISON")
    print("="*90)
    print(f"\n📸 Testing image: {args.image}\n")

    # Test local engines
    print("🔍 Testing Local Engines...")
    print("-" * 90)

    local_results = []
    engines = args.local_engines if args.local_engines else ['tesseract', 'apple', 'paddleocr']

    if 'tesseract' in engines:
        print("\n   Testing Tesseract...")
        local_results.append(test_tesseract(args.image))

    if 'apple' in engines:
        print("\n   Testing Apple Vision...")
        local_results.append(test_apple_vision(args.image))

    if 'easyocr' in engines:
        print("\n   Testing EasyOCR...")
        local_results.append(test_easyocr(args.image))

    if 'paddleocr' in engines:
        print("\n   Testing PaddleOCR...")
        local_results.append(test_paddleocr(args.image))

    # Test cloud API
    cloud_result = {}
    if not args.skip_cloud:
        print("\n" + "-" * 90)
        print("☁️  Testing Cloud Vision API...")
        print("-" * 90)
        print("\n   Testing Claude 3.5 Sonnet...")
        cloud_result = test_vision_api_simple(args.image, api_key=args.api_key)
    else:
        cloud_result = {'success': False, 'engine': 'Claude 3.5 Sonnet (skipped)'}

    # Print comparison
    print_side_by_side_comparison(local_results, cloud_result)

    print("\n" + "="*90)
    print(" " * 35 + "✅ Testing Complete!")
    print("="*90)
    print("\nFor detailed analysis, see:")
    print("  - M2_SETUP_GUIDE.md (local setup)")
    print("  - VISION_API_TESTING.md (cloud setup)")
    print("  - FLYER_EXTRACTION_STRATEGY.md (overall strategy)")


if __name__ == "__main__":
    main()
