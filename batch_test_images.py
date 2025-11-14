#!/usr/bin/env python3
"""
Batch Test Multiple Flyer Images

Test extraction accuracy on a directory of flyer images.
Generates a summary report comparing all methods across all images.

Usage:
    # Test all images in a directory
    python batch_test_images.py --dir data/raw/samples

    # Test specific images
    python batch_test_images.py --images img1.png img2.jpg img3.png

    # Use specific engine
    python batch_test_images.py --dir data/raw/samples --engine paddleocr

    # Compare with Vision API
    python batch_test_images.py --dir data/raw/samples --compare-api
"""

import argparse
from pathlib import Path
from typing import List, Dict
import json
import time
from datetime import datetime

# Import test functions
try:
    from test_local_ocr import (
        test_tesseract, test_easyocr, test_paddleocr,
        test_apple_vision, OCRResult
    )
except ImportError:
    print("❌ Error: test_local_ocr.py not found")
    exit(1)


def find_images(directory: Path) -> List[Path]:
    """Find all image files in directory."""

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG']
    images = []

    for ext in extensions:
        images.extend(directory.glob(ext))

    return sorted(images)


def test_image_with_engine(image_path: Path, engine: str) -> OCRResult:
    """Test single image with specific engine."""

    engine_map = {
        'tesseract': test_tesseract,
        'paddleocr': test_paddleocr,
        'easyocr': test_easyocr,
        'apple': test_apple_vision,
    }

    test_func = engine_map.get(engine.lower())
    if not test_func:
        raise ValueError(f"Unknown engine: {engine}")

    return test_func(image_path)


def test_image_with_api(image_path: Path, api_key: str = None) -> Dict:
    """Test single image with Vision API."""

    try:
        import anthropic
        import base64
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
            'engine': 'Claude API',
            'processing_time': round(end - start, 2),
            'items': items,
            'cost': round(cost, 4)
        }

    except Exception as e:
        return {
            'success': False,
            'engine': 'Claude API',
            'error': str(e),
            'items': []
        }


def batch_test(
    images: List[Path],
    engine: str = 'paddleocr',
    compare_api: bool = False,
    api_key: str = None
) -> Dict:
    """Test multiple images and generate summary."""

    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(images),
        'engine': engine,
        'results': []
    }

    print(f"\n🧪 Testing {len(images)} images with {engine}...")
    print("=" * 80)

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Testing: {image_path.name}")

        # Test with local engine
        local_result = test_image_with_engine(image_path, engine)

        image_result = {
            'image': str(image_path),
            'filename': image_path.name,
            'local': {
                'success': local_result.success,
                'time': local_result.processing_time,
                'items_found': len(local_result.items_found),
                'prices_found': len(local_result.prices_found),
            }
        }

        if local_result.success:
            print(f"   ✅ {engine}: {len(local_result.items_found)} items, "
                  f"{len(local_result.prices_found)} prices ({local_result.processing_time}s)")
        else:
            print(f"   ❌ {engine} failed: {local_result.error}")

        # Test with API if requested
        if compare_api:
            api_result = test_image_with_api(image_path, api_key)

            image_result['api'] = {
                'success': api_result['success'],
                'time': api_result.get('processing_time', 0),
                'items_found': len(api_result.get('items', [])),
                'cost': api_result.get('cost', 0)
            }

            if api_result['success']:
                print(f"   ✅ Claude API: {len(api_result['items'])} items "
                      f"(${api_result['cost']:.4f}, {api_result['processing_time']}s)")

        results['results'].append(image_result)

    return results


def print_summary(results: Dict):
    """Print summary statistics."""

    print("\n" + "=" * 80)
    print(" " * 30 + "📊 SUMMARY REPORT")
    print("=" * 80)

    total = results['total_images']
    engine = results['engine']

    # Local stats
    successful_local = [r for r in results['results'] if r['local']['success']]

    if successful_local:
        total_items = sum(r['local']['items_found'] for r in successful_local)
        total_prices = sum(r['local']['prices_found'] for r in successful_local)
        avg_time = sum(r['local']['time'] for r in successful_local) / len(successful_local)
        avg_items = total_items / len(successful_local)
        avg_prices = total_prices / len(successful_local)

        print(f"\n🔍 {engine.upper()} Results:")
        print(f"   - Successful: {len(successful_local)}/{total} images")
        print(f"   - Total items found: {total_items}")
        print(f"   - Total prices found: {total_prices}")
        print(f"   - Average per image: {avg_items:.1f} items, {avg_prices:.1f} prices")
        print(f"   - Average time: {avg_time:.2f}s per image")
        print(f"   - Total time: {sum(r['local']['time'] for r in successful_local):.2f}s")
        print(f"   - Cost: $0.00 (free!)")

    # API stats if available
    api_results = [r for r in results['results'] if 'api' in r]
    if api_results:
        successful_api = [r for r in api_results if r['api']['success']]

        if successful_api:
            total_items_api = sum(r['api']['items_found'] for r in successful_api)
            total_cost = sum(r['api']['cost'] for r in successful_api)
            avg_time_api = sum(r['api']['time'] for r in successful_api) / len(successful_api)
            avg_items_api = total_items_api / len(successful_api)

            print(f"\n☁️  CLAUDE API Results:")
            print(f"   - Successful: {len(successful_api)}/{total} images")
            print(f"   - Total items found: {total_items_api}")
            print(f"   - Average per image: {avg_items_api:.1f} items")
            print(f"   - Average time: {avg_time_api:.2f}s per image")
            print(f"   - Total cost: ${total_cost:.4f}")
            print(f"   - Cost per image: ${total_cost/len(successful_api):.4f}")

            # Comparison
            if successful_local:
                accuracy_ratio = (avg_items / avg_items_api * 100) if avg_items_api > 0 else 0

                print(f"\n📈 Comparison:")
                print(f"   - {engine} found {accuracy_ratio:.1f}% of items vs Claude API")
                print(f"   - Cost savings: ${total_cost:.2f} (100% saved using local)")

                if accuracy_ratio >= 90:
                    print(f"\n✅ Recommendation: Use {engine} (local)")
                    print(f"   Reason: Achieves {accuracy_ratio:.0f}% of API accuracy at $0 cost")
                elif accuracy_ratio >= 75:
                    print(f"\n⚠️  Recommendation: Hybrid approach")
                    print(f"   - Use {engine} for initial extraction")
                    print(f"   - Use API for low-confidence cases")
                    print(f"   - Estimated cost: ~30-40% of full API cost")
                else:
                    print(f"\n❌ Recommendation: Use Claude API")
                    print(f"   Reason: Local only achieves {accuracy_ratio:.0f}% accuracy")

    # Per-image breakdown
    print(f"\n" + "=" * 80)
    print(" " * 30 + "📋 PER-IMAGE RESULTS")
    print("=" * 80)

    print(f"\n{'Image':<40} {'Local Items':<15} {'API Items':<15} {'Cost'}")
    print("-" * 80)

    for r in results['results']:
        filename = r['filename'][:38]
        local_items = r['local']['items_found'] if r['local']['success'] else 'FAILED'

        if 'api' in r and r['api']['success']:
            api_items = r['api']['items_found']
            cost = f"${r['api']['cost']:.4f}"
        else:
            api_items = 'N/A'
            cost = 'N/A'

        print(f"{filename:<40} {str(local_items):<15} {str(api_items):<15} {cost}")


def save_report(results: Dict, output_path: Path):
    """Save detailed results to JSON."""

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch test flyer images")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dir", type=Path, help="Directory containing images")
    input_group.add_argument("--images", nargs='+', type=Path, help="Specific image files")

    # Engine options
    parser.add_argument(
        "--engine",
        default="paddleocr",
        choices=['tesseract', 'paddleocr', 'easyocr', 'apple'],
        help="OCR engine to test (default: paddleocr)"
    )

    # API comparison
    parser.add_argument(
        "--compare-api",
        action='store_true',
        help="Also test with Claude Vision API for comparison"
    )
    parser.add_argument("--api-key", type=str, help="Anthropic API key")

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/batch_test_results.json"),
        help="Output path for detailed results"
    )

    args = parser.parse_args()

    # Get images
    if args.dir:
        if not args.dir.exists():
            print(f"❌ Directory not found: {args.dir}")
            return

        images = find_images(args.dir)

        if not images:
            print(f"❌ No images found in: {args.dir}")
            print("   Supported formats: PNG, JPG, JPEG, WebP")
            return
    else:
        images = args.images

        # Validate all exist
        for img in images:
            if not img.exists():
                print(f"❌ Image not found: {img}")
                return

    print("=" * 80)
    print(" " * 25 + "🧪 BATCH IMAGE TESTING")
    print("=" * 80)
    print(f"\nFound {len(images)} images to test")
    print(f"Engine: {args.engine}")

    if args.compare_api:
        print("API Comparison: Enabled (will also test with Claude API)")

    # Run tests
    results = batch_test(
        images,
        engine=args.engine,
        compare_api=args.compare_api,
        api_key=args.api_key
    )

    # Print summary
    print_summary(results)

    # Save detailed results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(results, args.output)

    print("\n" + "=" * 80)
    print(" " * 35 + "✅ Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
