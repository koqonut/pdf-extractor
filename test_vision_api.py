#!/usr/bin/env python3
"""
Test Claude 3.5 Sonnet Vision API for Flyer Data Extraction

This script validates the "Tier 3" approach to determine:
1. Extraction accuracy (items, prices, units)
2. Response quality and structure
3. Cost per flyer
4. Processing time

Usage:
    python test_vision_api.py --image path/to/flyer.png
    python test_vision_api.py --image path/to/flyer.png --api-key YOUR_KEY
"""

import anthropic
import base64
import json
import time
from pathlib import Path
from typing import Dict, List
import argparse
import sys


def load_image_base64(image_path: Path) -> tuple[str, str]:
    """Load image and convert to base64."""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    # Determine media type
    suffix = image_path.suffix.lower()
    media_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    media_type = media_type_map.get(suffix, 'image/jpeg')

    return image_data, media_type


def extract_with_claude_vision(
    image_path: Path,
    api_key: str = None,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict:
    """
    Extract items and prices from flyer using Claude Vision API.

    Args:
        image_path: Path to flyer image
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Claude model to use

    Returns:
        Dict with extraction results and metadata
    """

    print(f"📸 Loading image: {image_path}")
    image_data, media_type = load_image_base64(image_path)

    print(f"🤖 Initializing Claude {model}...")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Craft the extraction prompt
    prompt = """Extract ALL items and prices from this retail flyer image.

For each item, extract:
- Product name (as shown on flyer)
- Price (numeric value)
- Unit (each, lb, kg, 100g, etc.)
- Original price (if on sale)
- Promotional details (e.g., "Buy 2 Get 1 Free", "Save $2", etc.)
- Confidence (0.0-1.0) based on text clarity

Return ONLY valid JSON in this exact format (no markdown, no code blocks):
{
  "metadata": {
    "total_items_found": 0,
    "page_quality": "high/medium/low",
    "extraction_notes": "any relevant observations"
  },
  "items": [
    {
      "name": "Product Name",
      "price": "5.99",
      "unit": "each",
      "original_price": "7.99",
      "promotion": "Save $2",
      "confidence": 0.95,
      "location": "top-left/center/etc"
    }
  ]
}

Rules:
1. Price must be numeric string (e.g., "5.99" not "$5.99")
2. Include ALL visible items, even if partially visible
3. Set confidence lower if text is blurry or unclear
4. Leave fields as null if not applicable
5. For multi-buy deals (e.g., "2 for $5"), calculate unit price if possible
6. Include brand names if visible
"""

    print("⏳ Sending request to Claude API...")
    start_time = time.time()

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,  # Deterministic for data extraction
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

        end_time = time.time()
        processing_time = end_time - start_time

        # Extract response
        response_text = message.content[0].text

        # Parse JSON response
        # Claude might wrap in markdown code blocks, so clean it
        if response_text.strip().startswith("```"):
            # Remove markdown code blocks
            lines = response_text.strip().split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith('json'):
                response_text = response_text[4:]

        extracted_data = json.loads(response_text.strip())

        # Calculate costs
        # Claude 3.5 Sonnet pricing (as of 2024):
        # Input: $3 per million tokens
        # Output: $15 per million tokens

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * 3.00
        output_cost = (output_tokens / 1_000_000) * 15.00
        total_cost = input_cost + output_cost

        # Compile results
        result = {
            "success": True,
            "model": model,
            "extracted_data": extracted_data,
            "performance": {
                "processing_time_seconds": round(processing_time, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "estimated_cost_usd": round(total_cost, 4),
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4)
            },
            "raw_response": response_text
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def print_results(result: Dict):
    """Pretty print extraction results."""

    if not result.get("success"):
        print("\n❌ Extraction failed!")
        print(f"Error: {result.get('error')}")
        print(f"Type: {result.get('error_type')}")
        return

    print("\n✅ Extraction successful!")
    print("\n" + "="*70)
    print("📊 PERFORMANCE METRICS")
    print("="*70)

    perf = result["performance"]
    print(f"⏱️  Processing Time: {perf['processing_time_seconds']}s")
    print(f"🎫 Total Tokens: {perf['total_tokens']:,}")
    print(f"   - Input: {perf['input_tokens']:,} tokens (${perf['input_cost_usd']:.4f})")
    print(f"   - Output: {perf['output_tokens']:,} tokens (${perf['output_cost_usd']:.4f})")
    print(f"💰 Cost: ${perf['estimated_cost_usd']:.4f} per page")

    print("\n" + "="*70)
    print("📦 EXTRACTED DATA")
    print("="*70)

    data = result["extracted_data"]

    if "metadata" in data:
        meta = data["metadata"]
        print(f"\n📋 Metadata:")
        print(f"   - Total items found: {meta.get('total_items_found', 'N/A')}")
        print(f"   - Page quality: {meta.get('page_quality', 'N/A')}")
        if meta.get('extraction_notes'):
            print(f"   - Notes: {meta['extraction_notes']}")

    if "items" in data:
        items = data["items"]
        print(f"\n🛒 Items Found: {len(items)}")
        print("\n" + "-"*70)

        for i, item in enumerate(items, 1):
            print(f"\n{i}. {item.get('name', 'Unknown')}")
            print(f"   💵 Price: ${item.get('price', 'N/A')}")

            if item.get('unit'):
                print(f"   📦 Unit: {item['unit']}")

            if item.get('original_price'):
                print(f"   🏷️  Original: ${item['original_price']}")

            if item.get('promotion'):
                print(f"   🎉 Promo: {item['promotion']}")

            confidence = item.get('confidence', 0)
            confidence_emoji = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🔴"
            print(f"   {confidence_emoji} Confidence: {confidence:.0%}")

            if i >= 10 and len(items) > 10:
                print(f"\n... and {len(items) - 10} more items")
                break

    # Cost projections
    print("\n" + "="*70)
    print("💰 COST PROJECTIONS")
    print("="*70)

    cost_per_page = perf['estimated_cost_usd']

    print(f"\nAssuming {cost_per_page:.4f} per page:")
    print(f"   - 10-page flyer: ${cost_per_page * 10:.3f}")
    print(f"   - 100 flyers (avg 12 pages): ${cost_per_page * 1200:.2f}")
    print(f"   - 1,000 flyers: ${cost_per_page * 12000:.2f}")

    # Quality assessment
    print("\n" + "="*70)
    print("🎯 QUALITY ASSESSMENT")
    print("="*70)

    if "items" in data and len(data["items"]) > 0:
        confidences = [item.get('confidence', 0) for item in data["items"]]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        high_conf = sum(1 for c in confidences if c > 0.9)
        med_conf = sum(1 for c in confidences if 0.7 < c <= 0.9)
        low_conf = sum(1 for c in confidences if c <= 0.7)

        print(f"\n📊 Confidence Distribution:")
        print(f"   🟢 High (>90%): {high_conf} items ({high_conf/len(confidences)*100:.0f}%)")
        print(f"   🟡 Medium (70-90%): {med_conf} items ({med_conf/len(confidences)*100:.0f}%)")
        print(f"   🔴 Low (<70%): {low_conf} items ({low_conf/len(confidences)*100:.0f}%)")
        print(f"   📈 Average: {avg_confidence:.1%}")


def save_results(result: Dict, output_path: Path):
    """Save results to JSON file."""

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Claude 3.5 Sonnet Vision API for flyer extraction"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to flyer image (PNG, JPG, JPEG)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use (default: claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Validate image exists
    if not args.image.exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    # Run extraction
    print("="*70)
    print("🧪 CLAUDE VISION API TEST - FLYER EXTRACTION")
    print("="*70)

    result = extract_with_claude_vision(
        args.image,
        api_key=args.api_key,
        model=args.model
    )

    # Print results
    print_results(result)

    # Save if requested
    if args.output:
        save_results(result, args.output)

    # Exit code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
