#!/usr/bin/env python3
"""
Download a sample flyer for testing the vision API.

This script helps you get a sample flyer image quickly for testing purposes.
"""

import requests
from pathlib import Path
from playwright.sync_api import sync_playwright
import sys


def download_sample_from_url(url: str, output_path: Path):
    """Download image from direct URL."""

    print(f"📥 Downloading from: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Saved to: {output_path}")
    print(f"📏 Size: {len(response.content) / 1024:.1f} KB")


def screenshot_flipp_page(url: str, output_path: Path):
    """Screenshot a Flipp flyer page using browser automation."""

    print(f"🌐 Opening {url} with browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )

        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for flyer to load
            page.wait_for_timeout(3000)

            # Try to find the flyer image element
            # This selector may need adjustment based on actual Flipp structure
            flyer_elem = page.query_selector('.flyer-page, .flyer-image, [class*="flyer"]')

            if flyer_elem:
                print("📸 Taking screenshot of flyer element...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                flyer_elem.screenshot(path=output_path)
            else:
                # Fallback: screenshot visible viewport
                print("📸 Taking screenshot of page...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=output_path)

            print(f"✅ Screenshot saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        finally:
            browser.close()


def main():
    """Download sample flyers for testing."""

    data_dir = Path("data/raw/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("📥 SAMPLE FLYER DOWNLOADER")
    print("=" * 70)

    print("\nOptions:")
    print("1. Download from direct image URL")
    print("2. Screenshot Flipp page (requires Playwright)")
    print("3. Manual instructions")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        url = input("Enter image URL: ").strip()
        filename = input("Enter filename (e.g., metro_sample.png): ").strip()

        if not filename:
            filename = "sample.png"

        output_path = data_dir / filename

        try:
            download_sample_from_url(url, output_path)
            print("\n✅ Done! Test with:")
            print(f"   python test_vision_api.py --image {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif choice == "2":
        print("\nNote: This requires Playwright to be installed:")
        print("   pip install playwright")
        print("   playwright install chromium")

        url = input("\nEnter Flipp URL: ").strip()
        filename = input("Enter filename (e.g., foodbasics_sample.png): ").strip()

        if not filename:
            filename = "sample.png"

        output_path = data_dir / filename

        try:
            screenshot_flipp_page(url, output_path)
            print("\n✅ Done! Test with:")
            print(f"   python test_vision_api.py --image {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    else:
        print("\n📝 Manual Instructions:")
        print("\n1. Open browser and go to:")
        print("   - https://flipp.com")
        print("   - Search for 'Metro' or 'Food Basics' or 'No Frills'")
        print("   - Open a flyer")
        print("\n2. Take screenshot:")
        print("   - Windows: Win+Shift+S")
        print("   - Mac: Cmd+Shift+4")
        print("   - Linux: Shift+PrtScn or use Flameshot")
        print(f"\n3. Save to: {data_dir}/")
        print("\n4. Test with:")
        print(f"   python test_vision_api.py --image {data_dir}/your_image.png")


if __name__ == "__main__":
    main()
