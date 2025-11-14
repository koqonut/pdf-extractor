#!/bin/bash
# Quick Local OCR Test for M2 MacBook Air
# This script sets up and tests the best local OCR options

set -e

echo "======================================================================"
echo "🍎 M2 MacBook Air - Local OCR Quick Test"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is optimized for macOS"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Python
if ! command_exists python3; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "✅ Python found: $(python3 --version)"

# Check pip
if ! command_exists pip3; then
    echo "❌ pip not found"
    exit 1
fi
echo "✅ pip found"

# Check Homebrew (for Tesseract)
if ! command_exists brew; then
    echo "⚠️  Homebrew not found. Install from https://brew.sh"
    echo "   (Optional - needed for Tesseract)"
else
    echo "✅ Homebrew found"
fi

echo -e "\n${YELLOW}Step 2: Choose installation level...${NC}"
echo "1) Quick (Apple Vision only - 0 min, already installed)"
echo "2) Recommended (Apple Vision + PaddleOCR - 5 min) ⭐"
echo "3) Full (All engines - 10 min)"

read -p "Select option (1-3): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Installing: Apple Vision only${NC}"
        pip3 install pyobjc-framework-Vision pyobjc-framework-Quartz pillow
        TEST_ENGINES="apple"
        ;;
    2)
        echo -e "\n${GREEN}Installing: Apple Vision + PaddleOCR (Recommended)${NC}"
        pip3 install pyobjc-framework-Vision pyobjc-framework-Quartz
        pip3 install paddlepaddle paddleocr pillow opencv-python
        TEST_ENGINES="apple paddleocr"
        ;;
    3)
        echo -e "\n${GREEN}Installing: All engines${NC}"

        # Tesseract
        if command_exists brew; then
            echo "Installing Tesseract..."
            brew install tesseract 2>/dev/null || echo "Tesseract already installed"
        fi

        # Python packages
        pip3 install pyobjc-framework-Vision pyobjc-framework-Quartz
        pip3 install paddlepaddle paddleocr
        pip3 install easyocr
        pip3 install pytesseract opencv-python pillow
        TEST_ENGINES="tesseract apple easyocr paddleocr"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✅ Installation complete!${NC}"

# Check for test image
echo -e "\n${YELLOW}Step 3: Checking for test image...${NC}"

TEST_IMAGE="data/raw/samples/test_flyer.png"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "⚠️  No test image found at: $TEST_IMAGE"
    echo ""
    echo "To get a test image:"
    echo "1. Visit https://flipp.com"
    echo "2. Search for 'Metro' or 'Food Basics'"
    echo "3. Take a screenshot of a flyer page"
    echo "4. Save to: $TEST_IMAGE"
    echo ""
    read -p "Press Enter when you have a test image ready..."

    if [ ! -f "$TEST_IMAGE" ]; then
        echo "❌ Still no test image found. Exiting."
        exit 1
    fi
fi

echo "✅ Test image found: $TEST_IMAGE"

# Run test
echo -e "\n${YELLOW}Step 4: Running OCR tests...${NC}"
echo "This may take a few minutes on first run (downloading models)"
echo ""

python3 test_local_ocr.py --image "$TEST_IMAGE" --engines $TEST_ENGINES

echo -e "\n${GREEN}======================================================================"
echo "✅ Test complete!"
echo "======================================================================${NC}"

echo ""
echo "Next steps:"
echo "1. Review the results above"
echo "2. Test on more flyers: python3 test_local_ocr.py --image <path>"
echo "3. Compare with Vision API: python3 test_vision_api.py --image <path>"
echo ""
echo "See M2_SETUP_GUIDE.md for more details!"
