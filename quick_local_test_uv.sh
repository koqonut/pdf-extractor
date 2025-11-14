#!/bin/bash
# Quick Local OCR Test for M2 MacBook Air (UV Version)
# This script sets up and tests the best local OCR options using UV

set -e

echo "======================================================================"
echo "🍎 M2 MacBook Air - Local OCR Quick Test (UV Edition)"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check UV
if ! command_exists uv; then
    echo "⚠️  UV not found. Installing UV..."

    # Install UV
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command_exists uv; then
            echo "❌ UV installation failed. Please install manually:"
            echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
    else
        echo "❌ Please install UV manually:"
        echo "   https://github.com/astral-sh/uv#installation"
        exit 1
    fi
fi

echo "✅ UV found: $(uv --version)"

echo -e "\n${YELLOW}Step 2: Choose installation level...${NC}"
echo "1) Quick (Apple Vision only - 10 seconds) ⚡"
echo "2) Recommended (Apple Vision + PaddleOCR - 1 minute) ⭐"
echo "3) Full (All local OCR engines - 3 minutes)"
echo "4) Everything (Local + Cloud Vision API - 4 minutes)"

read -p "Select option (1-4): " choice

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "\n${BLUE}Creating virtual environment...${NC}"
    uv venv
    echo "✅ Virtual environment created"
fi

# Activate venv
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Install based on choice
case $choice in
    1)
        echo -e "\n${GREEN}Installing: Apple Vision only${NC}"
        uv pip install -e ".[pdf,ocr-apple]"
        TEST_ENGINES="apple"
        ;;
    2)
        echo -e "\n${GREEN}Installing: Apple Vision + PaddleOCR (Recommended)${NC}"
        uv pip install -e ".[pdf,ocr-apple,ocr-paddle]"
        TEST_ENGINES="apple paddleocr"
        ;;
    3)
        echo -e "\n${GREEN}Installing: All local OCR engines${NC}"

        # Check for Homebrew and Tesseract (optional)
        if command_exists brew; then
            echo "Installing Tesseract via Homebrew..."
            brew install tesseract 2>/dev/null || echo "Tesseract already installed"
        else
            echo "⚠️  Homebrew not found. Skipping Tesseract installation."
            echo "   Install Homebrew from https://brew.sh to use Tesseract"
        fi

        uv pip install -e ".[pdf,ocr-all]"
        TEST_ENGINES="apple paddleocr tesseract easyocr"
        ;;
    4)
        echo -e "\n${GREEN}Installing: Everything (Local + Cloud Vision API)${NC}"

        if command_exists brew; then
            brew install tesseract 2>/dev/null || echo "Tesseract already installed"
        fi

        uv pip install -e ".[all]"
        TEST_ENGINES="apple paddleocr"

        # Check for API key
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo ""
            echo "⚠️  Note: To test Cloud Vision API, set your API key:"
            echo "   export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
        fi
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
        echo "❌ Still no test image found."
        echo ""
        echo "Setup complete! To test later, run:"
        echo "  source .venv/bin/activate"
        echo "  python test_local_ocr.py --image <your-image.png> --engines $TEST_ENGINES"
        exit 0
    fi
fi

echo "✅ Test image found: $TEST_IMAGE"

# Run test
echo -e "\n${YELLOW}Step 4: Running OCR tests...${NC}"
echo "This may take a few minutes on first run (downloading models)"
echo ""

python test_local_ocr.py --image "$TEST_IMAGE" --engines $TEST_ENGINES

echo -e "\n${GREEN}======================================================================"
echo "✅ Test complete!"
echo "======================================================================${NC}"

echo ""
echo "Next steps:"
echo "1. Review the results above"
echo "2. Test on more flyers:"
echo "   ${BLUE}python test_local_ocr.py --image <path>${NC}"
echo ""
echo "3. Compare with Vision API:"
echo "   ${BLUE}python compare_all_methods.py --image <path>${NC}"
echo ""
echo "4. Deactivate venv when done:"
echo "   ${BLUE}deactivate${NC}"
echo ""
echo "See UV_QUICKSTART.md for more details!"
