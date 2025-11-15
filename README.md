# pdf2img - Flyer Data Extraction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Extract items and prices from retail flyers using OCR and ML.**

Compare local (free, 90-95% accuracy) vs cloud (paid, 96-98% accuracy) extraction methods.

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup project
git clone <this-repo>
cd pdf-extractor
uv venv && source .venv/bin/activate

# 3. Install 2025 models (best accuracy - beats GPT-4o!) 🚀 NEW!
uv pip install -e ".[recommended-2025]"

# Alternative: For 8GB M2 Air
uv pip install -e ".[m2-performance]"

# 3b. Authenticate with Hugging Face (required for MiniCPM-V)
huggingface-cli login
# Or set: export HF_TOKEN=your_token_here

# 4. Get a flyer image from https://flipp.com
# Save screenshot as: data/raw/samples/test.png

# 5. Test it!
python test_ocr.py test --image data/raw/samples/test.png --engine minicpm
# Or for fastest: python test_ocr.py test --image test.png --engine got-ocr

# Compare all engines
python test_ocr.py compare --image test.png
```

**Result:** Items and prices extracted in 10-15 seconds with **92-95% accuracy** (beats GPT-4o!), **$0 cost**, saves ~$290 per 1000 pages vs Claude API.

**New:** Unified test script with plugin system - adding new OCR engines is now trivial!

---

## 🎯 Comparison: Local vs Cloud

### 2025 Models (Latest & Best!) 🚀

| Method | Accuracy | Speed | Cost (1000 flyers) | M2 Air 8GB? |
|--------|----------|-------|-------------------|-------------|
| **MiniCPM-V 2.6** 🏆 | **92-95%** | 10-15s | **$0** | ✅ (4-bit) |
| **GOT-OCR 2.0** ⚡ | 90-93% | **2-3s** | **$0** | ✅ Great |
| **Phi-3.5 Vision** | 88-92% | 5-8s | **$0** | ✅ Great |

### 2024 Models (Still Great!)

| Method | Accuracy | Speed | Cost (1000 flyers) | M2 Air 8GB? |
|--------|----------|-------|-------------------|-------------|
| **Surya** | 90-93% | 2-4s | **$0** | ✅ Great |
| **Qwen2-VL-2B** | 92-95% | 10-15s | **$0** | ✅ (4-bit) |

### Cloud APIs

| Method | Accuracy | Speed | Cost (1000 flyers) | Notes |
|--------|----------|-------|-------------------|-------|
| **Claude API** | 96-98% | 3-4s | **$290** | Only 2-3% better than MiniCPM-V |

**Recommendation:** Use **MiniCPM-V 2.6** (beats GPT-4o, free, local). Or **GOT-OCR 2.0** for fastest speed. Only use Claude API if you need 96-98% accuracy.

📖 **[2025 Models Reference Guide →](docs/models-2025.md)**

## What's Included

- ✅ **2025 State-of-the-Art Models** 🚀 NEW!
  - **MiniCPM-V 2.6** (92-95%, beats GPT-4o!)
  - **GOT-OCR 2.0** (90-93%, 2-3s, lightweight)
  - **Phi-3.5 Vision** (88-92%, M2 optimized)
  - **PaliGemma 2** (87-91%, Google's VLM)
- ✅ **Direct image testing** (PNG, JPG, WebP - no PDF conversion needed!)
- ✅ **PDF to Image conversion** (PyMuPDF - 35x faster than pdf2image)
- ✅ **Traditional OCR** (Tesseract, PaddleOCR, EasyOCR, Apple Vision)
- ✅ **2024 ML-based OCR** (Surya, TrOCR, DocTR - 90-95% accuracy)
- ✅ **Vision-Language Models** (Qwen2-VL, Florence-2 - structured extraction)
- ✅ **Cloud Vision API** (Claude 3.5 Sonnet - 96-98% accuracy)
- ✅ **Batch testing** - Test multiple images at once
- ✅ **Complete testing suite** - Compare all methods side-by-side
- ✅ **M2 MacBook Air optimized** - 4-bit quantization, Neural Engine
- ✅ **Cost analysis tools** - Local ($0) vs Cloud ($0.024/page)

## 📚 Documentation

**This README has everything to get started!**

For advanced topics:
- **[2025 Models Reference](docs/models-2025.md)** - Detailed model comparisons, M2 optimization, advanced configuration
- **[AI Assistant Guide](.github/CLAUDE.md)** - Codebase documentation for AI assistants

## Project Organization

```
pdf-extractor/
├── README.md                  # 👈 You are here - everything you need!
│
├── test_ocr.py                # 🎯 Main CLI - unified test interface
│
├── pdf2img/                   # Main package
│   ├── config.py              # Configuration & paths
│   └── engines/               # OCR engine plugins
│       ├── base.py            # Plugin system core
│       ├── minicpm.py         # MiniCPM-V 2.6 (best accuracy)
│       ├── got_ocr.py         # GOT-OCR 2.0 (fastest)
│       └── phi3.py            # Phi-3.5 Vision
│
├── tests/                     # Pytest test suite (44 tests, 56% coverage)
│   ├── conftest.py            # Fixtures & mock engine
│   ├── test_base.py           # Core plugin tests
│   └── test_engines.py        # Engine integration tests
│
├── docs/                      # Advanced documentation
│   └── models-2025.md         # 2025 models reference
│
├── scripts/                   # Utility scripts (deprecated - use test_ocr.py)
├── notebooks/                 # Jupyter exploration
├── .github/                   # CI/CD, PR templates, AI docs
└── data/                      # Data directories (gitignored)
```

**Clean & minimal** - removed unused ML scaffolding, moved utilities to `scripts/`.

**Industry standard structure:** Config at root, source in package, tests separate, scripts organized.

---

## 🛠️ Development Setup

**Auto-formatting with pre-commit** (optional but recommended):

```bash
# Install pre-commit
uv pip install pre-commit

# Install git hooks
pre-commit install

# Now black and ruff will auto-format on every commit
# Or run manually: pre-commit run --all-files
```

This will automatically format your code with `black` and lint with `ruff` on every commit.

**Running Tests:**

```bash
# Install dev dependencies (includes pytest)
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=pdf2img --cov-report=html
# View coverage: open htmlcov/index.html

# Run only fast tests (skip slow model tests)
pytest -m "not slow"

# Run specific test file
pytest tests/test_base.py

# Run with verbose output
pytest -v

# Run tests and show print statements
pytest -s
```

**Test Organization:**
- `tests/test_base.py` - Core plugin system tests (fast)
- `tests/test_engines.py` - Engine integration tests (slow if models installed)
- `tests/conftest.py` - Shared fixtures and mock engine

**Test Markers:**
- `@pytest.mark.unit` - Fast unit tests (no external dependencies)
- `@pytest.mark.slow` - Slow tests requiring model downloads
- `@pytest.mark.integration` - Integration tests

---

## 🔧 Troubleshooting

### Missing Dependencies Error

If you see errors like `No module named 'tiktoken'` or `No module named 'torchvision'`:

```bash
# You need to install the specific engine dependencies
# For GOT-OCR:
uv pip install -e ".[ocr-got]"

# For Phi-3.5 Vision:
uv pip install -e ".[vlm-phi3]"

# For Surya:
uv pip install -e ".[ocr-surya]"

# Or install all recommended 2025 models:
uv pip install -e ".[recommended-2025]"
```

### Hugging Face Authentication Error

If you see `Access to model openbmb/MiniCPM-V-2_6 is restricted`:

```bash
# 1. Visit the model page and request access (approval usually instant):
# https://huggingface.co/openbmb/MiniCPM-V-2_6
# Click "Request Access" button

# 2. Get a token from https://huggingface.co/settings/tokens
# Click "New token" → "Read" access is enough

# 3. Login with the token:
huggingface-cli login
# Paste your token when prompted

# Or set as environment variable:
export HF_TOKEN=your_token_here

# 4. Try again - should work now!
python test_ocr.py compare --image test_fb.png
```

### All Engines Failing

If all engines show "FAIL" status:

```bash
# Check what's installed:
pip list | grep -E "transformers|torch|surya"

# Reinstall with specific engines:
uv pip install -e ".[ocr-got,vlm-phi3,ocr-surya]"
```

---

## 🔌 Plugin System for OCR Engines

The project uses a **plugin pattern** that makes adding new OCR engines trivial!

### Using Engines

```python
# List all available engines
from pdf2img.engines import list_engines
print(list_engines())  # ['minicpm', 'got-ocr', 'phi3', 'surya', ...]

# Get an engine
from pdf2img.engines import get_engine
engine = get_engine('minicpm')

# Extract text
result = engine.extract('flyer.png')
print(result.text)
```

### Adding a New Engine (Super Easy!)

1. **Create** `pdf2img/engines/myengine.py`:

```python
from pdf2img.engines import OCREngine, OCRResult, register_engine

@register_engine  # Auto-registers!
class MyEngine(OCREngine):
    name = "my-engine"
    model_size = "1B"
    ram_usage_gb = 2.0

    def extract(self, image_path, **kwargs):
        # Your OCR code here
        return OCRResult(
            engine=self.name,
            text=extracted_text,
            processing_time=time_taken,
            model_size=self.model_size
        )
```

2. **Import** in `pdf2img/engines/__init__.py`:

```python
from .myengine import MyEngine  # That's it!
```

3. **Test** immediately:

```bash
python test_ocr.py test --image flyer.png --engine my-engine
```

**No need to modify test scripts, CLI parsers, or comparison logic!** It all works automatically.

### CLI Commands

```bash
# List all engines
python test_ocr.py list

# Get engine info
python test_ocr.py info --engine minicpm

# Test one engine
python test_ocr.py test --image flyer.png --engine got-ocr

# Compare all engines
python test_ocr.py compare --image flyer.png

# Batch process
python test_ocr.py batch data/images/*.png --engine phi3

# Show engines table
python test_ocr.py engines-table
```

---
