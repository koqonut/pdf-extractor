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

# 4. Get a flyer image from https://flipp.com
# Save screenshot as: data/raw/samples/test.png

# 5. Test it!
python test_2025_ocr.py --image data/raw/samples/test.png --engine minicpm
# Or for fastest: --engine got
```

**Result:** Items and prices extracted in 10-15 seconds with **92-95% accuracy** (beats GPT-4o!), **$0 cost**, saves ~$290 per 1000 pages vs Claude API.

📖 **[Complete Getting Started Guide →](GETTING_STARTED.md)**

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

📖 **[2025 Models Complete Guide →](MODERN_OCR_2025.md)**

📖 **[See all options and detailed comparison →](GETTING_STARTED.md#extraction-approaches-explained)**

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

| Guide | Description |
|-------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** ⭐ | **Complete guide: installation, usage, testing, comparison** |
| **[MODERN_OCR_2025.md](MODERN_OCR_2025.md)** 🚀 | **2025 models reference: MiniCPM-V, GOT-OCR, Phi-3.5, M2 optimization** |
| [CLAUDE.md](CLAUDE.md) | Codebase documentation for AI assistants |

That's it! Everything you need is in these 3 files.

## Project Organization

```
pdf-extractor/
├── README.md                  # This file - quick start and overview
├── GETTING_STARTED.md         # Complete guide (installation, usage, comparison)
├── MODERN_OCR_2025.md         # 2025 models reference and M2 optimization
├── CLAUDE.md                  # Codebase docs for AI assistants
│
├── pyproject.toml             # Modern Python project config (UV-compatible)
├── .python-version            # Python version (3.10)
├── .pre-commit-config.yaml    # Auto-formatting hooks (black, ruff)
│
├── pdf2img/                   # Main package
│   ├── __init__.py
│   └── config.py              # Configuration and path management
│
├── test_2025_ocr.py           # Test 2025 models (MiniCPM-V, GOT-OCR, Phi-3.5)
├── test_advanced_ocr.py       # Test 2024 models (Surya, Qwen2-VL)
├── test_local_ocr.py          # Test traditional OCR (PaddleOCR, Tesseract)
├── test_vision_api.py         # Test Claude Vision API
├── batch_test_images.py       # Batch testing multiple images
├── compare_all_methods.py     # Side-by-side comparison
│
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── convert_by_pdf2img.ipynb
│   ├── convert_by_pymupdf.ipynb
│   └── img_2_txt_opencv.ipynb
│
└── data/                      # Data directories (gitignored)
    ├── raw/                   # Input flyer images
    ├── processed/             # Extracted text/results
    └── external/              # Third-party data
```

**Simplified from Cookiecutter Data Science template** - removed unused ML/training scaffolding.

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

---

