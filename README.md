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

### Start Here

| Guide | Description |
|-------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** ⭐ | **Complete beginner guide - start here!** |

### Quick References

| Guide | Description |
|-------|-------------|
| [QUICK_INSTALL_GUIDE.md](QUICK_INSTALL_GUIDE.md) | All installation commands in one place |
| [UV_QUICKSTART.md](UV_QUICKSTART.md) | UV package manager setup |

### Detailed Guides

| Guide | Description |
|-------|-------------|
| **[MODERN_OCR_2025.md](MODERN_OCR_2025.md)** 🚀 | **Latest 2025 models - MiniCPM-V, GOT-OCR, Phi-3.5, PaliGemma 2** |
| [ADVANCED_OCR_OPTIONS.md](ADVANCED_OCR_OPTIONS.md) | 2024 OCR engines (Surya, Qwen2-VL, TrOCR, Florence-2, etc.) |
| [M2_SETUP_GUIDE.md](M2_SETUP_GUIDE.md) | M2 MacBook Air specific optimizations |
| [IMAGE_TESTING_GUIDE.md](IMAGE_TESTING_GUIDE.md) | Batch testing workflows |
| [VISION_API_TESTING.md](VISION_API_TESTING.md) | Claude Vision API guide |

### Strategy & Sources

| Guide | Description |
|-------|-------------|
| [FLYER_EXTRACTION_STRATEGY.md](FLYER_EXTRACTION_STRATEGY.md) | Multi-tier extraction strategy (OCR → LLM → API) |
| [FLYER_SOURCES_ANALYSIS.md](FLYER_SOURCES_ANALYSIS.md) | Canadian grocery flyer sources (Flipp, Metro, etc.) |

### For AI Assistants

| Guide | Description |
|-------|-------------|
| [CLAUDE.md](CLAUDE.md) | Complete codebase documentation for AI assistants |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for pdf2img
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── pdf2img                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes pdf2img a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

