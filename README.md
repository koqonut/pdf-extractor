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

# 3. Install best local OCR (Surya - 90-93% accuracy, free)
uv pip install -e ".[ocr-surya]"

# 4. Get a flyer image from https://flipp.com
# Save screenshot as: data/raw/samples/test.png

# 5. Test it!
python test_advanced_ocr.py --image data/raw/samples/test.png --engine surya
```

**Result:** Items and prices extracted in 2-4 seconds with 90-93% accuracy, $0 cost.

📖 **[Complete Getting Started Guide →](GETTING_STARTED.md)**

---

## 🎯 Comparison: Local vs Cloud

| Method | Accuracy | Speed | Cost (1000 flyers) | Runs On |
|--------|----------|-------|-------------------|---------|
| **Surya** (Modern OCR) | 90-93% | 2-4s | **$0** | M2 Air |
| **Qwen2-VL** (Vision-LLM) | 92-95% | 10-15s | **$0** | M2 Air |
| **Claude API** (Cloud) | 96-98% | 3-4s | **$290** | Cloud |

**Recommendation:** Start with Surya (free, fast, good accuracy). Add Qwen2-VL for hard cases. Use Claude API only if needed.

📖 **[See all options and detailed comparison →](GETTING_STARTED.md#extraction-approaches-explained)**

## What's Included

- ✅ **Direct image testing** (PNG, JPG, WebP - no PDF conversion needed!)
- ✅ **PDF to Image conversion** (PyMuPDF - 35x faster than pdf2image)
- ✅ **Traditional OCR** (Tesseract, PaddleOCR, EasyOCR, Apple Vision)
- ✅ **Modern ML-based OCR** (Surya, TrOCR, DocTR - 90-95% accuracy) ⭐ NEW!
- ✅ **Vision-Language Models** (Qwen2-VL, Florence-2 - structured extraction) ⭐ NEW!
- ✅ **Cloud Vision API** (Claude 3.5 Sonnet - 96-98% accuracy)
- ✅ **Batch testing** - Test multiple images at once
- ✅ **Complete testing suite** - Compare all methods side-by-side
- ✅ **M2 MacBook Air optimized** - Uses Neural Engine
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
| [ADVANCED_OCR_OPTIONS.md](ADVANCED_OCR_OPTIONS.md) | All OCR engines explained (Surya, Qwen2-VL, TrOCR, etc.) |
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

