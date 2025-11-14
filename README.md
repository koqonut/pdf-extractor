# pdf2img

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Extract images from PDFs and flyers with OCR. Test local (free) vs cloud (accurate) extraction methods.

## Quick Start

**Using UV (Recommended - 10x faster):**

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup and test (choose one):

# Option 1: Local OCR only (M2 Air optimized)
uv venv && source .venv/bin/activate
uv pip install -e ".[pdf,ocr-paddle]"

# Option 2: Cloud Vision API
uv venv && source .venv/bin/activate
uv pip install -e ".[vision-api]"
export ANTHROPIC_API_KEY='your-key'

# Option 3: Everything
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```

**Or use automated script:**
```bash
chmod +x quick_local_test_uv.sh
./quick_local_test_uv.sh
```

**📖 See [UV_QUICKSTART.md](UV_QUICKSTART.md) for detailed instructions**

## What's Included

- ✅ **PDF to Image conversion** (PyMuPDF - 35x faster than pdf2image)
- ✅ **Local OCR engines** (Tesseract, PaddleOCR, EasyOCR, Apple Vision)
- ✅ **Cloud Vision API** (Claude 3.5 Sonnet - 95%+ accuracy)
- ✅ **Complete testing suite** - Compare all methods side-by-side
- ✅ **M2 MacBook Air optimized** - Uses Neural Engine
- ✅ **Cost analysis tools** - Local ($0) vs Cloud ($0.024/page)

## Documentation

| Guide | Description |
|-------|-------------|
| [UV_QUICKSTART.md](UV_QUICKSTART.md) | UV setup and dependency management ⭐ |
| [M2_SETUP_GUIDE.md](M2_SETUP_GUIDE.md) | Local OCR testing on M2 MacBook Air |
| [VISION_API_TESTING.md](VISION_API_TESTING.md) | Cloud Vision API testing |
| [FLYER_EXTRACTION_STRATEGY.md](FLYER_EXTRACTION_STRATEGY.md) | Complete extraction strategy |
| [FLYER_SOURCES_ANALYSIS.md](FLYER_SOURCES_ANALYSIS.md) | Canadian grocery flyer sources |
| [CLAUDE.md](CLAUDE.md) | Complete project documentation |

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

