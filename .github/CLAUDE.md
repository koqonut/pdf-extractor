# CLAUDE.md - AI Assistant Guide for pdf-extractor

> **Last Updated**: 2025-11-14
> **Project**: pdf2img - PDF Image Extraction Tool
> **Template**: Cookiecutter Data Science
> **License**: MIT

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Key Technologies & Dependencies](#key-technologies--dependencies)
4. [Development Setup](#development-setup)
5. [Code Conventions & Standards](#code-conventions--standards)
6. [Development Workflows](#development-workflows)
7. [Data Pipeline Architecture](#data-pipeline-architecture)
8. [Key Entry Points](#key-entry-points)
9. [Important Notebooks](#important-notebooks)
10. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

**Purpose**: Extract images from PDF files using multiple approaches (pdf2image, PyMuPDF) and perform OCR (Optical Character Recognition) on extracted images.

**Current Status**:
- **Maturity**: Early-stage development (1 commit)
- **Active Development**: PDF-to-image conversion and OCR workflows
- **Implementation**: Working examples in Jupyter notebooks, template structure for production code

**Core Capabilities**:
1. **PDF to Image Conversion**: Two approaches implemented
   - `pdf2image` library (simpler, slower)
   - PyMuPDF (faster, more features including embedded image extraction)
2. **OCR Processing**: Image-to-text extraction using OpenCV + Tesseract
3. **ML Pipeline**: Template structure ready for machine learning workflows

**Project Goals**:
- Provide flexible PDF image extraction solutions
- Support multiple extraction strategies for different use cases
- Enable downstream text extraction via OCR
- Establish ML pipeline for potential document classification/analysis

---

## Codebase Structure

```
/home/user/pdf-extractor/
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
├── Makefile                   # Task automation (lint, format, setup)
├── pyproject.toml            # Modern Python project config (Flit build system)
├── setup.cfg                 # Flake8 linter configuration
├── environment.yml           # Conda environment specification
│
├── docs/                     # MkDocs documentation
│   ├── README.md            # Documentation build instructions
│   ├── mkdocs.yml           # MkDocs configuration
│   └── docs/
│       ├── index.md         # Documentation index
│       └── getting-started.md
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── convert_by_pdf2img.ipynb      # PDF→Image (pdf2image library)
│   ├── convert_by_pymupdf.ipynb      # PDF→Image (PyMuPDF, faster)
│   └── img_2_txt_opencv.ipynb        # Image→Text (OCR via Tesseract)
│
├── pdf2img/                 # Main Python package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # ⭐ Configuration & path management
│   ├── dataset.py          # Data processing CLI (template)
│   ├── features.py         # Feature engineering CLI (template)
│   ├── plots.py            # Visualization CLI (template)
│   └── modeling/
│       ├── __init__.py
│       ├── train.py        # Model training CLI (template)
│       └── predict.py      # Inference CLI (template)
│
├── data/                    # Data directories (gitignored)
│   ├── raw/                # Original unprocessed data
│   ├── interim/            # Intermediate transformed data
│   ├── processed/          # Final processed datasets
│   └── external/           # Third-party data sources
│
├── models/                  # Trained models (gitignored except .gitkeep)
├── references/              # Data dictionaries, manuals
└── reports/                 # Generated analysis outputs
    └── figures/            # Generated graphics and visualizations
```

### Directory Purposes

| Directory | Purpose | Gitignored |
|-----------|---------|------------|
| `pdf2img/` | Main source code package | No |
| `notebooks/` | Jupyter notebooks for exploration and prototyping | No |
| `data/` | All data files (raw, interim, processed, external) | **Yes** |
| `models/` | Trained and serialized models | **Yes** (except .gitkeep) |
| `reports/figures/` | Generated plots and visualizations | No |
| `docs/` | MkDocs documentation source | No |
| `references/` | Documentation and manuals | No |

---

## Key Technologies & Dependencies

### PDF Processing
- **pdf2image** (with poppler): Convert PDF pages to PIL Image objects
- **PyMuPDF (pymupdf)**: Fast PDF rendering, page-to-image conversion, embedded image extraction

### Computer Vision & OCR
- **OpenCV (cv2)**: Image preprocessing (grayscale, thresholding, morphological operations)
- **Tesseract**: Open-source OCR engine
- **pytesseract**: Python wrapper for Tesseract OCR

### Data Science Stack
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization

### CLI & Development Tools
- **typer**: Modern CLI framework (used in all module scripts)
- **loguru**: Advanced logging with colored output and tqdm integration
- **tqdm**: Progress bars for long-running operations
- **python-dotenv**: Environment variable management from .env files

### Code Quality
- **black**: Code formatter (line length: 99 characters)
- **flake8**: Python linting
- **isort**: Import sorting (ruff-based)

### Documentation & Environment
- **mkdocs**: Documentation generation
- **ipykernel**: Jupyter notebook support
- **conda**: Environment management

### Python Version
- **Required**: Python 3.10.x (specified in pyproject.toml: `~=3.10`)
- **Conda env**: Python 3.11 (environment.yml)

---

## Development Setup

### 1. Create Conda Environment

```bash
# Using environment.yml (recommended)
conda env create -f environment.yml
conda activate ds_101

# OR using Makefile
make create_environment
conda activate pdf2img
```

### 2. Install Dependencies

```bash
# If using conda environment.yml (already done in step 1)
# Dependencies are automatically installed

# OR using pip
make requirements

# OR manually
pip install -r requirements.txt
```

### 3. Install Package in Development Mode

```bash
pip install -e .
```

### 4. Verify Setup

```bash
# Check Python version
python --version  # Should be 3.10.x or 3.11

# Test imports
python -c "import pdf2img; from pdf2img import config"

# Run linter
make lint
```

---

## Code Conventions & Standards

### Style Guidelines

**Line Length**: 99 characters (configured in pyproject.toml)

**Code Formatting**: Use `black` formatter
```bash
make format          # Auto-format code
make lint           # Check code style
```

**Import Sorting**: Use `isort` with black profile
- First-party modules: `pdf2img`
- Force sort within sections
- Run via: `isort --profile black pdf2img/`

### Linting Configuration (setup.cfg)

**Ignored Flake8 Errors**:
- `E731`: Lambda assignment
- `E266`: Too many leading '#' for block comment
- `E501`: Line too long (handled by black)
- `C901`: Function complexity
- `W503`: Line break before binary operator

**Excluded from Linting**:
- `.git/`, `notebooks/`, `references/`, `models/`, `data/`

### Logging Standards

**Use loguru for all logging**:
```python
from loguru import logger

logger.info("Processing started")
logger.warning("Potential issue detected")
logger.error("Operation failed")
```

**Progress Bars**: Use `tqdm` for long-running operations
```python
from tqdm import tqdm

for item in tqdm(items, desc="Processing"):
    # Process item
    pass
```

**Integration**: The config.py automatically integrates loguru with tqdm for clean output.

### Path Management

**Always use config.py for paths**:
```python
from pdf2img.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# Good
input_path = RAW_DATA_DIR / "input.pdf"

# Avoid hardcoding paths
# Bad: input_path = "data/raw/input.pdf"
```

### CLI Module Pattern

All module scripts follow this pattern:
```python
import typer
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from pdf2img.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "input.csv",
    output_path: Path = PROCESSED_DATA_DIR / "output.csv",
):
    """Process data from input to output."""
    logger.info(f"Processing {input_path} -> {output_path}")

    # Your implementation here
    for item in tqdm(items):
        # Process
        pass

    logger.success("Processing complete")

if __name__ == "__main__":
    app()
```

---

## Development Workflows

### Common Tasks (via Makefile)

```bash
# View all available commands
make help

# Clean compiled Python files
make clean

# Install dependencies
make requirements

# Format code
make format

# Lint code
make lint

# Create conda environment
make create_environment

# Process dataset (template - needs implementation)
make data
```

### Development Cycle

1. **Explore in Notebooks**: Prototype in `notebooks/` directory
2. **Implement in Modules**: Move working code to `pdf2img/` package
3. **Format & Lint**: Run `make format && make lint`
4. **Test**: Verify functionality
5. **Document**: Update docstrings and documentation
6. **Commit**: Commit with clear messages

### Notebook Naming Convention

Format: `<number>-<initials>-<description>.ipynb`

Examples:
- `1.0-jqp-initial-data-exploration.ipynb`
- `2.0-kq-pdf-conversion-comparison.ipynb`

Current notebooks don't follow this convention yet but should be renamed if more are added.

---

## Data Pipeline Architecture

### Data Flow Pattern

```
┌─────────────────┐
│   Raw PDFs      │  data/raw/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PDF to Images   │  notebooks/convert_*.ipynb
│ (pdf2image or   │
│  PyMuPDF)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Files     │  data/interim/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OCR Processing  │  notebooks/img_2_txt_opencv.ipynb
│ (Tesseract)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extracted Text  │  data/processed/
└─────────────────┘
```

### Standard Data Directories

| Directory | Stage | Description |
|-----------|-------|-------------|
| `data/raw/` | Input | Original, immutable PDFs and source data |
| `data/interim/` | Processing | Converted images, intermediate results |
| `data/processed/` | Output | Final datasets, extracted text, features |
| `data/external/` | Reference | Third-party data sources |

### Module Pipeline (Template Structure)

```
dataset.py → features.py → modeling/train.py → modeling/predict.py
                ↓                    ↓
            plots.py            models/*.pkl
```

**Note**: Most modules are currently templates with placeholder comments. Implementation needed.

---

## Key Entry Points

### 1. Configuration (`pdf2img/config.py`) ⭐ **FULLY IMPLEMENTED**

**Purpose**: Central configuration, path management, logging setup

**Key Features**:
- Loads `.env` file for environment variables
- Defines all standard directory paths
- Initializes loguru logger with tqdm integration
- Project root auto-detection

**Usage**:
```python
from pdf2img.config import (
    PROJ_ROOT,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    logger
)
```

### 2. Dataset Processing (`pdf2img/dataset.py`) **TEMPLATE**

**Purpose**: Load and process raw datasets

**Current State**: Template with comments for implementation

**Default Paths**:
- Input: `data/raw/dataset.csv`
- Output: `data/processed/dataset.csv`

**Usage**:
```bash
python pdf2img/dataset.py --input-path data/raw/input.csv --output-path data/processed/output.csv
```

### 3. Feature Engineering (`pdf2img/features.py`) **TEMPLATE**

**Purpose**: Generate features from processed data

**Current State**: Template with comments for implementation

### 4. Visualization (`pdf2img/plots.py`) **TEMPLATE**

**Purpose**: Generate plots and visualizations

**Default Output**: `reports/figures/plot.png`

### 5. Model Training (`pdf2img/modeling/train.py`) **TEMPLATE**

**Purpose**: Train ML models

**Default Paths**:
- Features: `data/processed/features.csv`
- Labels: `data/processed/labels.csv`
- Model output: `models/model.pkl`

### 6. Model Inference (`pdf2img/modeling/predict.py`) **TEMPLATE**

**Purpose**: Run predictions with trained models

**Default Paths**:
- Model: `models/model.pkl`
- Test features: `data/processed/test_features.csv`
- Predictions: `data/processed/test_predictions.csv`

---

## Important Notebooks

### 1. `convert_by_pdf2img.ipynb` - PDF to Image (Simple Approach)

**Library**: pdf2image
**Performance**: ~38 seconds for 14 pages
**Resolution**: 300 DPI

**Key Code**:
```python
from pdf2image import convert_from_path

# Convert PDF to images
images = convert_from_path('input.pdf', dpi=300)

# Save images
for i, image in enumerate(images):
    image.save(f'data/images/page_{i}.png', 'PNG')
```

**Pros**: Simple API, reliable
**Cons**: Slower performance

### 2. `convert_by_pymupdf.ipynb` - PDF to Image (Advanced Approach) ⭐

**Library**: PyMuPDF (pymupdf)
**Performance**: ~1.1 seconds for 14 pages (35x faster!)
**Resolution**: Configurable via matrix parameter

**Key Code**:
```python
import pymupdf

# Open PDF
doc = pymupdf.open('input.pdf')

# Method 1: Render pages to images (fast)
for page_num, page in enumerate(doc):
    pix = page.get_pixmap()
    pix.save(f'data/images/page_{page_num}.png')

# Method 2: Extract embedded images from PDF
for page_num, page in enumerate(doc):
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        with open(f'image_{page_num}_{img_index}.{image_ext}', 'wb') as f:
            f.write(image_bytes)
```

**Pros**: Much faster, can extract embedded images, more control
**Cons**: More complex API

### 3. `img_2_txt_opencv.ipynb` - OCR Processing

**Libraries**: OpenCV + pytesseract
**Purpose**: Extract text from images

**Key Code**:
```python
import cv2
import pytesseract

# Load image
img = cv2.imread('input.png')

# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# OCR
text = pytesseract.image_to_string(close)
print(text)
```

**Preprocessing Steps**:
1. Convert to grayscale
2. Apply Otsu's thresholding
3. Morphological closing to reduce noise

**Example Output**: Successfully extracted "SAVOUR SUMMER SAVINGS" and pricing from metro flyer

---

## AI Assistant Guidelines

### When Working with This Codebase

#### ✅ DO:

1. **Always check config.py first** for available paths and configurations
2. **Use existing path constants** from `pdf2img.config` instead of hardcoding paths
3. **Follow the template pattern** in existing modules when adding new functionality
4. **Use loguru logger** instead of print statements for output
5. **Add tqdm progress bars** for long-running operations
6. **Format code with black** before committing: `make format`
7. **Run linter** to check code quality: `make lint`
8. **Start in notebooks** for prototyping, then move to modules for production code
9. **Respect the data directory structure**: raw → interim → processed
10. **Use typer for CLI** interfaces to maintain consistency
11. **Document functions** with clear docstrings
12. **Update this CLAUDE.md** file when making structural changes

#### ❌ DON'T:

1. **Don't commit data files** (they're gitignored for a reason)
2. **Don't hardcode file paths** - use config.py constants
3. **Don't use print()** for logging - use loguru logger
4. **Don't exceed 99 character line length**
5. **Don't modify .gitignore** to include data files
6. **Don't skip linting** before committing code
7. **Don't create new directory structures** without updating this documentation
8. **Don't mix different logging approaches** - standardize on loguru
9. **Don't implement production features in notebooks** - use them for exploration only
10. **Don't assume requirements.txt exists** - dependencies are in environment.yml and pyproject.toml

### Understanding Template vs. Implemented Code

**Fully Implemented**:
- `pdf2img/config.py` - Configuration and path management
- `notebooks/*.ipynb` - Working PDF conversion and OCR examples

**Template/Placeholder** (needs implementation):
- `pdf2img/dataset.py` - Dataset processing logic
- `pdf2img/features.py` - Feature engineering logic
- `pdf2img/plots.py` - Visualization logic
- `pdf2img/modeling/train.py` - Model training logic
- `pdf2img/modeling/predict.py` - Inference logic

### Recommended Approach for New Features

1. **Research Phase**: Explore in a Jupyter notebook
2. **Prototype Phase**: Implement working solution in notebook
3. **Production Phase**: Move stable code to appropriate module
4. **Testing Phase**: Test CLI interface and error handling
5. **Documentation Phase**: Update docstrings and this CLAUDE.md

### Common Tasks

#### Adding a New PDF Processing Method

1. Create notebook: `notebooks/N-convert_by_<method>.ipynb`
2. Prototype and test the conversion approach
3. If successful, consider adding to a new module or updating existing ones
4. Document performance characteristics and use cases

#### Adding a New Data Processing Step

1. Identify where it fits in the pipeline (dataset, features, etc.)
2. Update the relevant module in `pdf2img/`
3. Follow the existing typer CLI pattern
4. Use config.py paths
5. Add logging and progress bars
6. Test the CLI interface

#### Debugging Issues

1. **Import errors**: Check that environment is activated and package is installed (`pip install -e .`)
2. **Path errors**: Verify paths using `pdf2img.config` constants
3. **Missing dependencies**: Check `environment.yml` and reinstall environment
4. **Linting failures**: Run `make format` to auto-fix formatting issues

### Performance Considerations

**PDF Conversion Performance**:
- **pdf2image**: Slower (~38s for 14 pages) but simpler
- **PyMuPDF**: Faster (~1.1s for 14 pages) but more complex

**Recommendation**: Use PyMuPDF for production workflows requiring speed, pdf2image for simple one-off conversions.

**OCR Performance**:
- Preprocessing significantly improves accuracy
- Consider batching for multiple images
- Tesseract can be slow on high-resolution images

### Git Workflow

**Current Branch**: `claude/claude-md-mhy90kpcwfbwk8c5-01UXNn3h8SBDAMyy7nkhrLrM`

**Commit Guidelines**:
1. Write clear, descriptive commit messages
2. Format and lint before committing
3. Keep commits focused on single changes
4. Reference issue numbers if applicable

**Push Guidelines**:
```bash
# Always push to the feature branch
git push -u origin claude/claude-md-mhy90kpcwfbwk8c5-01UXNn3h8SBDAMyy7nkhrLrM
```

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Activate environment | `conda activate ds_101` or `conda activate pdf2img` |
| Format code | `make format` |
| Lint code | `make lint` |
| Clean compiled files | `make clean` |
| Install dependencies | `make requirements` |
| View help | `make help` |
| Build docs | `cd docs && mkdocs build` |
| Serve docs locally | `cd docs && mkdocs serve` |

### Essential Imports

```python
# Configuration and paths
from pdf2img.config import (
    PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, logger
)

# CLI
import typer

# Progress and logging
from loguru import logger
from tqdm import tqdm

# PDF processing
from pdf2image import convert_from_path
import pymupdf  # PyMuPDF

# OCR
import cv2
import pytesseract

# Data processing
import pandas as pd
import numpy as np
```

### File Locations

| What | Where |
|------|-------|
| Configuration | `pdf2img/config.py` |
| Working examples | `notebooks/*.ipynb` |
| Source code | `pdf2img/*.py` |
| ML models | `models/` (gitignored) |
| Input data | `data/raw/` (gitignored) |
| Output data | `data/processed/` (gitignored) |
| Visualizations | `reports/figures/` |
| Documentation | `docs/` |

---

## Project Metadata

**Project Name**: pdf2img
**Version**: 0.0.1
**Author**: koqonut
**License**: MIT
**Python Version**: 3.10.x
**Build System**: Flit Core 3.2+
**Template**: Cookiecutter Data Science

**Repository Stats** (as of last commit):
- Total commits: 1
- Latest commit: `13d179e` - "notebooks to convert pdf to image using pdf2img, pymupdf"
- Python files: 8
- Jupyter notebooks: 3
- Total Python LOC: ~180

---

## Additional Resources

- **Cookiecutter Data Science**: https://cookiecutter-data-science.drivendata.org/
- **PyMuPDF Documentation**: https://pymupdf.readthedocs.io/
- **pdf2image GitHub**: https://github.com/Belval/pdf2image
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **Typer Documentation**: https://typer.tiangolo.com/
- **Loguru Documentation**: https://loguru.readthedocs.io/

---

## Questions or Issues?

When assisting with this project:
1. Refer to this guide first
2. Check existing notebooks for working examples
3. Review config.py for available paths and setup
4. Follow established patterns in template modules
5. Ask clarifying questions if unsure about project direction
6. Update this document when making structural changes

**Last reviewed**: 2025-11-14
**Document version**: 1.0
**Maintainer**: AI Assistant (Claude)
