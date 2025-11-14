# UV Quick Start Guide

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

## Why UV?

- ⚡ **10-100x faster** than pip
- 🔒 **Automatic lock file** for reproducible installs
- 🎯 **Simple commands** - easier than pip/poetry/pipenv
- 🦀 **Written in Rust** - blazing fast performance
- 💾 **Built-in caching** - downloads once, reuse everywhere

---

## Install UV (One-Time Setup)

### macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Or via Homebrew (macOS):
```bash
brew install uv
```

**Verify installation:**
```bash
uv --version
```

---

## Quick Start (Choose Your Path)

### Option 1: Just Test Local OCR (Recommended for M2 Air) ⭐

Install only what you need for local testing:

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install core + PaddleOCR (best local OCR)
uv pip install -e ".[pdf,ocr-paddle]"
```

**Test immediately:**
```bash
python test_local_ocr.py --image data/raw/samples/test_flyer.png --engines paddleocr
```

---

### Option 2: Test Cloud Vision API

```bash
# Create and activate venv
uv venv
source .venv/bin/activate

# Install core + Vision API
uv pip install -e ".[pdf,vision-api]"

# Set API key
export ANTHROPIC_API_KEY='sk-ant-your-key'

# Test
python test_vision_api.py --image data/raw/samples/test_flyer.png
```

---

### Option 3: Install Everything (Complete Testing)

```bash
uv venv
source .venv/bin/activate

# Install all dependencies
uv pip install -e ".[all]"
```

**This installs:**
- ✅ All OCR engines (Tesseract, PaddleOCR, EasyOCR, Apple Vision)
- ✅ Vision API (Claude 3.5 Sonnet)
- ✅ PDF processing (PyMuPDF, pdf2image)
- ✅ ML tools (pandas, numpy, scikit-learn)
- ✅ Dev tools (black, flake8, pytest)

---

## Installation Options Reference

| Command | What It Installs | Use Case |
|---------|-----------------|----------|
| `uv pip install -e .` | Core only | Basic setup |
| `uv pip install -e ".[pdf]"` | + PDF processing | Convert PDFs to images |
| `uv pip install -e ".[ocr-paddle]"` | + PaddleOCR | Best local OCR ⭐ |
| `uv pip install -e ".[ocr-apple]"` | + Apple Vision | macOS native OCR |
| `uv pip install -e ".[ocr-tesseract]"` | + Tesseract | Traditional OCR |
| `uv pip install -e ".[ocr-easy]"` | + EasyOCR | Deep learning OCR |
| `uv pip install -e ".[ocr-all]"` | + All local OCR | Test everything local |
| `uv pip install -e ".[vision-api]"` | + Claude API | Cloud extraction |
| `uv pip install -e ".[local]"` | + All local tools | Everything except API |
| `uv pip install -e ".[all]"` | Everything | Complete installation |

---

## Common Workflows

### For Quick Local Testing (M2 Air):

```bash
# 1. Setup (one-time)
uv venv
source .venv/bin/activate
uv pip install -e ".[pdf,ocr-paddle]"

# 2. Test
python test_local_ocr.py --image your_flyer.png --engines paddleocr

# 3. Done! (Takes ~5 minutes including download)
```

---

### For Vision API Testing:

```bash
# 1. Setup
uv venv
source .venv/bin/activate
uv pip install -e ".[vision-api]"

# 2. Set API key
export ANTHROPIC_API_KEY='your-key'

# 3. Test
python test_vision_api.py --image your_flyer.png
```

---

### For Complete Comparison:

```bash
# 1. Setup
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"

# 2. Compare all methods
python compare_all_methods.py --image your_flyer.png
```

---

## UV Commands Cheat Sheet

| Task | UV Command | Old Way (pip) |
|------|-----------|---------------|
| Create venv | `uv venv` | `python -m venv .venv` |
| Install package | `uv pip install package` | `pip install package` |
| Install from pyproject | `uv pip install -e .` | `pip install -e .` |
| Install extras | `uv pip install -e ".[dev]"` | `pip install -e ".[dev]"` |
| Sync dependencies | `uv pip sync` | `pip install -r requirements.txt` |
| List packages | `uv pip list` | `pip list` |
| Uninstall | `uv pip uninstall package` | `pip uninstall package` |

---

## Why This Is Better

### Before (pip):
```bash
# Slow, unreliable
pip install -e .
# Takes 2-5 minutes, may fail halfway
```

### After (uv):
```bash
# Fast, cached, reliable
uv pip install -e .
# Takes 10-30 seconds, cached for reuse
```

---

## Updating Dependencies

```bash
# Update all packages
uv pip install --upgrade -e ".[all]"

# Update specific package
uv pip install --upgrade paddleocr
```

---

## Working with Multiple Projects

UV automatically caches packages across all projects:

```bash
# First project - downloads packages
cd project1
uv pip install paddleocr  # Downloads ~300MB

# Second project - reuses cache
cd ../project2
uv pip install paddleocr  # Instant! Uses cache
```

---

## Troubleshooting

### Issue: "uv: command not found"

```bash
# Install uv first:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew:
brew install uv
```

### Issue: "No module named 'pdf2img'"

```bash
# Make sure you installed in editable mode:
uv pip install -e .
```

### Issue: Virtual environment not activated

```bash
# Activate it:
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# You should see (.venv) in your prompt
```

### Issue: Package conflicts

```bash
# Remove venv and start fresh:
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

---

## Development Workflow

```bash
# 1. Clone repo
git clone <repo-url>
cd pdf-extractor

# 2. Create venv
uv venv

# 3. Activate
source .venv/bin/activate

# 4. Install for development
uv pip install -e ".[dev]"

# 5. Run tests
pytest

# 6. Format code
make format

# 7. Lint
make lint
```

---

## Comparing Installation Times

| Method | Time (First Install) | Time (Cached) |
|--------|---------------------|---------------|
| pip | 3-5 minutes | 2-4 minutes |
| uv | 30-60 seconds | 5-10 seconds |
| **Speed Up** | **3-5x faster** | **12-24x faster** |

---

## Next Steps

1. **For M2 Air local testing:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[pdf,ocr-paddle]"
   python test_local_ocr.py --image your_flyer.png --engines paddleocr
   ```

2. **For Vision API testing:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[vision-api]"
   export ANTHROPIC_API_KEY='your-key'
   python test_vision_api.py --image your_flyer.png
   ```

3. **For complete comparison:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[all]"
   python compare_all_methods.py --image your_flyer.png
   ```

---

## More Information

- UV Documentation: https://github.com/astral-sh/uv
- This project's guides:
  - `M2_SETUP_GUIDE.md` - Local OCR on M2 MacBook Air
  - `VISION_API_TESTING.md` - Cloud Vision API testing
  - `FLYER_EXTRACTION_STRATEGY.md` - Overall strategy
  - `CLAUDE.md` - Complete project documentation
