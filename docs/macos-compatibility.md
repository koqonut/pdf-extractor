# macOS Apple Silicon (M1/M2/M3) Compatibility Guide

## ✅ Confirmed Working OCR Engines

### 1. Apple Vision Framework (RECOMMENDED for macOS) ⭐

**Why:** Native Apple framework, optimized for Apple Silicon, excellent accuracy and speed.

**Python Wrappers:**

#### ocrmac (Simple, Fast)
```bash
pip install ocrmac

# Usage:
python -c "import ocrmac; print(ocrmac.ocr('image.png'))"
```
- **Performance:** 207ms per image on M3 Max
- **Languages:** Supports multiple languages
- **License:** MIT (Open Source)
- **Source:** https://github.com/straussmaximilian/ocrmac

#### apple-vision-utils (Feature-Rich)
```bash
pip install apple-vision-utils

# Command line usage:
apple-vision image.png --output result.txt
```
- **Formats:** PNG, JPEG, TIFF, WebP
- **Multi-language support**
- **License:** Open Source
- **Source:** https://pypi.org/project/apple-vision-utils/

---

### 2. Tesseract OCR (Traditional, Reliable)

**Why:** Industry standard, ARM-optimized since v5.0, works via Homebrew.

```bash
# Install via Homebrew:
brew install tesseract

# Python wrapper:
pip install pytesseract

# Usage:
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('image.png'))
```

**Performance:**
- **M2 Max:** 198 images in 13 seconds
- **Intel Mac:** Same task takes 1 minute 34 seconds
- **Optimization:** ARM-optimized in Tesseract 5.0+

**Source:** https://github.com/tesseract-ocr/tesseract

---

### 3. Surya OCR (Modern, Multi-language)

**Why:** Modern deep learning OCR, 90+ languages, works on MPS with some limitations.

```bash
pip install surya-ocr

# Usage:
from surya import OCRModel
model = OCRModel()
results = model.ocr(['image.png'], device='mps')
```

**Status:**
- ✅ Works on M1/M2 with MPS
- ⚠️ Some performance issues with PyTorch MPS implementation
- ⚠️ Text detection has MPS bug (Apple-side issue)
- ✅ Can use CPU fallback for stability

**Source:** https://github.com/VikParuchuri/surya

---

### 4. EasyOCR (Multi-language)

**Why:** Easy to use, 80+ languages, MPS support added Sept 2023.

```bash
pip install easyocr

# Usage:
import easyocr
reader = easyocr.Reader(['en'], gpu=True)  # Uses MPS on Apple Silicon
result = reader.readtext('image.png')
```

**Status:**
- ✅ MPS support as of September 2023
- ✅ Works on Apple Silicon
- **Source:** https://github.com/JaidedAI/EasyOCR

---

## ⚠️ Limited/Problematic on Apple Silicon

### PaddleOCR
- ❌ Poor native Apple Silicon support
- ⚠️ Requires Rosetta 2 emulation
- ⚠️ Newer versions don't support M1/M2 natively
- 🔧 Workaround: Use with `arch -x86_64 python` via Rosetta

### TrOCR (Transformer-based)
- ⚠️ MPS tensor operation errors
- ❌ Training doesn't work on MPS (even M4)
- 🔧 Workaround: Use CPU mode

---

## ❌ Does NOT Work on macOS

### 2025 VLMs (GOT-OCR, MiniCPM-V, Phi-3.5)
See main README for details.

---

## 📊 Performance Comparison on M2 Air 16GB

| Engine | Speed | Accuracy | Memory | MPS Support | Status |
|--------|-------|----------|--------|-------------|--------|
| **Apple Vision** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Excellent | ~500MB | ✅ Native | ✅ Best choice |
| **Tesseract 5.0+** | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Good | ~200MB | N/A (CPU) | ✅ Reliable |
| **Surya** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Excellent | ~2GB | ⚠️ Partial | ✅ Works |
| **EasyOCR** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Very Good | ~1-2GB | ✅ Yes | ✅ Works |
| **PaddleOCR** | ⭐⭐ Slow | ⭐⭐⭐⭐ Very Good | ~1GB | ❌ No | ⚠️ Rosetta only |
| **TrOCR** | ⭐⭐ Slow | ⭐⭐⭐⭐ Very Good | ~1GB | ❌ Broken | ⚠️ CPU only |

---

## 🎯 Recommendations for M2 Air 16GB

### For Best Performance:
```bash
# Use Apple Vision Framework (fastest, most accurate on macOS)
pip install ocrmac
```

### For Multi-language Support:
```bash
# Use EasyOCR (80+ languages, MPS support)
pip install easyocr
```

### For Traditional OCR:
```bash
# Use Tesseract (industry standard)
brew install tesseract
pip install pytesseract
```

### For Modern Deep Learning OCR:
```bash
# Use Surya (90+ languages, layout analysis)
pip install surya-ocr
```

---

## 🔬 Testing on Your M2 Air

### Test Script:
```python
import time
from PIL import Image

def test_ocr_engine(engine_name, ocr_func, image_path):
    """Test an OCR engine"""
    start = time.time()
    try:
        text = ocr_func(image_path)
        elapsed = time.time() - start
        print(f"✅ {engine_name}: {elapsed:.2f}s")
        print(f"   Extracted {len(text)} characters")
        return True
    except Exception as e:
        print(f"❌ {engine_name}: {e}")
        return False

# Test Apple Vision
def test_vision(path):
    import ocrmac
    return ocrmac.ocr(path).as_text()

# Test Tesseract
def test_tesseract(path):
    import pytesseract
    return pytesseract.image_to_string(Image.open(path))

# Test EasyOCR
def test_easyocr(path):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(path)
    return ' '.join([text for _, text, _ in result])

# Run tests
image = "data/raw/samples/test_fb.png"
test_ocr_engine("Apple Vision", test_vision, image)
test_ocr_engine("Tesseract", test_tesseract, image)
test_ocr_engine("EasyOCR", test_easyocr, image)
```

---

## 📝 Key Takeaways

1. **Apple Vision Framework is the best choice for macOS** - Native, fast, accurate
2. **Tesseract works great** - ARM-optimized, reliable, traditional approach
3. **EasyOCR and Surya work** - Modern deep learning, good multi-language support
4. **2025 VLMs (GOT-OCR, MiniCPM-V, Phi-3.5) don't work** - Use Linux with CUDA
5. **For production macOS OCR:** Use Apple Vision Framework or Tesseract

---

## 🐳 For 2025 VLMs: Use Docker with CUDA

If you need GOT-OCR, MiniCPM-V, or Phi-3.5, use a Linux machine with NVIDIA GPU or cloud GPU instance (AWS, Google Cloud, etc).

```bash
# Future: Docker container with CUDA support
# This will allow running 2025 VLMs on any platform including macOS
```

---

## Sources

- Apple Vision Framework: https://github.com/straussmaximilian/ocrmac
- Tesseract ARM support: https://formulae.brew.sh/formula/tesseract
- Surya MPS discussion: https://github.com/VikParuchuri/surya/issues/207
- EasyOCR M1 support: https://github.com/JaidedAI/EasyOCR/issues/406
- M2 performance: https://www.owlocr.com/blog/posts/m2-max-stunningly-fast-in-text-recognition
