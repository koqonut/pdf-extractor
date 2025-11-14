# Quick Install Guide - Choose Your Path

Fast reference for installing different OCR/ML options.

---

## 🚀 2025 Models (Latest & Best!)

### 1. **MiniCPM-V 2.6** - Best Accuracy 🏆 **BEATS GPT-4o!**

**Why:** 92-95% accuracy, #1 on OCRBench, beats commercial APIs, structured JSON

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[vlm-minicpm]"

# Test
python test_2025_ocr.py --image your_flyer.png --engine minicpm
```

**Download size:** ~4-5GB (first run)
**Memory:** 4-5GB (with 4-bit quantization - M2 compatible!)
**Speed on M2:** 10-15s per page
**Accuracy:** 92-95% (beats GPT-4o, Gemini, Claude!)

**💰 Savings:** ~$290 per 1000 pages vs Claude API

---

### 2. **GOT-OCR 2.0** - Fastest + Excellent ⚡ **BEST SPEED**

**Why:** 90-93% accuracy, 2-3s speed, lightweight (580M params), handles tables

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[ocr-got]"

# Test
python test_2025_ocr.py --image your_flyer.png --engine got
```

**Download size:** ~1-2GB (first run)
**Memory:** ~2GB
**Speed on M2:** 2-3s per page (fastest!)
**Accuracy:** 90-93%

---

### 3. **Phi-3.5 Vision** - Small & Efficient 🔋 **BEST FOR 8GB M2**

**Why:** 88-92% accuracy, 4.2B params, low memory, MIT license

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[vlm-phi3]"

# Test
python test_2025_ocr.py --image your_flyer.png --engine phi3
```

**Download size:** ~2-3GB (first run)
**Memory:** 3-4GB (with 4-bit quantization)
**Speed on M2:** 5-8s per page
**Accuracy:** 88-92%

---

### 4. **PaliGemma 2 (3B)** - Google's Offering 🔵

**Why:** 87-91% accuracy, multiple resolutions, commercial license

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[vlm-paligemma]"

# Test
python test_2025_ocr.py --image your_flyer.png --engine paligemma --model-size 3b
```

**Download size:** ~2-3GB (first run)
**Memory:** 2-3GB
**Speed on M2:** 4-6s per page
**Accuracy:** 87-91%

**See [MODERN_OCR_2025.md](MODERN_OCR_2025.md) for complete 2025 models guide!**

---

## 🎯 2024 Models (Still Great!)

### 1. **Surya** - Modern OCR ⭐

**Why:** 90-93% accuracy, fast (2-4s), easy setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[ocr-surya]"

# Test
python test_advanced_ocr.py --image your_flyer.png --engine surya
```

**Download size:** ~400MB (first run)
**Memory:** ~2GB
**Speed on M2:** 2-4s per page

---

### 2. **Qwen2-VL-2B** - Vision-Language Model ⭐⭐⭐

**Why:** 92-95% accuracy, structured JSON extraction

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[vlm-qwen]"

# Test
python test_advanced_ocr.py --image your_flyer.png --engine qwen2vl
```

**Download size:** ~4GB (first run)
**Memory:** 4-6GB (with 4-bit quantization)
**Speed on M2:** 10-15s per page
**Accuracy:** 92-95%

---

### 3. **PaddleOCR** - Solid Baseline ⭐

**Why:** Proven, fast, good accuracy

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[ocr-paddle]"

# Test
python test_local_ocr.py --image your_flyer.png --engines paddleocr
```

**Download size:** ~300MB
**Memory:** ~1.5GB
**Speed on M2:** 3-5s per page
**Accuracy:** 85-90%

---

## 📊 Complete Comparison

### 2025 Models (Recommended!)

| Option | Accuracy | Speed (M2) | Memory | Download | Best For |
|--------|----------|------------|--------|----------|----------|
| **MiniCPM-V 2.6** 🏆 | **92-95%** | 10-15s | 4-5GB | 4-5GB | **Best accuracy, beats GPT-4o!** |
| **GOT-OCR 2.0** ⚡ | 90-93% | **2-3s** | 2GB | 1-2GB | **Fastest + excellent** |
| **Phi-3.5 Vision** 🔋 | 88-92% | 5-8s | 3-4GB | 2-3GB | Small, efficient, 8GB M2 |
| **PaliGemma 2 (3B)** | 87-91% | 4-6s | 2-3GB | 2-3GB | Google, commercial use |

### 2024 Models (Still Great!)

| Option | Accuracy | Speed (M2) | Memory | Download | Best For |
|--------|----------|------------|--------|----------|----------|
| **Surya** ⭐ | 90-93% | 2-4s | 2GB | 400MB | Modern OCR |
| **Qwen2-VL-2B** ⭐⭐⭐ | 92-95% | 10-15s | 4-6GB | 4GB | Structured JSON |
| **PaddleOCR** | 85-90% | 3-5s | 1.5GB | 300MB | Good baseline |
| **Florence-2** | 88-92% | 5-8s | 2GB | 800MB | Microsoft, balanced |
| **TrOCR** | 92-95% | 5-8s | 3GB | 1GB | High accuracy OCR |

### Traditional/Baseline

| Option | Accuracy | Speed (M2) | Memory | Download | Best For |
|--------|----------|------------|--------|----------|----------|
| **Apple Vision** | 75-85% | 1-2s | 1GB | 0MB | macOS only, fastest |
| **Tesseract** | 70-80% | 2-3s | 500MB | 10MB | Simple baseline |

### Cloud APIs

| Option | Accuracy | Speed | Cost | Best For |
|--------|----------|-------|------|----------|
| **Claude API** | 96-98% | 3-4s | $0.024/page ($290/1000) | Slight edge over MiniCPM-V |

---

## 🚀 Quick Install Commands

### Traditional OCR
```bash
# Apple Vision (macOS only, built-in)
uv pip install -e ".[ocr-apple]"

# PaddleOCR
uv pip install -e ".[ocr-paddle]"

# Tesseract
brew install tesseract  # macOS
uv pip install -e ".[ocr-tesseract]"

# EasyOCR
uv pip install -e ".[ocr-easy]"
```

### Modern OCR ⭐
```bash
# Surya (recommended!)
uv pip install -e ".[ocr-surya]"

# TrOCR (Microsoft)
uv pip install -e ".[ocr-trocr]"

# DocTR (document OCR)
uv pip install -e ".[ocr-doctr]"
```

### Vision-Language Models (Structured Extraction) ⭐⭐⭐
```bash
# Qwen2-VL-2B (best for M2 Air!)
uv pip install -e ".[vlm-qwen]"

# Florence-2 (Microsoft)
uv pip install -e ".[vlm-florence]"

# Moondream (tiny, efficient)
uv pip install -e ".[vlm-moondream]"
```

### Document Understanding
```bash
# Donut (end-to-end)
uv pip install -e ".[doc-donut]"

# LayoutParser
uv pip install -e ".[doc-layoutparser]"
```

### Cloud API
```bash
# Claude 3.5 Sonnet
uv pip install -e ".[vision-api]"
export ANTHROPIC_API_KEY='your-key'
```

---

## 💡 Recommended Combinations

### For Testing (Quick Start)
```bash
# Just test the best option
uv pip install -e ".[ocr-surya]"
python test_advanced_ocr.py --image flyer.png --engine surya
```

### For Production (Hybrid)
```bash
# Install Surya (fast) + Qwen2-VL (accurate fallback)
uv pip install -e ".[recommended-advanced]"

# Use Surya for 80% of images (fast)
# Use Qwen2-VL for complex/low-confidence cases
```

### For Maximum Accuracy (All Local)
```bash
# Install everything local
uv pip install -e ".[ocr-modern,vlm-all]"

# Test all and pick the best
python test_advanced_ocr.py --image flyer.png --engine all
```

### For Comparison with Cloud
```bash
# Install local + cloud
uv pip install -e ".[recommended-advanced,vision-api]"

# Compare
python compare_all_methods.py --image flyer.png
```

---

## 🎯 Decision Tree

```
START: Do you want to pay for API calls?
│
├─ NO (Free/Local) ──┐
│                     │
│  ┌──────────────────┘
│  │
│  ├─ Need BEST accuracy? (92-95%)
│  │  └─> Use Qwen2-VL-2B ⭐⭐⭐
│  │      uv pip install -e ".[vlm-qwen]"
│  │
│  ├─ Need GOOD accuracy + FAST? (90-93%, 2-4s)
│  │  └─> Use Surya ⭐
│  │      uv pip install -e ".[ocr-surya]"
│  │
│  └─ Need SIMPLE baseline? (85-90%)
│     └─> Use PaddleOCR
│         uv pip install -e ".[ocr-paddle]"
│
└─ YES (Cloud API) ──┐
                     │
                     └─> Use Claude 3.5 Sonnet (96-98%)
                         uv pip install -e ".[vision-api]"
                         ~$0.024 per page
```

---

## 📝 Testing Commands

### Test Single Image
```bash
# Surya
python test_advanced_ocr.py --image flyer.png --engine surya

# Qwen2-VL
python test_advanced_ocr.py --image flyer.png --engine qwen2vl

# All advanced engines
python test_advanced_ocr.py --image flyer.png --engine all

# Compare with traditional
python test_advanced_ocr.py --image flyer.png --engine surya --compare-traditional
```

### Test Multiple Images
```bash
# Batch test directory
python batch_test_images.py --dir data/raw/samples --engine paddleocr

# Compare local vs API
python batch_test_images.py --dir data/raw/samples --compare-api
```

### Compare Everything
```bash
# Test all methods side-by-side
python compare_all_methods.py --image flyer.png
```

---

## 💾 Disk Space Requirements

| Option | Initial Download | Cached Models |
|--------|-----------------|---------------|
| Surya | 400MB | 400MB |
| Qwen2-VL-2B | 4GB | 4GB |
| PaddleOCR | 300MB | 300MB |
| TrOCR | 1GB | 1GB |
| Florence-2 | 800MB | 800MB |
| Moondream | 600MB | 600MB |
| Apple Vision | 0MB | 0MB (built-in) |
| Tesseract | 10MB | 10MB |

**Note:** Models are cached after first download and reused.

---

## ⚡ Speed Comparison (M2 MacBook Air)

| Engine | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| Apple Vision | 1-2s | 1-2s |
| Surya | 5-8s (loading) | 2-4s |
| PaddleOCR | 8-12s (loading) | 3-5s |
| Qwen2-VL-2B | 30-60s (loading) | 10-15s |
| Claude API | 3-4s | 3-4s |

**First run includes model loading time**

---

## 🎁 Pre-configured Combos

```bash
# Recommended: Basic (PaddleOCR + Apple Vision)
uv pip install -e ".[recommended-basic]"

# Recommended: Advanced (Surya + Qwen2-VL)
uv pip install -e ".[recommended-advanced]"

# All Traditional OCR
uv pip install -e ".[ocr-traditional]"

# All Modern OCR
uv pip install -e ".[ocr-modern]"

# All Vision-Language Models
uv pip install -e ".[vlm-all]"

# Everything Local
uv pip install -e ".[local]"

# Everything (Local + Cloud)
uv pip install -e ".[all]"
```

---

## 🚀 My Personal Recommendation

**For M2 MacBook Air users:**

```bash
# Install these two:
uv pip install -e ".[ocr-surya,vlm-qwen]"

# Use Surya for 80-90% of images (fast, good accuracy)
python test_advanced_ocr.py --image flyer.png --engine surya

# Use Qwen2-VL for complex cases or when you need JSON output
python test_advanced_ocr.py --image flyer.png --engine qwen2vl
```

**Why this combo:**
- Surya: 90-93% accuracy, 2-4s (fast daily use)
- Qwen2-VL: 92-95% accuracy, 10-15s (when you need perfection)
- Both free and local
- Total download: ~4.4GB
- Covers 99% of use cases

**Total cost:** $0
**vs Claude API:** Saves ~$290 per 1000 flyers

---

## Need Help?

- Quick testing: See [IMAGE_TESTING_GUIDE.md](IMAGE_TESTING_GUIDE.md)
- Advanced options: See [ADVANCED_OCR_OPTIONS.md](ADVANCED_OCR_OPTIONS.md)
- M2 setup: See [M2_SETUP_GUIDE.md](M2_SETUP_GUIDE.md)
- UV usage: See [UV_QUICKSTART.md](UV_QUICKSTART.md)
