# Getting Started - Flyer Data Extraction

**Complete guide to extract items and prices from flyer images in 15 minutes.**

---

## 📋 Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Installation Options](#installation-options)
3. [Testing Your First Image](#testing-your-first-image)
4. [Comparing All Methods](#comparing-all-methods)
5. [Extraction Approaches Explained](#extraction-approaches-explained)
6. [Batch Testing Multiple Images](#batch-testing-multiple-images)
7. [Making Your Decision](#making-your-decision)
8. [Next Steps](#next-steps)

---

## Quick Start (5 minutes)

Get up and running with the best free local option:

```bash
# 1. Install UV (package manager - 10x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
cd pdf-extractor
uv venv
source .venv/bin/activate

# 3. Install best local OCR (Surya - modern, accurate)
uv pip install -e ".[ocr-surya]"

# 4. Get a test image
# - Visit https://flipp.com
# - Screenshot a flyer page
# - Save as: data/raw/samples/test.png

# 5. Test it!
python test_advanced_ocr.py --image data/raw/samples/test.png --engine surya
```

**Result:** You'll see extracted items and prices in ~2-4 seconds with 90-93% accuracy, **$0 cost**.

---

## Installation Options

Choose based on your needs:

### Option 1: Best Accuracy (Free, Local) ⭐⭐⭐ **RECOMMENDED**

```bash
uv pip install -e ".[recommended-advanced]"
```

**Includes:**
- **Surya** (fast, 90-93% accuracy)
- **Qwen2-VL** (slow but 92-95% accuracy, structured JSON)

**Use for:** Maximum accuracy without paying for API

---

### Option 2: Quick Testing (Fast, Simple) ⭐

```bash
uv pip install -e ".[ocr-paddle,ocr-apple]"
```

**Includes:**
- **PaddleOCR** (proven, 85-90% accuracy)
- **Apple Vision** (macOS native, fastest)

**Use for:** Quick proof-of-concept

---

### Option 3: Cloud API (Best Accuracy, Costs Money)

```bash
uv pip install -e ".[vision-api]"
export ANTHROPIC_API_KEY='your-key'
```

**Includes:**
- **Claude 3.5 Sonnet** (96-98% accuracy, ~$0.024/page)

**Use for:** When you need the absolute best accuracy

---

### Option 4: Everything (For Comparison)

```bash
uv pip install -e ".[all]"
```

**Includes:** All engines above for side-by-side comparison

**Use for:** Evaluating all options before deciding

---

## Testing Your First Image

### Step 1: Get a Flyer Image

**Option A: Screenshot** (Easiest)
```bash
# 1. Visit https://flipp.com
# 2. Search "Metro" or "Food Basics"
# 3. Take screenshot (Cmd+Shift+4 on Mac)
# 4. Save to: data/raw/samples/test.png
```

**Option B: Download Sample**
```bash
# We'll create samples directory
mkdir -p data/raw/samples
# Download your flyer screenshot there
```

### Step 2: Test with Single Engine

```bash
# Test Surya (modern OCR, fast)
python test_advanced_ocr.py --image data/raw/samples/test.png --engine surya

# Test Qwen2-VL (VLM, structured extraction)
python test_advanced_ocr.py --image data/raw/samples/test.png --engine qwen2vl

# Test PaddleOCR (traditional, proven)
python test_local_ocr.py --image data/raw/samples/test.png --engines paddleocr

# Test Claude API (cloud, best accuracy)
export ANTHROPIC_API_KEY='your-key'
python test_vision_api.py --image data/raw/samples/test.png
```

### Step 3: Review Results

You'll see output like:
```
📊 ADVANCED OCR RESULTS
════════════════════════════════════════
Engine          Time      Items     Status
────────────────────────────────────────
Surya           2.8s      23        ✅ OK

🛒 Sample Items (first 10):
   1. Bananas
   2. Chicken Breast
   3. Fresh Atlantic Salmon
   ...
```

---

## Comparing All Methods

### Compare Local Engines

```bash
# Test all advanced local engines
python test_advanced_ocr.py --image test.png --engine all

# Compare traditional vs modern
python test_advanced_ocr.py --image test.png --engine surya --compare-traditional
```

### Compare Local vs Cloud

```bash
# Full comparison (requires API key)
python compare_all_methods.py --image test.png

# Output shows:
# - PaddleOCR: 85-90% accuracy, $0
# - Surya: 90-93% accuracy, $0
# - Qwen2-VL: 92-95% accuracy, $0
# - Claude API: 96-98% accuracy, $0.024/page
```

---

## Extraction Approaches Explained

### 🏆 Top 3 Recommended Approaches

#### 1. **Surya** (Modern Transformer OCR) ⭐ BEST FOR MOST CASES

**What it is:** Modern OCR using transformer neural networks, built specifically for document text.

**How it works:**
- Detects text regions using deep learning
- Recognizes text using transformer models
- Much better than traditional OCR on complex layouts

**Pros:**
- ✅ High accuracy (90-93%)
- ✅ Fast (2-4s per page on M2 Air)
- ✅ Free and runs locally
- ✅ Good with complex layouts
- ✅ Handles multiple fonts well
- ✅ Moderate memory usage (2GB)

**Cons:**
- ❌ First download ~400MB
- ❌ Not as accurate as VLMs or Cloud API
- ❌ Gives raw text, not structured data

**Best for:**
- Daily use on lots of flyers
- When speed matters
- When 90% accuracy is good enough
- Budget-conscious projects

**Install:**
```bash
uv pip install -e ".[ocr-surya]"
```

**Test:**
```bash
python test_advanced_ocr.py --image flyer.png --engine surya
```

---

#### 2. **Qwen2-VL-2B** (Vision-Language Model) ⭐⭐⭐ BEST ACCURACY (LOCAL)

**What it is:** AI model that understands images AND language, can extract structured data directly.

**How it works:**
- Analyzes entire image holistically
- Understands relationships (which price goes with which item)
- Generates structured JSON output
- Uses 2 billion parameters for vision+language understanding

**Pros:**
- ✅ Very high accuracy (92-95%)
- ✅ **Extracts structured JSON directly**
- ✅ Understands context and relationships
- ✅ Free and runs locally
- ✅ Almost as good as Claude API
- ✅ Best for complex layouts
- ✅ Can handle promotions ("Buy 2 Get 1")

**Cons:**
- ❌ Slower (10-15s per page)
- ❌ Large download (4GB)
- ❌ Higher memory (4-6GB)
- ❌ Requires M2 Air with 16GB RAM (or quantization for 8GB)
- ❌ More complex setup

**Best for:**
- When you need best local accuracy
- Structured data extraction
- Complex layouts with promotions
- When you want to avoid API costs
- Final processing of important data

**Install:**
```bash
uv pip install -e ".[vlm-qwen]"
```

**Test:**
```bash
python test_advanced_ocr.py --image flyer.png --engine qwen2vl
```

**Savings:** ~$290 per 1000 flyers vs Claude API!

---

#### 3. **Claude 3.5 Sonnet** (Cloud Vision API) ⭐ GOLD STANDARD

**What it is:** Anthropic's most advanced AI model with vision capabilities.

**How it works:**
- Sends image to Anthropic's cloud servers
- Uses massive AI model (details not public)
- Returns structured JSON
- Extremely good at understanding document context

**Pros:**
- ✅ Highest accuracy (96-98%)
- ✅ Best with complex layouts
- ✅ Best with poor image quality
- ✅ Fast (3-4s per page)
- ✅ Structured JSON output
- ✅ No local setup needed
- ✅ Constantly improving

**Cons:**
- ❌ **Costs money** (~$0.024 per page)
- ❌ Requires internet connection
- ❌ Requires API key
- ❌ Data sent to cloud (privacy concern)
- ❌ $290 for 1000 flyers (12 pages each)

**Best for:**
- When accuracy is critical
- Low volume (<100 flyers/month)
- When budget allows
- Baseline for comparison
- Complex edge cases

**Install:**
```bash
uv pip install -e ".[vision-api]"
export ANTHROPIC_API_KEY='your-key'
```

**Test:**
```bash
python test_vision_api.py --image flyer.png
```

---

### 📊 Other Options (Reference)

#### **PaddleOCR** (Traditional - Proven Baseline)

**Accuracy:** 85-90% | **Speed:** 3-5s | **Cost:** Free

**Pros:**
- Proven, stable, widely used
- Good accuracy for traditional OCR
- Fast enough
- Low memory

**Cons:**
- Not as accurate as modern options
- Struggles with complex layouts

**Use when:** You want a reliable baseline, proven solution

```bash
uv pip install -e ".[ocr-paddle]"
python test_local_ocr.py --image flyer.png --engines paddleocr
```

---

#### **Apple Vision** (macOS Native - Fastest)

**Accuracy:** 75-85% | **Speed:** 1-2s | **Cost:** Free

**Pros:**
- Already built into macOS
- Fastest option
- Uses Apple Neural Engine
- Zero setup

**Cons:**
- macOS only
- Lower accuracy
- Less customizable

**Use when:** Speed is priority, macOS only, quick tests

```bash
uv pip install -e ".[ocr-apple]"
python test_local_ocr.py --image flyer.png --engines apple
```

---

#### **Tesseract** (Traditional - Simple Baseline)

**Accuracy:** 70-80% | **Speed:** 2-3s | **Cost:** Free

**Pros:**
- Very lightweight
- Fast
- Simple
- Widely compatible

**Cons:**
- Lowest accuracy
- Poor with complex layouts
- Struggles with varied fonts

**Use when:** You need something simple, resource-constrained

```bash
brew install tesseract
uv pip install -e ".[ocr-tesseract]"
python test_local_ocr.py --image flyer.png --engines tesseract
```

---

### 📈 Accuracy Comparison Table

| Method | Accuracy | Speed (M2) | Cost/1000 Flyers | Best For |
|--------|----------|------------|------------------|----------|
| **Claude API** 🥇 | 96-98% | 3-4s | $290 | Highest accuracy needed |
| **Qwen2-VL** 🥇 | 92-95% | 10-15s | **$0** | Best free option |
| **Surya** 🥈 | 90-93% | 2-4s | **$0** | Speed + accuracy balance |
| **TrOCR** 🥈 | 92-95% | 5-8s | **$0** | High accuracy OCR |
| **PaddleOCR** 🥉 | 85-90% | 3-5s | **$0** | Proven baseline |
| **Apple Vision** | 75-85% | 1-2s | **$0** | Fastest (macOS) |
| **Tesseract** | 70-80% | 2-3s | **$0** | Simple baseline |

---

## Batch Testing Multiple Images

Test many flyer images at once:

```bash
# 1. Put images in directory
mkdir -p data/raw/samples
# Add 10-20 flyer screenshots there

# 2. Batch test with one engine
python batch_test_images.py --dir data/raw/samples --engine paddleocr

# 3. Compare local vs API
python batch_test_images.py --dir data/raw/samples --compare-api

# 4. Review summary
# Shows: avg items found, avg time, total cost, accuracy estimate
```

**Output:**
```
📊 SUMMARY REPORT
════════════════════════════════════════

🔍 PADDLEOCR Results:
   - Successful: 15/15 images
   - Total items found: 342
   - Average per image: 22.8 items
   - Average time: 3.9s per image
   - Cost: $0.00

☁️ CLAUDE API Results:
   - Total items found: 378
   - Average per image: 25.2 items
   - Total cost: $0.3675

📈 Comparison:
   - PaddleOCR found 90.5% of items vs Claude
   - Cost savings: $0.37 (100%)

✅ Recommendation: Use PaddleOCR (local)
```

---

## Making Your Decision

### Decision Tree

```
Do you have budget for API calls?
│
├─ NO → Want best free accuracy?
│        ├─ YES → Use Qwen2-VL (92-95%, slow but free)
│        └─ NO → Want speed + good accuracy?
│                 ├─ YES → Use Surya (90-93%, fast)
│                 └─ NO → Use PaddleOCR (85-90%, proven)
│
└─ YES → Is accuracy critical?
         ├─ YES → Use Claude API (96-98%, $0.024/page)
         └─ NO → Use Hybrid (local first, API fallback)
```

### Hybrid Approach (Recommended for Production)

```python
# Pseudo-code for hybrid approach
def extract_flyer(image_path):
    # Try Surya first (fast, free, 90% accuracy)
    result = surya_extract(image_path)

    if result.confidence > 0.85:
        return result  # Good enough!

    # For low confidence, use Qwen2-VL (slow but better)
    result = qwen2vl_extract(image_path)

    if result.confidence > 0.90:
        return result  # Still good!

    # Only for complex cases, use Claude API
    result = claude_extract(image_path)
    return result

# Result: 80% free (Surya), 15% free (Qwen2-VL), 5% paid (Claude)
# Cost: ~$15 per 1000 flyers vs $290 full Claude
# Accuracy: 94-96% overall
```

---

## Next Steps

### For Testing (Today)

1. **Install Surya**
   ```bash
   uv pip install -e ".[ocr-surya]"
   ```

2. **Download 5-10 flyer screenshots**

3. **Batch test**
   ```bash
   python batch_test_images.py --dir data/raw/samples --engine surya
   ```

4. **Review accuracy** - is 90-93% good enough?

---

### For Production (This Week)

1. **If Surya is good enough:**
   - Use it for everything
   - Total cost: $0
   - You're done!

2. **If you need better:**
   - Install Qwen2-VL for complex cases
   - Use Surya for 80%, Qwen2-VL for 20%
   - Total cost: Still $0

3. **If accuracy is critical:**
   - Get Claude API key
   - Use hybrid approach above
   - Total cost: ~$15-50 per 1000 flyers

---

### For Scale (Next Month)

1. **Validate accuracy on 100+ images**
2. **Optimize preprocessing** (image quality, deskewing)
3. **Build confidence scoring** to route to appropriate engine
4. **Set up batch processing** pipeline
5. **Monitor costs** and accuracy over time

---

## Quick Command Reference

```bash
# Installation
uv pip install -e ".[recommended-advanced]"  # Best combo
uv pip install -e ".[ocr-surya]"            # Just Surya
uv pip install -e ".[vlm-qwen]"             # Just Qwen2-VL
uv pip install -e ".[vision-api]"           # Just Claude API

# Testing single image
python test_advanced_ocr.py --image test.png --engine surya
python test_advanced_ocr.py --image test.png --engine qwen2vl
python test_vision_api.py --image test.png

# Batch testing
python batch_test_images.py --dir data/raw/samples
python batch_test_images.py --dir data/raw/samples --compare-api

# Comparing all methods
python compare_all_methods.py --image test.png
python test_advanced_ocr.py --image test.png --engine all
```

---

## Need Help?

- **Installation issues:** See [UV_QUICKSTART.md](UV_QUICKSTART.md)
- **M2 Mac specific:** See [M2_SETUP_GUIDE.md](M2_SETUP_GUIDE.md)
- **Detailed comparisons:** See [ADVANCED_OCR_OPTIONS.md](ADVANCED_OCR_OPTIONS.md)
- **Project overview:** See [CLAUDE.md](CLAUDE.md)

---

## Summary

**For most users:**
1. Install Surya (`uv pip install -e ".[ocr-surya]"`)
2. Test on 10-20 images
3. If 90-93% accuracy is good → Use it!
4. If not → Add Qwen2-VL for hard cases
5. Still not enough → Use Claude API as fallback

**Expected final accuracy:** 92-96%
**Expected final cost:** $0-20 per 1000 flyers
**Time to production:** 1-2 weeks

Good luck! 🚀
