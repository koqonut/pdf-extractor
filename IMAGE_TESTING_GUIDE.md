# Testing with Images Directly - Quick Guide

**Yes! You can test directly with images (PNG/JPG) - no PDF conversion needed!**

This is actually the **easiest and recommended** way to test.

---

## Quick Start: Download & Test

### Step 1: Get Sample Flyer Images (2 minutes)

**Option A: Manual Screenshots** (Easiest)

```bash
# 1. Visit Flipp
open https://flipp.com

# 2. Search for your favorite grocery store:
#    - Metro
#    - Food Basics
#    - No Frills
#    - Loblaws
#    - Walmart

# 3. Open a flyer and take screenshots
#    Mac: Cmd + Shift + 4
#    Windows: Win + Shift + S
#    Linux: PrtScn or Flameshot

# 4. Save to this directory
mkdir -p data/raw/samples
# Save your screenshots there
```

**Option B: Download Multiple Pages**

Use browser DevTools to download flyer images directly:

1. Visit Flipp, open a flyer
2. Open DevTools (F12)
3. Go to Network tab
4. Look for large image requests (typically PNG or JPG, 500KB-2MB)
5. Right-click → Open in new tab → Save image

**Option C: Use Our Download Helper**

```bash
python download_sample_flyer.py
```

---

### Step 2: Test Single Image (30 seconds)

```bash
# Setup (if not done)
uv venv
source .venv/bin/activate
uv pip install -e ".[pdf,ocr-paddle]"

# Test one image
python test_local_ocr.py --image data/raw/samples/metro_page1.png --engines paddleocr
```

**You'll get results like:**

```
📊 OCR ENGINE COMPARISON
═══════════════════════════════════════════
Engine          Time      Prices    Items     Status
───────────────────────────────────────────
PaddleOCR       3.5s      24        22        ✅ OK

💰 Prices Found: 24
🛒 Items Found: 22

1. Bananas → $0.69
2. Chicken Breast → $5.99
3. ...
```

---

### Step 3: Test Multiple Images (Batch Testing) ⭐

```bash
# Test all images in a directory
python batch_test_images.py --dir data/raw/samples

# Compare with Claude API
python batch_test_images.py --dir data/raw/samples --compare-api

# Use different engine
python batch_test_images.py --dir data/raw/samples --engine apple
```

**You'll get a summary like:**

```
📊 SUMMARY REPORT
═══════════════════════════════════════════

🔍 PADDLEOCR Results:
   - Successful: 10/10 images
   - Total items found: 234
   - Total prices found: 198
   - Average per image: 23.4 items, 19.8 prices
   - Average time: 3.8s per image
   - Total time: 38s
   - Cost: $0.00 (free!)

☁️  CLAUDE API Results:
   - Successful: 10/10 images
   - Total items found: 256
   - Average per image: 25.6 items
   - Average time: 3.2s per image
   - Total cost: $0.2450
   - Cost per image: $0.0245

📈 Comparison:
   - paddleocr found 91.4% of items vs Claude API
   - Cost savings: $0.25 (100% saved using local)

✅ Recommendation: Use paddleocr (local)
   Reason: Achieves 91% of API accuracy at $0 cost
```

---

## Supported Image Formats

✅ PNG (recommended - best quality)
✅ JPG/JPEG (good)
✅ WebP (good)
✅ Any format that PIL/OpenCV can read

**Recommended specs for best results:**
- **Resolution**: 1200px+ width
- **DPI**: 300+ (for scanned images)
- **Format**: PNG (lossless)
- **Quality**: High contrast, not blurry

---

## Testing Workflow Examples

### Example 1: Quick Test (One Image)

```bash
# Download one flyer page screenshot
# Save as: data/raw/samples/test.png

# Test with PaddleOCR (best local)
python test_local_ocr.py --image data/raw/samples/test.png --engines paddleocr

# Test with Claude API
export ANTHROPIC_API_KEY='your-key'
python test_vision_api.py --image data/raw/samples/test.png

# Compare both
python compare_all_methods.py --image data/raw/samples/test.png
```

**Time:** 2 minutes
**Cost:** Free (local) or ~$0.025 (with API)

---

### Example 2: Test 10 Pages from One Flyer

```bash
# 1. Screenshot 10 pages from a Metro flyer
# Save as: data/raw/samples/metro_page1.png, metro_page2.png, ...

# 2. Batch test all pages
python batch_test_images.py --dir data/raw/samples

# 3. Review summary
# See how many items/prices extracted per page
```

**Time:** 5-10 minutes
**Cost:** $0

---

### Example 3: Compare 3 Different Stores

```bash
# 1. Create directories
mkdir -p data/raw/samples/metro
mkdir -p data/raw/samples/nofrills
mkdir -p data/raw/samples/foodbasics

# 2. Download 5 pages from each store
# Save to respective directories

# 3. Test each store
python batch_test_images.py --dir data/raw/samples/metro --engine paddleocr
python batch_test_images.py --dir data/raw/samples/nofrills --engine paddleocr
python batch_test_images.py --dir data/raw/samples/foodbasics --engine paddleocr

# 4. Compare results
# See which store's flyers are easier to extract
```

**Time:** 15-20 minutes
**Cost:** $0
**Insight:** Which stores have the cleanest layouts for OCR

---

### Example 4: Validate Accuracy with Ground Truth

```bash
# 1. Take one flyer page screenshot
# Save as: data/raw/samples/validation.png

# 2. Manually count items and prices (ground truth)
# Write down: 28 items, 24 prices

# 3. Test extraction
python test_local_ocr.py --image data/raw/samples/validation.png --engines paddleocr

# 4. Compare results
# If it found 23/24 prices → 95.8% accuracy!
```

**Time:** 10 minutes
**Result:** Know exact accuracy for your use case

---

## Batch Testing Commands Reference

```bash
# Test all images in directory
python batch_test_images.py --dir data/raw/samples

# Test specific images
python batch_test_images.py --images img1.png img2.png img3.png

# Use different engine
python batch_test_images.py --dir data/raw/samples --engine apple

# Compare with Vision API
python batch_test_images.py --dir data/raw/samples --compare-api

# Save detailed results
python batch_test_images.py --dir data/raw/samples --output my_results.json

# Test everything
python batch_test_images.py --dir data/raw/samples --compare-api --engine paddleocr
```

---

## What to Look For in Results

### Good Results (Ready for Production)

✅ **Accuracy**: 85%+ of items found compared to manual count
✅ **Prices**: 95%+ of prices extracted correctly
✅ **Speed**: Under 5 seconds per image
✅ **Consistency**: Similar results across different flyer pages

### Needs Improvement

⚠️ **Accuracy**: Below 80% of items
⚠️ **Missing prices**: Less than 90% extracted
⚠️ **Inconsistent**: Wild variation between pages

**Solutions:**
- Try different OCR engine (test all 4)
- Improve image quality (higher resolution)
- Preprocess images (denoise, deskew)
- Use Vision API for hard cases
- Implement hybrid approach

---

## Directory Structure Recommendation

```bash
data/raw/samples/
├── metro/
│   ├── 2024-01-11_page1.png
│   ├── 2024-01-11_page2.png
│   └── ...
├── nofrills/
│   ├── 2024-01-11_page1.png
│   └── ...
├── foodbasics/
│   └── ...
└── test/
    ├── validation_page1.png  # For accuracy testing
    └── validation_page2.png
```

This makes it easy to:
- Test by store
- Track by date
- Organize test vs real data

---

## Tips for Best Results

### Image Quality

✅ **DO:**
- Use PNG format (lossless)
- Screenshot at high resolution (1920x1080+)
- Ensure good lighting/contrast
- Keep text horizontal (not rotated)
- Capture full page

❌ **DON'T:**
- Use heavily compressed JPG
- Screenshot at low resolution (<1200px)
- Include blurry or pixelated text
- Capture at weird angles

### File Naming

✅ **Good:**
- `metro_2024-01-11_page01.png`
- `foodbasics_weekly_p1.png`
- `nofrills_jan_flyer_page_01.png`

❌ **Avoid:**
- `Screen Shot 2024-01-11 at 3.45.32 PM.png`
- `image.png`
- `flyer.jpg`

### Testing Strategy

1. **Start small**: Test 1-3 images first
2. **Validate accuracy**: Manually count items on 1-2 images
3. **Scale up**: Test 10-20 images once confident
4. **Compare engines**: Try all 4 OCR engines on same images
5. **Document results**: Save reports for reference

---

## Example: Real Testing Session

```bash
# Day 1: Get samples (10 minutes)
mkdir -p data/raw/samples
# Screenshot 5 pages from Metro flyer
# Save as metro_p1.png through metro_p5.png

# Day 1: Quick test (5 minutes)
uv venv
source .venv/bin/activate
uv pip install -e ".[ocr-paddle]"
python test_local_ocr.py --image data/raw/samples/metro_p1.png --engines paddleocr

# Result: 22 items, 18 prices found in 3.5s
# Manual count: 24 items, 19 prices
# Accuracy: 91.7% items, 94.7% prices ✅

# Day 1: Batch test all (2 minutes)
python batch_test_images.py --dir data/raw/samples

# Result: Average 21.4 items per page, 17.8 prices
# Total time: 17.5s for 5 pages
# Decision: Good enough for POC! ✅

# Day 2: Test more stores (20 minutes)
# Download samples from No Frills and Food Basics
# Repeat testing
# Compare which store is easiest to extract

# Result: Metro (91%), Food Basics (89%), No Frills (78%)
# Decision: Start with Metro and Food Basics
```

---

## Troubleshooting

### Issue: No images found

```bash
# Check directory
ls -la data/raw/samples/

# Supported extensions
*.png, *.jpg, *.jpeg, *.PNG, *.JPG, *.JPEG, *.webp
```

### Issue: Poor extraction results

```bash
# Try different engine
python test_local_ocr.py --image your_image.png --engines apple

# Compare with Vision API to see max possible
python test_vision_api.py --image your_image.png
```

### Issue: Images too large

```bash
# Resize before testing
from PIL import Image
img = Image.open('large.png')
img.thumbnail((2000, 2000), Image.LANCZOS)
img.save('resized.png')
```

---

## Next Steps

1. **Get images**: Download 5-10 flyer screenshots
2. **Run batch test**: `python batch_test_images.py --dir data/raw/samples`
3. **Review results**: Check accuracy and speed
4. **Decide on approach**:
   - If >85% accuracy → Use local OCR (free!)
   - If 75-85% → Use hybrid (local + API fallback)
   - If <75% → Use Vision API or improve images

**Ready to start? Just run:**

```bash
mkdir -p data/raw/samples
# Add your screenshots there
python batch_test_images.py --dir data/raw/samples
```

Good luck! 🚀
