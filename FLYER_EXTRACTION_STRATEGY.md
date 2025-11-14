# Flyer Data Extraction Strategy - Expert Recommendations

## Problem Statement
Extract structured data (items + prices) from retail flyers in various formats (PDF, PNG, JPEG, online) with requirements:
- **Reliable**: High accuracy (>95% for prices)
- **Robust**: Handle various layouts, fonts, qualities
- **Inexpensive**: Minimize operational costs

---

## Recommended Architecture: Hybrid Multi-Tier Approach

### Tier 1: Traditional OCR + Rule-Based Extraction (Cheapest)
**Cost**: ~$0.0001 per flyer | **Accuracy**: 70-85%

```
Image → Tesseract OCR → Layout Analysis → Regex Price Extraction → Structured Output
```

**Use for**: Clean, standard layouts (grocery flyers, simple retail)

**Stack**:
- Tesseract 5.x with LSTM models
- OpenCV for preprocessing (deskew, denoise, binarization)
- Regex for price patterns ($X.XX, X.XX, XX¢)
- Spatial analysis for item-price pairing

**Pros**: Nearly free, fast, runs locally, no API dependencies
**Cons**: Brittle with complex layouts, poor with low-quality images

### Tier 2: Layout-Aware OCR + Local LLM (Moderate Cost)
**Cost**: ~$0.001-0.005 per flyer | **Accuracy**: 85-92%

```
Image → PaddleOCR/EasyOCR → Bounding Boxes → Local Qwen/Phi-3 → JSON Output
```

**Use for**: Moderate complexity, when Tier 1 confidence is low

**Stack**:
- PaddleOCR or EasyOCR (better layout detection than Tesseract)
- Local small LLM (Phi-3, Qwen2-1.5B, Llama-3.2-1B)
- Structured output with JSON schema

**Pros**: Better layout understanding, still mostly local, reasonable accuracy
**Cons**: Requires GPU for LLM, more complex setup

### Tier 3: Vision-Language Models (Most Reliable)
**Cost**: ~$0.01-0.05 per flyer | **Accuracy**: 95-98%

```
Image → Claude 3.5 Sonnet / GPT-4o / Gemini Pro Vision → Structured JSON
```

**Use for**: Complex layouts, multiple columns, poor quality, when accuracy is critical

**Stack**:
- Claude 3.5 Sonnet (best price/performance for documents)
- GPT-4o (fast, good vision)
- Gemini 1.5 Flash (cheapest vision model)

**Pros**: Highest accuracy, handles any layout, understands context
**Cons**: API costs, requires internet, latency

---

## Cost Comparison (per 1000 flyers)

| Approach | Cost per Flyer | Total Cost | Accuracy | Processing Time |
|----------|---------------|------------|----------|-----------------|
| Tier 1: Tesseract + Regex | $0.0001 | **$0.10** | 70-85% | 2-5s |
| Tier 2: PaddleOCR + Local LLM | $0.003 | **$3.00** | 85-92% | 5-10s |
| Tier 3: Claude 3.5 Sonnet | $0.025 | **$25.00** | 95-98% | 3-6s |
| Tier 3: GPT-4o | $0.015 | **$15.00** | 93-96% | 2-4s |
| Tier 3: Gemini Flash | $0.008 | **$8.00** | 90-94% | 3-5s |

---

## Recommended POC Implementation Strategy

### Phase 1: Build Tier 1 (Week 1-2)
**Goal**: Get something working quickly at minimal cost

1. **Enhance OCR preprocessing**:
   ```python
   - Adaptive thresholding
   - Deskewing/rotation correction
   - Contrast enhancement
   - Noise removal
   ```

2. **Add layout analysis**:
   ```python
   - Detect text regions with contours
   - Identify columns/sections
   - Build spatial index
   ```

3. **Implement price extraction**:
   ```python
   - Regex patterns: r'\$?\d+\.\d{2}', r'\d+¢', r'\d+\sfor\s\$\d+'
   - Validate price ranges ($0.10 - $999.99)
   - Filter false positives
   ```

4. **Item-price pairing**:
   ```python
   - Spatial proximity (nearest neighbor)
   - Line/column alignment
   - Heuristic rules
   ```

**Expected Result**: 70-80% accuracy on clean flyers, very low cost

### Phase 2: Add Tier 2 for Fallback (Week 3-4)
**Goal**: Improve accuracy for medium complexity

1. **Integrate PaddleOCR or EasyOCR**:
   - Better bounding box detection
   - Multi-language support
   - Better with complex layouts

2. **Add local LLM for structured extraction**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Use small, efficient models
   model = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B params
   # OR "Qwen/Qwen2-1.5B-Instruct"  # 1.5B params

   prompt = f"""
   Extract items and prices from this OCR text:
   {ocr_text}

   Return JSON format:
   {{"items": [{{"name": "...", "price": "...", "unit": "..."}}, ...]}}
   """
   ```

3. **Implement confidence scoring**:
   - OCR confidence from PaddleOCR
   - Price pattern validation
   - LLM response validation
   - If confidence < 0.85, escalate to Tier 3

**Expected Result**: 85-90% accuracy, still mostly local processing

### Phase 3: Add Tier 3 for Complex Cases (Week 5)
**Goal**: Achieve 95%+ accuracy when needed

1. **Integrate Vision API** (Claude 3.5 Sonnet recommended):
   ```python
   import anthropic
   import base64

   def extract_with_vision(image_path):
       client = anthropic.Anthropic(api_key="...")

       with open(image_path, "rb") as f:
           image_data = base64.b64encode(f.read()).decode()

       message = client.messages.create(
           model="claude-3-5-sonnet-20241022",
           max_tokens=2000,
           messages=[{
               "role": "user",
               "content": [
                   {
                       "type": "image",
                       "source": {
                           "type": "base64",
                           "media_type": "image/png",
                           "data": image_data
                       }
                   },
                   {
                       "type": "text",
                       "text": """Extract ALL items and prices from this retail flyer.

                       Return valid JSON only, no markdown:
                       {
                         "items": [
                           {
                             "name": "Product name",
                             "price": "X.XX",
                             "unit": "each/lb/kg/etc",
                             "promotion": "Buy 2 get 1 free (if applicable)",
                             "confidence": 0.0-1.0
                           }
                         ]
                       }

                       Rules:
                       - Include prices in decimal format (e.g., 5.99 not $5.99)
                       - Set confidence based on text clarity
                       - Include promotional details if visible
                       """
                   }
               ]
           }]
       )

       return message.content[0].text
   ```

2. **Smart routing logic**:
   ```python
   def process_flyer(image_path):
       # Try Tier 1 first
       result_t1 = tier1_extract(image_path)

       if result_t1['confidence'] > 0.85:
           return result_t1  # Good enough!

       # Escalate to Tier 2
       result_t2 = tier2_extract(image_path)

       if result_t2['confidence'] > 0.90:
           return result_t2

       # Escalate to Tier 3 for complex cases
       result_t3 = tier3_vision_extract(image_path)
       return result_t3
   ```

**Expected Result**: 95-98% accuracy, optimized costs

---

## Best Practices for Cost Optimization

### 1. Caching & Deduplication
```python
# Hash images to avoid reprocessing
import hashlib

def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# Cache results
cache = {}  # Or use Redis/SQLite
image_hash = get_image_hash(image_path)
if image_hash in cache:
    return cache[image_hash]
```

### 2. Batch Processing
- For Tier 1/2: Process multiple flyers in parallel
- For Tier 3: Batch API calls (some providers offer discounts)

### 3. Smart Image Preprocessing
```python
# Reduce image size before API calls (saves cost)
from PIL import Image

def optimize_for_api(image_path, max_dimension=1568):
    img = Image.open(image_path)

    # Resize if too large
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save with optimization
    img.save('optimized.jpg', 'JPEG', quality=85, optimize=True)
    return 'optimized.jpg'
```

### 4. Selective Processing
- Use Tier 1 for 70% of flyers (standard layouts)
- Use Tier 2 for 20% (moderate complexity)
- Use Tier 3 for 10% (complex/critical)
- **Average cost**: ~$0.005 per flyer

---

## Specific Model Recommendations

### For Tier 3 (Vision APIs) - Ranked by Value

1. **Claude 3.5 Sonnet** ⭐ BEST CHOICE
   - Cost: ~$0.025 per flyer (assuming 1-2 images per flyer)
   - Accuracy: 96-98%
   - Best at understanding document structure
   - Excellent JSON output reliability
   - Great with promotional text

2. **Gemini 1.5 Flash** 💰 BEST BUDGET
   - Cost: ~$0.008 per flyer
   - Accuracy: 90-94%
   - Fast processing
   - Good for high-volume POC

3. **GPT-4o**
   - Cost: ~$0.015 per flyer
   - Accuracy: 93-96%
   - Fast, good vision capabilities
   - Can struggle with dense layouts

4. **Qwen2-VL** (Open Source) 🔓 BEST LOCAL
   - Cost: $0 (local GPU required)
   - Accuracy: 85-90%
   - 7B or 72B variants
   - Best open-source vision-language model

---

## Data Fetching Strategy

### For Online Flyers

```python
import requests
from bs4 import BeautifulSoup

def fetch_flyer_from_url(url):
    """Fetch flyer image from retailer website"""

    # Option 1: Direct image URL
    if url.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        response = requests.get(url)
        return response.content

    # Option 2: Scrape from webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find flyer images (customize selectors per retailer)
    flyer_imgs = soup.select('img.flyer-image')
    # or soup.find_all('img', class_='flyer')

    # Download images
    images = []
    for img in flyer_imgs:
        img_url = img.get('src') or img.get('data-src')
        if img_url:
            img_response = requests.get(img_url)
            images.append(img_response.content)

    return images

# For major retailers, consider using APIs:
# - Flipp API (aggregates Canadian/US flyers)
# - Reebee API
# - RetailMeNot API
```

### For PDFs

```python
import pymupdf

def extract_pages_from_pdf(pdf_path):
    """Convert PDF to high-quality images"""
    doc = pymupdf.open(pdf_path)
    images = []

    for page_num, page in enumerate(doc):
        # High DPI for better OCR
        pix = page.get_pixmap(dpi=300)
        img_path = f'page_{page_num}.png'
        pix.save(img_path)
        images.append(img_path)

    return images
```

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Enhance OCR preprocessing pipeline
- [ ] Implement price regex extraction
- [ ] Build basic item-price pairing
- [ ] Create validation rules

### Week 2: Tier 1 Complete
- [ ] Layout analysis (column detection)
- [ ] Spatial indexing for pairing
- [ ] Confidence scoring
- [ ] Test on 50-100 sample flyers

### Week 3: Add Tier 2
- [ ] Integrate PaddleOCR
- [ ] Set up local LLM (Phi-3 or Qwen2)
- [ ] Implement structured extraction
- [ ] Build routing logic

### Week 4: Polish Tier 2
- [ ] Fine-tune confidence thresholds
- [ ] Optimize LLM prompts
- [ ] Add error handling
- [ ] Performance testing

### Week 5: Add Tier 3
- [ ] Integrate Claude API
- [ ] Implement vision extraction
- [ ] Build complete routing system
- [ ] Cost tracking and optimization

### Week 6: Production Ready
- [ ] Caching layer
- [ ] Batch processing
- [ ] API endpoint
- [ ] Documentation

---

## Expected Final Results

### Accuracy
- **Overall**: 95-98% (item-price pairs correctly extracted)
- **Price accuracy**: 98-99% (critical for business)
- **Item name accuracy**: 92-96% (some ambiguity acceptable)

### Cost (per 1000 flyers with smart routing)
- 70% via Tier 1: $0.07
- 20% via Tier 2: $0.60
- 10% via Tier 3: $2.50
- **Total**: ~$3.17 per 1000 flyers = **$0.003 per flyer**

### Performance
- **Tier 1**: 2-5 seconds
- **Tier 2**: 5-10 seconds
- **Tier 3**: 3-6 seconds
- **Average**: ~4 seconds per flyer

---

## Conclusion

Your current repository provides a solid foundation (PDF conversion, basic OCR), but you need to add:

1. **Layout analysis and spatial reasoning**
2. **Structured data extraction**
3. **Smart multi-tier routing**
4. **Vision API integration for complex cases**

The hybrid approach balances cost, accuracy, and robustness. Start with Tier 1 to prove the concept, then add Tiers 2 and 3 to achieve production-grade reliability.

**Recommended first step**: Build Tier 1 with improved OCR preprocessing and price extraction. This will give you immediate results and help you understand where the hard cases are before investing in more expensive solutions.
