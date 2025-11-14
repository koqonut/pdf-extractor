# Canadian Grocery Flyer Sources - Analysis & Strategies

## Target Retailers

Based on your links, you're targeting major Canadian grocery chains:
- **Metro** - Mid-tier grocery chain
- **No Frills** - Discount grocery (owned by Loblaw)
- **Food Basics** - Discount grocery (owned by Metro Inc.)

## Flipp Platform Analysis

**Flipp.com** is a flyer aggregation platform that hosts digital flyers from 800+ retailers across North America.

### Key Characteristics (Based on Industry Knowledge)

1. **Format**: Interactive HTML5 viewer with high-resolution images
2. **Structure**: Each page is typically served as:
   - High-res JPEG/PNG images (1200-2000px width)
   - Overlaid with clickable hotspot data (items, prices, positions)
   - JSON data containing structured product information
3. **Pages**: Typical grocery flyer = 8-16 pages
4. **Quality**: Very high (300+ DPI equivalent)
5. **Accessibility**:
   - Web viewer (interactive)
   - PDF download option (sometimes)
   - Mobile apps (iOS/Android)

### Anti-Scraping Measures (Why 403 Errors)

All these sites implement:
- User-Agent filtering
- Rate limiting
- JavaScript rendering requirements
- CAPTCHA challenges
- IP-based blocking
- Session/cookie requirements

---

## Practical Data Access Strategies

### Strategy 1: Flipp API (RECOMMENDED) ⭐

**Flipp offers a Partner API** for legitimate business use cases.

```
Official API: https://corp.flipp.com/flipp-api
Contact: partnerships@flipp.com
```

**Pros**:
- Legal and legitimate
- Structured JSON data with prices, products, locations
- No scraping needed
- High reliability
- Includes metadata (store, dates, categories)

**Cons**:
- Requires business relationship
- May have costs (varies by use case)
- Approval process

**API Response Structure** (typical):
```json
{
  "flyer_id": "7614496",
  "retailer": "Food Basics",
  "valid_from": "2024-01-11",
  "valid_to": "2024-01-17",
  "pages": [
    {
      "page_number": 1,
      "image_url": "https://...",
      "items": [
        {
          "id": "12345",
          "name": "Bananas",
          "price": "0.69",
          "unit": "lb",
          "category": "Produce",
          "coordinates": {"x": 100, "y": 200, "width": 150, "height": 200}
        }
      ]
    }
  ]
}
```

**Recommendation**: Start here if building a commercial product. Much better than scraping.

### Strategy 2: Browser Automation (Selenium/Playwright)

Since direct HTTP requests get blocked, use a real browser.

```python
from playwright.sync_api import sync_playwright
import time

def fetch_flipp_flyer(url):
    """Fetch Flipp flyer using browser automation"""

    with sync_playwright() as p:
        # Launch browser (chromium, firefox, or webkit)
        browser = p.chromium.launch(headless=True)

        # Create context with realistic settings
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        page = context.new_page()

        # Navigate to flyer
        page.goto(url, wait_until='networkidle')

        # Wait for flyer to load
        page.wait_for_selector('.flyer-page', timeout=10000)

        # Extract page images
        pages = page.query_selector_all('.flyer-page img')

        image_urls = []
        for page_elem in pages:
            img_url = page_elem.get_attribute('src')
            if img_url:
                image_urls.append(img_url)

        # Alternative: Take screenshots of each page
        flyer_pages = page.query_selector_all('.flyer-page')
        for i, flyer_page in enumerate(flyer_pages):
            flyer_page.screenshot(path=f'page_{i}.png')

        # Check for structured data in page
        # Flipp often includes JSON data in script tags
        scripts = page.query_selector_all('script[type="application/json"]')
        for script in scripts:
            content = script.inner_text()
            if 'flyer' in content or 'items' in content:
                print(f"Found structured data: {content[:200]}...")

        browser.close()

        return image_urls

# Usage
url = "https://flipp.com/en-ca/toronto-on/flyer/7614496-food-basics-flyer?postal_code=M5R3N5"
images = fetch_flipp_flyer(url)
```

**Pros**:
- Works around 403 blocks
- Can extract both images and structured data
- Can handle JavaScript-rendered content

**Cons**:
- Slower (2-5s per flyer)
- More resource intensive
- May still face rate limiting
- Ethical/legal concerns (check ToS)
- Fragile (breaks when site changes)

### Strategy 3: Retailer PDF Downloads

Many retailers offer direct PDF downloads. This is the most reliable approach.

#### Metro.ca Strategy
```python
import requests
from datetime import datetime

def get_metro_flyer_pdf():
    """
    Metro typically provides PDF download links.
    Pattern: https://www.metro.ca/flyer/[region]/[date]/
    """

    # Metro flyer URLs are often predictable
    # They may have a direct PDF link or print version

    # Option 1: Look for print/PDF button in HTML
    # Option 2: Check for pattern like:
    # https://cdn.metro.ca/flyers/metro-on-[date].pdf

    # Note: Actual URL pattern needs inspection
    pass

def get_nofrills_flyer():
    """
    No Frills (Loblaw owned) often uses Flipp platform
    But may have direct PDF on their corporate site
    """
    pass
```

### Strategy 4: Mobile App APIs (Advanced)

Grocery store mobile apps often use internal APIs that return JSON.

```python
import requests

def fetch_from_mobile_api():
    """
    Reverse-engineer mobile app API calls

    Steps:
    1. Install retailer mobile app
    2. Use Charles Proxy or mitmproxy to intercept traffic
    3. Identify API endpoints for flyer data
    4. Replicate requests with proper headers
    """

    # Example (hypothetical endpoint)
    headers = {
        'User-Agent': 'MetroApp/3.2.1 (Android 12)',
        'X-API-Key': 'your-api-key',  # Found via reverse engineering
    }

    response = requests.get(
        'https://api.metro.ca/v1/flyers/current',
        headers=headers
    )

    return response.json()
```

**Pros**:
- Often returns clean JSON
- May include structured product data
- Less likely to block mobile user-agents

**Cons**:
- Requires reverse engineering
- API keys may be required
- Violates ToS (use carefully)
- APIs change frequently

### Strategy 5: Third-Party Aggregators

Use existing flyer data services:

#### Option A: Flipp API (mentioned above) ⭐

#### Option B: Reebee
- Similar to Flipp, aggregates Canadian flyers
- May have API or partnership options
- https://www.reebee.com

#### Option C: Flashfood API
- Focus on discounted items near expiry
- May have business API
- https://www.flashfood.com

#### Option D: Open Grocery Data Projects
- Check if any open data initiatives exist
- Some communities crowdsource flyer data

---

## Recommended Approach for Your POC

### Phase 1: Manual Collection (Week 1)
**Goal**: Build and test extraction pipeline with known data

```python
# 1. Manually download 20-30 sample flyers
#    - Use browser's "Print to PDF" feature
#    - Screenshot each page
#    - Save to data/raw/

# 2. Build extraction pipeline with these samples
#    - Test Tier 1 (OCR + Regex)
#    - Benchmark accuracy
#    - Identify hard cases

# 3. Validate results manually
#    - Create ground truth dataset
#    - Measure precision/recall
```

**Deliverable**: Working extraction pipeline with known accuracy on sample set.

### Phase 2: Browser Automation (Week 2-3)
**Goal**: Automate collection for ongoing testing

```python
from playwright.sync_api import sync_playwright

def collect_flyers_for_testing():
    """
    Collect flyers from Flipp using browser automation
    Limit to 5-10 flyers per day to avoid rate limiting
    """

    retailers = [
        "https://flipp.com/.../food-basics-flyer/...",
        "https://flipp.com/.../metro-flyer/...",
        "https://flipp.com/.../no-frills-flyer/...",
    ]

    for url in retailers:
        # Use Playwright to load and screenshot
        # Save as PDF or PNG
        # Rate limit: sleep(5-10 seconds between requests)
        pass
```

**Deliverable**: Automated collection of 10-20 new flyers per week for testing.

### Phase 3: Explore Flipp API (Week 3-4)
**Goal**: Investigate legitimate access

```
Actions:
1. Contact Flipp partnerships team
2. Explain use case (price comparison, consumer tool, etc.)
3. Negotiate API access or data partnership
4. If approved, integrate API
```

**Deliverable**:
- If API available: Direct JSON access (best case!)
- If not: Continue with browser automation or PDFs

### Phase 4: Scale Strategy (Week 5-6)

Based on Phase 3 results:

**If Flipp API available**:
```python
# Use API for structured data
# Skip extraction pipeline entirely
# Just normalize their JSON to your format
```

**If no API**:
```python
# Implement robust browser automation
# Add proxy rotation (if needed)
# Implement caching (don't re-fetch same flyer)
# Add rate limiting (respect servers)
# Consider using PDF downloads where available
```

---

## Legal & Ethical Considerations

### ✅ Generally Acceptable
- Using official APIs
- Downloading publicly available PDFs
- Manual browsing and screenshotting for personal use
- Respecting robots.txt
- Rate limiting requests
- Not overwhelming servers

### ⚠️ Gray Area
- Browser automation for data collection
- Reverse engineering mobile apps
- Scraping even if robots.txt allows

### ❌ Avoid
- Bypassing CAPTCHAs aggressively
- Distributed scraping to hide identity
- Reselling scraped data without permission
- Ignoring cease-and-desist requests
- DDoS-level request rates

### Best Practice
1. **Start with legitimate channels**: Contact retailers/Flipp for API access
2. **Be transparent**: Explain your use case
3. **Respect rate limits**: Don't hammer servers
4. **Check ToS**: Terms of Service for each site
5. **Consider value exchange**: Offer something back (traffic, data, etc.)

---

## Practical POC Implementation

### Option A: PDF-Based Approach (RECOMMENDED FOR POC) ⭐

```python
# Step 1: Manual PDF collection for testing
manually_download_pdfs = [
    "data/raw/metro_flyer_2024_01_11.pdf",
    "data/raw/nofrills_flyer_2024_01_11.pdf",
    "data/raw/foodbasics_flyer_2024_01_11.pdf",
]

# Step 2: Convert to images (you already have this!)
from notebooks.convert_by_pymupdf import convert_pdf_to_images

for pdf in manually_download_pdfs:
    images = convert_pdf_to_images(pdf)
    # Process with extraction pipeline

# Step 3: Extract items and prices
from extraction_pipeline import extract_items_and_prices

for image in images:
    results = extract_items_and_prices(image)
    print(results)
```

**Timeline**:
- Week 1: Collect 20 PDFs manually, build extraction
- Week 2-3: Refine extraction accuracy to 90%+
- Week 4: Automate PDF collection via browser automation

### Option B: Screenshot-Based Approach

```python
from playwright.sync_api import sync_playwright

def screenshot_flyer_pages(url, output_dir):
    """Take screenshots of each flyer page"""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        page.goto(url)
        page.wait_for_load_state('networkidle')

        # Flipp has navigation buttons
        # Find total pages
        page_count = page.locator('.page-indicator').text_content()
        # Parse "1 of 12" -> 12 pages

        for i in range(page_count):
            # Screenshot current page
            page.screenshot(path=f'{output_dir}/page_{i}.png', full_page=False)

            # Click next button
            page.click('.next-page-button')
            page.wait_for_timeout(500)  # Wait for page to load

        browser.close()

# Usage
screenshot_flyer_pages(
    "https://flipp.com/.../food-basics-flyer/...",
    "data/raw/foodbasics_2024_01_11"
)
```

---

## Sample Flyer Characteristics (Canadian Grocers)

Based on industry knowledge:

### Metro
- **Pages**: 8-12 pages typically
- **Layout**: 2-3 column layout, professional design
- **Item Density**: Moderate (20-30 items per page)
- **Price Format**: "$X.XX" or "X.XX" with $ symbol
- **Promotions**: "2 for $5", "Buy 1 Get 1", "$X off"
- **Quality**: High (clean, professional printing)
- **Difficulty**: Medium (Tier 1 should work for 70-80%)

### No Frills
- **Pages**: 12-16 pages (larger flyers)
- **Layout**: Dense, 3-4 columns, budget-focused
- **Item Density**: High (40-50 items per page)
- **Price Format**: Large bold prices, yellow price tags
- **Promotions**: Heavy promotional text
- **Quality**: Medium (more cluttered)
- **Difficulty**: Medium-Hard (may need Tier 2 for complex pages)

### Food Basics
- **Pages**: 10-14 pages
- **Layout**: Similar to Metro (same parent company)
- **Item Density**: Moderate-High (30-40 items per page)
- **Price Format**: Standard "$X.XX"
- **Promotions**: "PC Optimum Points", multi-buy deals
- **Quality**: High
- **Difficulty**: Medium (Tier 1 likely sufficient)

---

## Next Steps for Your POC

### Week 1 Action Items

1. **Manual Collection**:
   ```bash
   # Visit each site in browser
   # Download/screenshot 5 flyers from each retailer
   # Save to data/raw/
   ```

2. **Convert to Images**:
   ```bash
   # Use your existing PyMuPDF notebook
   # Convert all PDFs to PNG at 300 DPI
   ```

3. **Manual Ground Truth**:
   ```bash
   # For 5 sample pages, manually create JSON with all items/prices
   # Use as ground truth for accuracy measurement
   ```

4. **Build Extraction Pipeline** (I can help with this!):
   ```python
   # Enhance OCR preprocessing
   # Add layout analysis
   # Implement price extraction
   # Build item-price pairing
   ```

### Week 2-3 Action Items

1. **Automate Collection**:
   - Set up Playwright browser automation
   - Collect 2-3 flyers per day per retailer
   - Build up test dataset

2. **Refine Extraction**:
   - Test on diverse flyers
   - Measure accuracy vs ground truth
   - Iterate on preprocessing and extraction logic

3. **Contact Flipp**:
   - Reach out to partnerships@flipp.com
   - Explain use case
   - Request API access or data partnership

---

## Code Template: Complete Workflow

```python
# complete_flyer_pipeline.py

from playwright.sync_api import sync_playwright
import pymupdf
from pathlib import Path
from datetime import datetime

class FlyerPipeline:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_from_flipp(self, url, retailer_name):
        """Fetch flyer using browser automation"""

        timestamp = datetime.now().strftime("%Y%m%d")
        retailer_dir = self.output_dir / f"{retailer_name}_{timestamp}"
        retailer_dir.mkdir(exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, wait_until='networkidle')

            # Take full-page screenshot or save as PDF
            page.pdf(path=retailer_dir / "flyer.pdf")

            browser.close()

        return retailer_dir / "flyer.pdf"

    def convert_to_images(self, pdf_path):
        """Convert PDF to high-quality images"""

        doc = pymupdf.open(pdf_path)
        image_dir = pdf_path.parent / "images"
        image_dir.mkdir(exist_ok=True)

        images = []
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_path = image_dir / f"page_{page_num:02d}.png"
            pix.save(img_path)
            images.append(img_path)

        return images

    def extract_items_and_prices(self, image_path):
        """Extract structured data from flyer image"""
        # To be implemented based on Tier 1/2/3 strategy
        pass

    def process_flyer(self, url, retailer_name):
        """Complete pipeline"""

        # Step 1: Fetch
        pdf_path = self.fetch_from_flipp(url, retailer_name)

        # Step 2: Convert
        images = self.convert_to_images(pdf_path)

        # Step 3: Extract
        all_items = []
        for img in images:
            items = self.extract_items_and_prices(img)
            all_items.extend(items)

        return all_items

# Usage
pipeline = FlyerPipeline()

flyers = [
    ("https://flipp.com/.../metro-flyer/...", "metro"),
    ("https://flipp.com/.../nofrills-flyer/...", "nofrills"),
    ("https://flipp.com/.../foodbasics-flyer/...", "foodbasics"),
]

for url, retailer in flyers:
    items = pipeline.process_flyer(url, retailer)
    print(f"Extracted {len(items)} items from {retailer}")
```

---

## Summary & Recommendation

### For Your POC:

1. **Week 1-2**: Use **manual PDF downloads + your existing conversion pipeline**
   - Fastest to prove concept
   - No scraping issues
   - Focus on extraction quality

2. **Week 3**: Add **Playwright automation** for ongoing collection
   - Automate what you proved manually
   - Rate limit to 5-10 flyers per day

3. **Week 4**: **Contact Flipp for API**
   - Best long-term solution
   - May be available for legitimate use case
   - Worth the ask!

4. **Week 5-6**: **Production hardening**
   - Based on what you learned
   - Choose best data source
   - Optimize extraction pipeline

**Want me to help you implement the extraction pipeline or set up Playwright automation?** I can start with whichever approach you prefer!
