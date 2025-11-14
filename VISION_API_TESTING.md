# Vision API Testing Guide

Test Claude 3.5 Sonnet's ability to extract structured data from flyer images.

## Quick Start

### 1. Install Dependencies

```bash
pip install anthropic requests playwright
```

### 2. Get API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Create an API key
4. Set environment variable:

```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

### 3. Get a Sample Flyer Image

**Option A: Manual Screenshot (Recommended)**
```bash
# 1. Visit https://flipp.com
# 2. Search for "Metro" or "Food Basics" or "No Frills"
# 3. Open a flyer
# 4. Take screenshot (Win+Shift+S on Windows, Cmd+Shift+4 on Mac)
# 5. Save to: data/raw/samples/test_flyer.png
```

**Option B: Download from URL**
```bash
python download_sample_flyer.py
```

### 4. Run Test

**Using Python Script:**
```bash
python test_vision_api.py --image data/raw/samples/test_flyer.png
```

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/test_claude_vision.ipynb
```

## What Gets Tested

The test will provide:

### 📊 Performance Metrics
- Processing time per page
- Token usage (input/output)
- Cost per page in USD
- Cost projections for scale (100 flyers, 1000 flyers)

### 📦 Extraction Quality
- Number of items found
- Item names and prices
- Units (each, lb, kg, etc.)
- Promotional details ("Buy 2 Get 1", "Save $2", etc.)
- Confidence scores (0-100%)

### 🎯 Quality Assessment
- Confidence distribution (high/medium/low)
- Average confidence score
- Price validation
- Missing fields analysis

## Example Output

```
================================================================================
📊 PERFORMANCE METRICS
================================================================================

⏱️  Processing Time: 3.2s
🎫 Total Tokens: 2,847
   - Input: 1,523 tokens ($0.0046)
   - Output: 1,324 tokens ($0.0199)
💰 Cost: $0.0245 per page

================================================================================
📦 EXTRACTED DATA
================================================================================

📋 Metadata:
   - Total items found: 24
   - Page quality: high
   - Notes: Clear text, good image quality

🛒 Items Found: 24

1. Bananas
   💵 $0.69
   📦 lb
   🟢 Confidence: 98%

2. Fresh Atlantic Salmon
   💵 $9.99
   📦 lb
   🏷️  Was: $12.99
   🎉 Save $3.00
   🟢 Confidence: 95%

... and 22 more items

================================================================================
💰 COST PROJECTIONS
================================================================================

Assuming $0.0245 per page:
   - 10-page flyer: $0.245
   - 100 flyers (avg 12 pages): $29.40
   - 1,000 flyers: $294.00

================================================================================
🎯 QUALITY ASSESSMENT
================================================================================

📊 Confidence Distribution:
   🟢 High (>90%): 20 items (83%)
   🟡 Medium (70-90%): 3 items (13%)
   🔴 Low (<70%): 1 items (4%)
   📈 Average: 91%
```

## Cost Analysis

Based on Claude 3.5 Sonnet pricing:
- **Input**: $3 per million tokens (~1,500 tokens per image = $0.0045)
- **Output**: $15 per million tokens (~1,300 tokens = $0.0195)
- **Total**: ~$0.024 per page

### Projected Costs

| Scenario | Pages | Cost |
|----------|-------|------|
| Single flyer (12 pages) | 12 | $0.29 |
| 100 flyers | 1,200 | $29 |
| 1,000 flyers | 12,000 | $290 |
| 10,000 flyers | 120,000 | $2,900 |

### With Smart Routing (70% Tier 1, 20% Tier 2, 10% Tier 3)

| Scenario | Cost with Hybrid | Savings |
|----------|-----------------|---------|
| 100 flyers | $3.50 | 88% |
| 1,000 flyers | $35 | 88% |
| 10,000 flyers | $350 | 88% |

## Comparison with Other Vision Models

| Model | Cost/Page | Accuracy | Speed | Notes |
|-------|-----------|----------|-------|-------|
| Claude 3.5 Sonnet | $0.024 | 96-98% | 3-4s | ⭐ Best for documents |
| GPT-4o | $0.015 | 93-96% | 2-3s | Fast, good overall |
| Gemini 1.5 Flash | $0.008 | 90-94% | 3-4s | 💰 Most affordable |
| Gemini 1.5 Pro | $0.035 | 94-97% | 4-5s | High quality |
| GPT-4 Vision | $0.040 | 94-96% | 5-6s | Legacy, expensive |

## Interpreting Results

### High Confidence (>90%)
✅ **Trust these extractions**
- Use directly in production
- Minimal validation needed
- Expected accuracy: 98%+

### Medium Confidence (70-90%)
⚠️ **Review or validate**
- May have slight OCR errors
- Check prices manually
- Consider re-processing with better image quality

### Low Confidence (<70%)
❌ **Requires manual review**
- Likely poor image quality
- Text may be blurry or occluded
- Consider rejecting and requesting better source

## Next Steps After Testing

### If Results Are Excellent (>95% accuracy, high confidence):

**Option 1: Use Vision API for Everything**
- Simplest implementation
- Highest accuracy
- Cost: ~$0.024 per page
- Best for: Low volume (<1,000 flyers/month)

**Option 2: Hybrid Approach**
- Build Tier 1 (OCR + Regex) for simple cases
- Use Vision API as fallback for complex/low-confidence cases
- Cost: ~$0.003-0.01 per page (average)
- Best for: Medium-high volume (>1,000 flyers/month)

### If Results Are Good (85-95% accuracy):

**Option 3: Vision API + Validation**
- Use Vision API but add validation rules
- Flag suspicious extractions for manual review
- Implement price range checks, duplicate detection
- Best for: When some errors are acceptable

### If Results Are Poor (<85% accuracy):

**Option 4: Investigate Issues**
- Is image quality sufficient? (try 300+ DPI)
- Is the flyer layout too complex?
- Test with different flyer types
- Consider preprocessing images first

## Troubleshooting

### Error: "Authentication failed"
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# Or set it:
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Error: "Image too large"
```python
# Resize image before sending
from PIL import Image

img = Image.open('flyer.png')
img.thumbnail((2048, 2048), Image.LANCZOS)
img.save('flyer_resized.png')
```

### Error: "Invalid JSON response"
- Claude occasionally returns markdown-wrapped JSON
- The script automatically handles this
- If it fails, check `result['raw_response']`

### Low Accuracy Results
1. **Check image quality**: Should be 300 DPI or higher
2. **Check image size**: At least 1200px wide for flyers
3. **Test different pages**: Some pages may be harder than others
4. **Adjust prompt**: You can customize the extraction prompt

## Advanced Usage

### Test Multiple Images

```python
from pathlib import Path
import test_vision_api

image_dir = Path("data/raw/samples")
results = []

for image_path in image_dir.glob("*.png"):
    print(f"Testing: {image_path}")
    result = extract_with_claude_vision(image_path)
    results.append(result)

# Calculate averages
avg_cost = sum(r['performance']['cost_usd'] for r in results) / len(results)
avg_items = sum(len(r['extracted_data']['items']) for r in results) / len(results)

print(f"Average cost: ${avg_cost:.4f}")
print(f"Average items: {avg_items:.1f}")
```

### Custom Extraction Prompts

Edit the prompt in `test_vision_api.py` or notebook to customize:
- Add specific item categories
- Request different output format
- Add validation rules
- Request additional metadata

### Compare with Ground Truth

```python
# Create manual ground truth
ground_truth = {
    "items": [
        {"name": "Bananas", "price": "0.69", "unit": "lb"},
        {"name": "Apples", "price": "1.99", "unit": "lb"},
        # ... add all items you manually identified
    ]
}

# Compare
extracted_names = {item['name'].lower() for item in result['data']['items']}
truth_names = {item['name'].lower() for item in ground_truth['items']}

precision = len(extracted_names & truth_names) / len(extracted_names)
recall = len(extracted_names & truth_names) / len(truth_names)

print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
```

## Files Created

- `test_vision_api.py` - Command-line test script
- `download_sample_flyer.py` - Helper to get sample images
- `notebooks/test_claude_vision.ipynb` - Interactive notebook
- `VISION_API_TESTING.md` - This guide

## Best Practices

1. **Start with high-quality images**: 300 DPI, PNG format
2. **Test on diverse samples**: Different retailers, layouts, qualities
3. **Track costs**: Monitor token usage and costs
4. **Validate results**: Compare against manual counts
5. **Iterate on prompt**: Refine prompt based on results
6. **Consider hybrid approach**: Don't use vision API for everything if cheaper methods work

## Support

Questions or issues:
1. Check this guide first
2. Review the code in `test_vision_api.py`
3. Check Claude API docs: https://docs.anthropic.com/
4. Test with a different image to isolate image-specific issues

## Summary

Testing the vision API first is a **smart strategy** because:
- ✅ Validates the maximum achievable accuracy
- ✅ Provides cost benchmarks
- ✅ Takes <5 minutes to test
- ✅ Helps decide if cheaper methods are worth building
- ✅ No complex setup required

**Recommendation**: Test 5-10 different flyers from different retailers to get reliable accuracy and cost estimates, then decide on your final architecture.
