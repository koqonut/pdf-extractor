# Scripts

Utility scripts for testing and comparison. Most of these are **deprecated** in favor of the unified `test_ocr.py` CLI.

## Active Scripts

- **`download_sample_flyer.py`** - Download sample grocery flyer for testing

## Deprecated Scripts (Use `test_ocr.py` instead)

These scripts were created during development but are now superseded by the unified CLI:

- `test_2025_ocr.py` → Use `python test_ocr.py test --engine minicpm`
- `test_advanced_ocr.py` → Use `python test_ocr.py compare`
- `test_local_ocr.py` → Use `python test_ocr.py test`
- `batch_test_images.py` → Use `python test_ocr.py compare`
- `compare_all_methods.py` → Use `python test_ocr.py compare`
- `test_vision_api.py` → Use `python test_ocr.py test --engine claude-api`

**Recommended:** Use the unified CLI at the project root:
```bash
python test_ocr.py --help
```
