"""
Pytest configuration and fixtures for OCR engine tests
"""

import time
from pathlib import Path

import pytest
from PIL import Image

from pdf2img.engines import OCREngine, OCRResult, register_engine


# Mock Engine for testing
@register_engine
class MockEngine(OCREngine):
    """Mock OCR engine for testing

    This engine doesn't actually do OCR, it just returns
    pre-configured responses for testing purposes.
    """

    name = "mock"
    model_size = "1B"
    ram_usage_gb = 1.0
    description = "Mock engine for testing"

    def __init__(self, mock_text: str = "Mock OCR text", fail: bool = False, **config):
        super().__init__(**config)
        self.mock_text = mock_text
        self.fail = fail
        self.call_count = 0

    def extract(self, image_path: Path, **kwargs) -> OCRResult:
        """Mock extraction that returns pre-configured text"""
        self.call_count += 1
        start_time = time.time()

        # Simulate processing time
        time.sleep(0.01)

        if self.fail:
            raise ValueError("Mock engine configured to fail")

        return OCRResult(
            engine=self.name,
            text=self.mock_text,
            processing_time=time.time() - start_time,
            model_size=self.model_size,
            ram_usage_gb=self.ram_usage_gb,
            metadata={"call_count": self.call_count},
        )


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image"""
    image_path = tmp_path / "test.png"

    # Create a simple 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(image_path)

    return image_path


@pytest.fixture
def sample_images(tmp_path):
    """Create multiple sample test images"""
    images = []

    for i in range(3):
        image_path = tmp_path / f"test_{i}.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(image_path)
        images.append(image_path)

    return images


@pytest.fixture
def mock_engine():
    """Create a mock OCR engine instance"""
    return MockEngine(mock_text="Test text from mock engine")


@pytest.fixture
def failing_mock_engine():
    """Create a mock OCR engine that fails"""
    return MockEngine(fail=True)


@pytest.fixture
def custom_mock_engine():
    """Factory fixture for creating custom mock engines"""

    def _create_mock(mock_text: str = "Custom text", fail: bool = False):
        return MockEngine(mock_text=mock_text, fail=fail)

    return _create_mock
