"""
Integration tests for OCR engine implementations

These tests verify that engine classes are properly structured
and can be instantiated. Actual OCR testing requires models to be
installed and is marked as slow.
"""

import pytest

from pdf2img.engines import get_engine, get_engine_info, list_engines

# List of expected engine names (update as new engines are added)
EXPECTED_ENGINES = ["mock"]  # Always have mock from conftest

# Engines that might be available if dependencies are installed
OPTIONAL_ENGINES = ["minicpm", "got-ocr", "phi3", "surya"]


class TestEngineAvailability:
    """Test which engines are available"""

    def test_mock_engine_available(self):
        """Test that mock engine is always available"""
        assert "mock" in list_engines()

    def test_list_engines_returns_list(self):
        """Test that list_engines returns a list"""
        engines = list_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0

    @pytest.mark.parametrize("engine_name", OPTIONAL_ENGINES)
    def test_optional_engine_info(self, engine_name):
        """Test getting info for optional engines if available"""
        if engine_name in list_engines():
            info = get_engine_info(engine_name)
            assert info["name"] == engine_name
            assert "model_size" in info
            assert "ram_usage_gb" in info


class TestEngineStructure:
    """Test that engines have correct structure"""

    def test_all_engines_have_required_attributes(self):
        """Test that all registered engines have required attributes"""
        for engine_name in list_engines():
            info = get_engine_info(engine_name)

            # Check required fields
            assert "name" in info
            assert "model_size" in info
            assert "ram_usage_gb" in info
            assert "description" in info
            assert "class" in info

            # Check types
            assert isinstance(info["name"], str)
            assert isinstance(info["model_size"], str)
            assert isinstance(info["ram_usage_gb"], (int, float))
            assert isinstance(info["description"], str)

    def test_all_engines_can_be_instantiated(self):
        """Test that all engines can be instantiated"""
        for engine_name in list_engines():
            try:
                engine = get_engine(engine_name)
                assert engine is not None
                assert hasattr(engine, "extract")
                assert callable(engine.extract)
            except ImportError:
                # Skip engines with missing dependencies
                pytest.skip(f"Dependencies for {engine_name} not installed")


@pytest.mark.slow
class TestEngineOCR:
    """Tests that actually run OCR (marked as slow)

    These tests require models to be downloaded and installed.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    def test_mock_engine_extraction(self, mock_engine, sample_image):
        """Test mock engine extraction (fast, always available)"""
        result = mock_engine.extract(sample_image)

        assert result.success is True
        assert len(result.text) > 0
        assert result.engine == "mock"

    @pytest.mark.skipif("minicpm" not in list_engines(), reason="MiniCPM not installed")
    def test_minicpm_engine_structure(self):
        """Test MiniCPM engine can be instantiated"""
        engine = get_engine("minicpm")
        assert engine.name == "minicpm"
        assert engine.model_size == "8B"
        assert engine.ram_usage_gb > 0

    @pytest.mark.skipif("got-ocr" not in list_engines(), reason="GOT-OCR not installed")
    def test_got_ocr_engine_structure(self):
        """Test GOT-OCR engine can be instantiated"""
        engine = get_engine("got-ocr")
        assert engine.name == "got-ocr"
        assert engine.model_size == "580M"
        assert engine.ram_usage_gb == 2.0

    @pytest.mark.skipif("phi3" not in list_engines(), reason="Phi-3.5 not installed")
    def test_phi3_engine_structure(self):
        """Test Phi-3.5 engine can be instantiated"""
        engine = get_engine("phi3")
        assert engine.name == "phi3"
        assert engine.model_size == "4.2B"
        assert engine.ram_usage_gb > 0

    @pytest.mark.skipif("surya" not in list_engines(), reason="Surya not installed")
    def test_surya_engine_structure(self):
        """Test Surya engine can be instantiated"""
        engine = get_engine("surya")
        assert engine.name == "surya"
        assert "400MB" in engine.model_size or "surya" in engine.model_size.lower()


class TestEngineConfiguration:
    """Test engine configuration options"""

    def test_mock_engine_with_custom_text(self, custom_mock_engine, sample_image):
        """Test mock engine with custom text"""
        custom_text = "This is custom mock text"
        engine = custom_mock_engine(mock_text=custom_text)

        result = engine.extract(sample_image)

        assert result.text == custom_text

    def test_mock_engine_failure_mode(self, custom_mock_engine, sample_image):
        """Test mock engine in failure mode"""
        engine = custom_mock_engine(fail=True)

        result = engine._safe_extract(sample_image)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.skipif("minicpm" not in list_engines(), reason="MiniCPM not installed")
    def test_minicpm_with_4bit_config(self):
        """Test MiniCPM with 4-bit quantization config"""
        engine = get_engine("minicpm", use_4bit=True)
        assert engine.use_4bit is True

        engine = get_engine("minicpm", use_4bit=False)
        assert engine.use_4bit is False

    @pytest.mark.skipif("phi3" not in list_engines(), reason="Phi-3.5 not installed")
    def test_phi3_with_4bit_config(self):
        """Test Phi-3.5 with 4-bit quantization config"""
        engine = get_engine("phi3", use_4bit=True)
        assert engine.use_4bit is True


class TestEngineMetadata:
    """Test engine metadata handling"""

    def test_mock_engine_metadata(self, mock_engine, sample_image):
        """Test that mock engine includes metadata"""
        result = mock_engine.extract(sample_image)

        assert result.metadata is not None
        assert isinstance(result.metadata, dict)
        assert "call_count" in result.metadata

    def test_metadata_increments(self, mock_engine, sample_image):
        """Test that metadata tracks state correctly"""
        result1 = mock_engine.extract(sample_image)
        count1 = result1.metadata["call_count"]

        result2 = mock_engine.extract(sample_image)
        count2 = result2.metadata["call_count"]

        assert count2 == count1 + 1


class TestEngineErrorHandling:
    """Test engine error handling"""

    def test_extract_with_nonexistent_image(self, mock_engine):
        """Test extraction with non-existent image"""
        from pathlib import Path

        fake_path = Path("nonexistent_image.png")

        # _safe_extract should handle this gracefully
        result = mock_engine._safe_extract(fake_path)

        # Result structure should still be valid
        assert isinstance(result.processing_time, float)
        assert result.engine == "mock"

    def test_failing_engine_returns_valid_result(self, failing_mock_engine, sample_image):
        """Test that failing engine still returns valid OCRResult"""
        result = failing_mock_engine._safe_extract(sample_image)

        # Should return valid OCRResult even on failure
        assert isinstance(result.processing_time, float)
        assert result.success is False
        assert result.text == ""
        assert result.error is not None


# Test markers for pytest
pytestmark = pytest.mark.unit  # Mark all tests in this file as unit tests
