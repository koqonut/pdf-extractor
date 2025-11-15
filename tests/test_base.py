"""
Tests for pdf2img.engines.base module

Tests the core plugin system: OCREngine, OCRResult, and registry functions.
"""

import pytest

from pdf2img.engines import (
    OCREngine,
    OCRResult,
    get_engine,
    get_engine_info,
    list_all_engine_info,
    list_engines,
    register_engine,
)


class TestOCRResult:
    """Tests for OCRResult dataclass"""

    def test_ocr_result_creation(self):
        """Test creating an OCRResult"""
        result = OCRResult(
            engine="test-engine",
            text="Extracted text",
            processing_time=1.5,
            model_size="1B",
            ram_usage_gb=2.0,
        )

        assert result.engine == "test-engine"
        assert result.text == "Extracted text"
        assert result.processing_time == 1.5
        assert result.model_size == "1B"
        assert result.ram_usage_gb == 2.0
        assert result.success is True
        assert result.error is None
        assert result.metadata == {}

    def test_ocr_result_with_error(self):
        """Test creating a failed OCRResult"""
        result = OCRResult(
            engine="test-engine",
            text="",
            processing_time=0.5,
            model_size="1B",
            success=False,
            error="Test error message",
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.text == ""

    def test_ocr_result_with_metadata(self):
        """Test OCRResult with custom metadata"""
        metadata = {"key": "value", "count": 42}
        result = OCRResult(
            engine="test",
            text="text",
            processing_time=1.0,
            model_size="1B",
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["key"] == "value"
        assert result.metadata["count"] == 42


class TestOCREngine:
    """Tests for OCREngine base class"""

    def test_engine_has_required_attributes(self, mock_engine):
        """Test that engine has all required attributes"""
        assert hasattr(mock_engine, "name")
        assert hasattr(mock_engine, "model_size")
        assert hasattr(mock_engine, "ram_usage_gb")
        assert hasattr(mock_engine, "description")
        assert hasattr(mock_engine, "extract")

    def test_engine_extract_method(self, mock_engine, sample_image):
        """Test that extract method returns OCRResult"""
        result = mock_engine.extract(sample_image)

        assert isinstance(result, OCRResult)
        assert result.engine == "mock"
        assert result.success is True

    def test_engine_safe_extract(self, mock_engine, sample_image):
        """Test _safe_extract wrapper method"""
        result = mock_engine._safe_extract(sample_image)

        assert isinstance(result, OCRResult)
        assert result.success is True

    def test_engine_safe_extract_handles_errors(self, failing_mock_engine, sample_image):
        """Test that _safe_extract handles exceptions"""
        result = failing_mock_engine._safe_extract(sample_image)

        assert isinstance(result, OCRResult)
        assert result.success is False
        assert result.error is not None
        assert "Mock engine configured to fail" in result.error

    def test_engine_repr(self, mock_engine):
        """Test engine string representation"""
        repr_str = repr(mock_engine)

        assert "MockEngine" in repr_str
        assert "name='mock'" in repr_str
        assert "size=1B" in repr_str
        assert "ram=1.0GB" in repr_str


class TestEngineRegistry:
    """Tests for engine registry system"""

    def test_list_engines(self):
        """Test listing registered engines"""
        engines = list_engines()

        assert isinstance(engines, list)
        assert len(engines) > 0
        assert "mock" in engines  # Mock engine from conftest

    def test_get_engine(self):
        """Test getting an engine by name"""
        engine = get_engine("mock")

        assert engine is not None
        assert engine.name == "mock"

    def test_get_engine_with_config(self, custom_mock_engine):
        """Test getting engine with custom configuration"""
        engine = get_engine("mock", mock_text="Custom config text")

        # Verify the engine was configured with custom text
        assert engine.mock_text == "Custom config text"

    def test_get_engine_unknown_raises_error(self):
        """Test that getting unknown engine raises ValueError"""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("nonexistent-engine")

    def test_get_engine_info(self):
        """Test getting engine information"""
        info = get_engine_info("mock")

        assert info["name"] == "mock"
        assert info["model_size"] == "1B"
        assert info["ram_usage_gb"] == 1.0
        assert "description" in info
        assert "class" in info

    def test_get_engine_info_unknown_raises_error(self):
        """Test that getting info for unknown engine raises ValueError"""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine_info("nonexistent-engine")

    def test_list_all_engine_info(self):
        """Test listing information for all engines"""
        all_info = list_all_engine_info()

        assert isinstance(all_info, list)
        assert len(all_info) > 0

        # Check first engine info structure
        info = all_info[0]
        assert "name" in info
        assert "model_size" in info
        assert "ram_usage_gb" in info
        assert "description" in info
        assert "class" in info

    def test_register_engine_decorator(self):
        """Test @register_engine decorator"""

        @register_engine
        class TestEngine(OCREngine):
            name = "test-decorator-engine"
            model_size = "500M"
            ram_usage_gb = 0.5
            description = "Test engine for decorator"

            def extract(self, image_path, **kwargs):
                return OCRResult(
                    engine=self.name,
                    text="test",
                    processing_time=0.1,
                    model_size=self.model_size,
                )

        # Engine should now be registered
        assert "test-decorator-engine" in list_engines()

        # Should be able to get it
        engine = get_engine("test-decorator-engine")
        assert engine.name == "test-decorator-engine"

    def test_register_engine_without_name_raises_error(self):
        """Test that registering engine without name raises ValueError"""

        with pytest.raises(ValueError, match="must set a unique 'name'"):

            @register_engine
            class BadEngine(OCREngine):
                # name not set, should use default "override_me"
                model_size = "1B"
                ram_usage_gb = 1.0

                def extract(self, image_path, **kwargs):
                    pass

    def test_register_non_engine_raises_error(self):
        """Test that registering non-OCREngine class raises TypeError"""

        with pytest.raises(TypeError, match="must inherit from OCREngine"):

            @register_engine
            class NotAnEngine:
                name = "not-an-engine"


class TestEngineIntegration:
    """Integration tests for engine system"""

    def test_multiple_engines_registered(self):
        """Test that multiple engines are registered"""
        engines = list_engines()

        # Should have at least mock engine + any real engines
        assert len(engines) >= 1
        assert "mock" in engines

    def test_engine_lazy_loading(self, mock_engine):
        """Test that models are lazy loaded"""
        # Mock engine doesn't have real lazy loading, but we test the pattern
        assert mock_engine._model is None

    def test_engine_call_count(self, mock_engine, sample_image):
        """Test that engine tracks calls correctly"""
        initial_count = mock_engine.call_count

        mock_engine.extract(sample_image)
        assert mock_engine.call_count == initial_count + 1

        mock_engine.extract(sample_image)
        assert mock_engine.call_count == initial_count + 2

    def test_extract_timing(self, mock_engine, sample_image):
        """Test that processing time is recorded"""
        result = mock_engine.extract(sample_image)

        assert result.processing_time > 0
        # Mock engine sleeps for 0.01s, so should be at least that
        assert result.processing_time >= 0.01

    def test_batch_extraction(self, mock_engine, sample_images):
        """Test extracting from multiple images"""
        results = []

        for image in sample_images:
            result = mock_engine.extract(image)
            results.append(result)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.engine == "mock" for r in results)
