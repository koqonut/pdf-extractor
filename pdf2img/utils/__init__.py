"""Utility functions for PDF/image processing."""

from .color_filter import (
    COLOR_RANGES,
    apply_color_filter,
    create_color_mask,
    extract_colored_regions,
)

__all__ = [
    "COLOR_RANGES",
    "create_color_mask",
    "extract_colored_regions",
    "apply_color_filter",
]
