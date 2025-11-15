"""
Color-based region extraction for targeted OCR.

Extracts text only from regions with specific background colors.
Useful for grocery flyers, highlighted documents, color-coded sections.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
from loguru import logger


# Predefined color ranges in HSV
COLOR_RANGES = {
    "red": [
        # Red wraps around in HSV (0-10 and 170-180)
        ((0, 100, 100), (10, 255, 255)),  # Lower red range
        ((170, 100, 100), (180, 255, 255)),  # Upper red range
    ],
    "yellow": [((20, 100, 100), (30, 255, 255))],
    "green": [((40, 40, 40), (80, 255, 255))],
    "blue": [((100, 100, 100), (130, 255, 255))],
    "orange": [((10, 100, 100), (25, 255, 255))],
    "purple": [((130, 100, 100), (160, 255, 255))],
}


def create_color_mask(image: Image.Image, color: str, tolerance: float = 1.0) -> np.ndarray:
    """Create binary mask for regions with specific background color.

    Args:
        image: PIL Image
        color: Color name ('red', 'yellow', 'green', 'blue', etc.)
        tolerance: Multiplier for color range (1.0 = default, >1.0 = more lenient)

    Returns:
        Binary mask (numpy array) where 255 = color match, 0 = no match
    """
    if color.lower() not in COLOR_RANGES:
        raise ValueError(f"Unknown color '{color}'. Available: {list(COLOR_RANGES.keys())}")

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Convert RGB to HSV
    try:
        import cv2

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    except ImportError:
        raise ImportError(
            "opencv-python is required for color filtering. "
            "Install with: pip install opencv-python"
        )

    # Get color ranges
    ranges = COLOR_RANGES[color.lower()]

    # Create combined mask for all ranges of this color
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        # Apply tolerance
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        if tolerance != 1.0:
            # Adjust saturation and value ranges with tolerance
            lower[1] = max(0, int(lower[1] / tolerance))
            lower[2] = max(0, int(lower[2] / tolerance))
            upper[1] = min(255, int(upper[1] * tolerance))
            upper[2] = min(255, int(upper[2] * tolerance))

        # Create mask for this range
        range_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)

    return mask


def extract_colored_regions(
    image_path: Path,
    colors: List[str],
    tolerance: float = 1.0,
    min_area: int = 100,
) -> List[Dict]:
    """Extract regions from image that have specific background colors.

    Args:
        image_path: Path to image file
        colors: List of color names to extract (e.g., ['red', 'yellow'])
        tolerance: Color matching tolerance (1.0 = default, >1.0 = more lenient)
        min_area: Minimum region area in pixels (filters out noise)

    Returns:
        List of dicts with keys:
            - 'color': Color name
            - 'bbox': (x, y, width, height) bounding box
            - 'mask': Binary mask for this region
            - 'image': Cropped PIL Image of the region
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for color filtering. "
            "Install with: pip install opencv-python"
        )

    # Load image
    image = Image.open(image_path)

    regions = []

    for color in colors:
        logger.info(f"Extracting {color} regions from {image_path.name}...")

        # Create color mask
        mask = create_color_mask(image, color, tolerance)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and extract regions
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Create mask for this specific region
            region_mask = np.zeros_like(mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)

            # Crop region from original image
            region_img = image.crop((x, y, x + w, y + h))

            regions.append(
                {
                    "color": color,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "mask": region_mask[y : y + h, x : x + w],
                    "image": region_img,
                }
            )

        logger.info(f"Found {len([r for r in regions if r['color'] == color])} {color} regions")

    return regions


def apply_color_filter(
    image: Image.Image, colors: List[str], tolerance: float = 1.0
) -> Image.Image:
    """Create filtered image showing only specified color regions (rest is white).

    Args:
        image: PIL Image
        colors: List of color names to keep
        tolerance: Color matching tolerance

    Returns:
        Filtered PIL Image with only specified colors visible
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for color filtering. "
            "Install with: pip install opencv-python"
        )

    img_array = np.array(image)

    # Create combined mask for all colors
    combined_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

    for color in colors:
        mask = create_color_mask(image, color, tolerance)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Create white background
    white_bg = np.ones_like(img_array) * 255

    # Apply mask: keep original colors where mask is True, white elsewhere
    result = np.where(combined_mask[:, :, None], img_array, white_bg)

    return Image.fromarray(result.astype(np.uint8))
