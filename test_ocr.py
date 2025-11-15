#!/usr/bin/env python3
"""
Unified OCR Testing Script

Test any OCR engine with a simple, consistent interface.
Uses the plugin system - adding new engines is trivial!

Usage:
    # List all available engines
    python test_ocr.py list

    # Test one engine
    python test_ocr.py test --image flyer.png --engine minicpm

    # Compare all engines
    python test_ocr.py compare --image flyer.png

    # Compare macOS-compatible engines only (apple-vision, tesseract, easyocr, surya)
    python test_ocr.py compare-macos --image flyer.png

    # Extract text from specific colored backgrounds (grocery flyers, highlights, etc.)
    python test_ocr.py test-colors --image flyer.png --colors red --colors yellow

    # Batch test multiple images
    python test_ocr.py batch --images data/images/*.png --engine got-ocr

    # Get engine info
    python test_ocr.py info --engine phi3
"""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Try to import the plugin system
try:
    from pdf2img.engines import (
        OCRResult,
        get_engine,
        get_engine_info,
        list_all_engine_info,
        list_engines,
    )
except ImportError as e:
    print(f"Error importing OCR engines: {e}")
    print("Make sure you've installed the package: pip install -e .")
    exit(1)


app = typer.Typer(help="Unified OCR testing tool with plugin support")
console = Console()


def print_result(result: OCRResult, verbose: bool = False):
    """Pretty print OCR result"""
    console.print(f"\n{'='*80}", style="bold")
    console.print(f"Engine: {result.engine}", style="bold cyan")
    console.print(f"Model Size: {result.model_size}")
    console.print(f"Processing Time: {result.processing_time:.2f}s", style="yellow")

    if result.ram_usage_gb:
        console.print(f"RAM Usage: ~{result.ram_usage_gb:.1f}GB")

    console.print(f"{'='*80}", style="bold")

    if result.success:
        console.print(f"\n✓ Extracted Text ({len(result.text)} chars):", style="bold green")
        console.print(f"{'-'*80}")
        console.print(result.text)
        console.print(f"{'-'*80}")

        if verbose and result.metadata:
            console.print(f"\nMetadata: {result.metadata}", style="dim")
    else:
        console.print(f"\n✗ FAILED: {result.error}", style="bold red")


@app.command()
def list():
    """List all available OCR engines"""
    engines = list_engines()

    if not engines:
        console.print("No OCR engines registered!", style="bold red")
        console.print("\nMake sure you've installed optional dependencies:")
        console.print("  uv pip install -e '.[recommended-2025]'")
        return

    console.print(f"\n[bold]Available OCR Engines ({len(engines)}):[/bold]\n")

    for engine_name in engines:
        console.print(f"  • {engine_name}", style="cyan")

    console.print("\nUse 'test_ocr.py info --engine <name>' for details")


@app.command()
def info(engine: str = typer.Option(..., "--engine", "-e", help="Engine name")):
    """Show detailed information about an engine"""
    try:
        info_dict = get_engine_info(engine)

        console.print(f"\n[bold]{info_dict['name']}[/bold]")
        console.print(f"{'-'*40}")
        console.print(f"Description: {info_dict['description']}")
        console.print(f"Model Size: {info_dict['model_size']}")
        console.print(f"RAM Usage: ~{info_dict['ram_usage_gb']:.1f}GB")
        console.print(f"Class: {info_dict['class']}")
        console.print()

    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        console.print(f"\nAvailable engines: {', '.join(list_engines())}")


@app.command()
def test(
    image: Path = typer.Option(..., "--image", "-i", help="Path to image file"),
    engine: str = typer.Option("minicpm", "--engine", "-e", help="Engine to use"),
    save: bool = typer.Option(False, "--save", help="Save output to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Test a single OCR engine on an image"""

    if not image.exists():
        console.print(f"Error: Image not found: {image}", style="bold red")
        raise typer.Exit(1)

    # Get engine
    try:
        eng = get_engine(engine)
    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        console.print(f"\nAvailable engines: {', '.join(list_engines())}")
        raise typer.Exit(1)

    # Run extraction
    console.print(f"\n[bold]Testing {engine} on {image.name}[/bold]\n")

    result = eng._safe_extract(image)

    # Print result
    print_result(result, verbose=verbose)

    # Save if requested
    if save and result.success:
        output_file = Path(f"output_{engine}_{image.stem}.txt")
        output_file.write_text(result.text)
        console.print(f"\n✓ Saved to {output_file}", style="bold green")


@app.command()
def compare(
    image: Path = typer.Option(..., "--image", "-i", help="Path to image file"),
    engines: Optional[List[str]] = typer.Option(
        None, "--engines", "-e", help="Specific engines to compare (default: all)"
    ),
    save_outputs: bool = typer.Option(False, "--save", help="Save all outputs"),
):
    """Compare multiple OCR engines on the same image"""

    if not image.exists():
        console.print(f"Error: Image not found: {image}", style="bold red")
        raise typer.Exit(1)

    # Determine which engines to test
    if engines:
        engines_to_test = engines
    else:
        engines_to_test = list_engines()

    if not engines_to_test:
        console.print("No engines available to compare!", style="bold red")
        raise typer.Exit(1)

    console.print(f"\n[bold]Comparing {len(engines_to_test)} engines on {image.name}[/bold]\n")

    results = []

    # Test each engine
    for engine_name in engines_to_test:
        try:
            eng = get_engine(engine_name)
            console.print(f"Running {engine_name}...", style="cyan")
            result = eng._safe_extract(image)
            results.append(result)

            if save_outputs and result.success:
                output_file = Path(f"output_{engine_name}_{image.stem}.txt")
                output_file.write_text(result.text)

        except Exception as e:
            logger.error(f"Failed to test {engine_name}: {e}")
            # Create a failed result so it still appears in the comparison table
            results.append(
                OCRResult(
                    engine=engine_name,
                    text="",
                    processing_time=0.0,
                    model_size="N/A",
                    ram_usage_gb=0.0,
                    success=False,
                    error=str(e),
                )
            )

    # Print comparison table
    console.print(f"\n{'='*100}")
    console.print("[bold]COMPARISON SUMMARY[/bold]")
    console.print(f"{'='*100}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Engine", style="cyan")
    table.add_column("Model Size")
    table.add_column("Time (s)", justify="right")
    table.add_column("RAM (GB)", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Status")

    for r in results:
        status = "✓ OK" if r.success else "✗ FAIL"
        status_style = "green" if r.success else "red"

        table.add_row(
            r.engine,
            r.model_size,
            f"{r.processing_time:.2f}",
            f"~{r.ram_usage_gb:.1f}" if r.ram_usage_gb else "N/A",
            str(len(r.text)) if r.success else "0",
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)
    console.print()

    if save_outputs:
        successful = sum(1 for r in results if r.success)
        console.print(
            f"✓ Saved {successful} outputs to output_*_{image.stem}.txt", style="bold green"
        )


@app.command()
def compare_macos(
    image: Path = typer.Option(..., "--image", "-i", help="Path to image file"),
    save_outputs: bool = typer.Option(False, "--save", help="Save all outputs"),
):
    """Compare macOS-compatible OCR engines (apple-vision, tesseract, easyocr, surya)"""

    if not image.exists():
        console.print(f"Error: Image not found: {image}", style="bold red")
        raise typer.Exit(1)

    # macOS-compatible engines only
    macos_engines = ["apple-vision", "tesseract", "easyocr", "surya"]

    # Filter to only available engines
    available_engines = list_engines()
    engines_to_test = [e for e in macos_engines if e in available_engines]

    if not engines_to_test:
        console.print("No macOS-compatible engines available!", style="bold red")
        console.print("\nInstall with: uv pip install -e '.[macos-compatible]'")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Comparing {len(engines_to_test)} macOS-compatible engines on {image.name}[/bold]"
    )
    console.print(f"[dim]Testing: {', '.join(engines_to_test)}[/dim]\n")

    results = []

    # Test each engine
    for engine_name in engines_to_test:
        try:
            eng = get_engine(engine_name)
            console.print(f"Running {engine_name}...", style="cyan")
            result = eng._safe_extract(image)
            results.append(result)

            if save_outputs and result.success:
                output_file = Path(f"output_{engine_name}_{image.stem}.txt")
                output_file.write_text(result.text)

        except Exception as e:
            logger.error(f"Failed to test {engine_name}: {e}")
            # Create a failed result so it still appears in the comparison table
            results.append(
                OCRResult(
                    engine=engine_name,
                    text="",
                    processing_time=0.0,
                    model_size="N/A",
                    ram_usage_gb=0.0,
                    success=False,
                    error=str(e),
                )
            )

    # Print comparison table
    console.print(f"\n{'='*100}")
    console.print("[bold]macOS COMPARISON SUMMARY[/bold]")
    console.print(f"{'='*100}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Engine", style="cyan")
    table.add_column("Model Size")
    table.add_column("Time (s)", justify="right")
    table.add_column("RAM (GB)", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Status")

    for r in results:
        status = "✓ OK" if r.success else "✗ FAIL"
        status_style = "green" if r.success else "red"

        table.add_row(
            r.engine,
            r.model_size,
            f"{r.processing_time:.2f}",
            f"~{r.ram_usage_gb:.1f}" if r.ram_usage_gb else "N/A",
            str(len(r.text)) if r.success else "0",
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)
    console.print()

    # Show best performer
    successful_results = [r for r in results if r.success]
    if successful_results:
        fastest = min(successful_results, key=lambda r: r.processing_time)
        console.print(
            f"⚡ [bold green]Fastest: {fastest.engine}[/bold green] ({fastest.processing_time:.2f}s)"
        )

        most_accurate = max(successful_results, key=lambda r: len(r.text))
        console.print(
            f"📝 [bold green]Most text extracted: {most_accurate.engine}[/bold green] ({len(most_accurate.text)} chars)"
        )
        console.print()

    if save_outputs:
        successful = sum(1 for r in results if r.success)
        console.print(
            f"✓ Saved {successful} outputs to output_*_{image.stem}.txt", style="bold green"
        )


@app.command()
def batch(
    images: List[Path] = typer.Argument(..., help="Image files to process"),
    engine: str = typer.Option("minicpm", "--engine", "-e", help="Engine to use"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
):
    """Batch process multiple images with one engine"""

    # Filter existing images
    valid_images = [img for img in images if img.exists()]

    if not valid_images:
        console.print("Error: No valid images found!", style="bold red")
        raise typer.Exit(1)

    console.print(f"\n[bold]Batch processing {len(valid_images)} images with {engine}[/bold]\n")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get engine
    try:
        eng = get_engine(engine)
    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)

    # Process each image
    results = []

    for img in valid_images:
        console.print(f"Processing {img.name}...", style="cyan")
        result = eng._safe_extract(img)
        results.append(result)

        # Save output
        if result.success:
            output_file = output_dir / f"{img.stem}_extracted.txt"
            output_file.write_text(result.text)
            console.print(f"  ✓ Saved to {output_file}", style="green")
        else:
            console.print(f"  ✗ Failed: {result.error}", style="red")

    # Summary
    console.print(f"\n{'='*80}")
    console.print("[bold]BATCH PROCESSING SUMMARY[/bold]")
    console.print(f"{'='*80}")
    console.print(f"Total images: {len(valid_images)}")
    console.print(f"Successful: {sum(1 for r in results if r.success)}", style="green")
    console.print(f"Failed: {sum(1 for r in results if not r.success)}", style="red")
    console.print(f"Average time: {sum(r.processing_time for r in results) / len(results):.2f}s")
    console.print(f"{'='*80}\n")


@app.command()
def engines_table():
    """Show a detailed table of all available engines"""
    all_info = list_all_engine_info()

    if not all_info:
        console.print("No engines available!", style="bold red")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Size")
    table.add_column("RAM (GB)", justify="right")

    for info in all_info:
        table.add_row(
            info["name"],
            info["description"],
            info["model_size"],
            f"~{info['ram_usage_gb']:.1f}",
        )

    console.print("\n[bold]OCR Engines[/bold]\n")
    console.print(table)
    console.print()


@app.command()
def test_colors(
    image: Path = typer.Option(..., "--image", "-i", help="Path to image file"),
    engine: str = typer.Option("apple-vision", "--engine", "-e", help="Engine to use"),
    colors: List[str] = typer.Option(
        ..., "--colors", "-c", help="Colors to filter (e.g., red yellow)"
    ),
    tolerance: float = typer.Option(
        1.2, "--tolerance", "-t", help="Color matching tolerance (1.0-2.0)"
    ),
    save_filtered: bool = typer.Option(False, "--save-filtered", help="Save filtered image"),
):
    """Extract text from specific colored regions only.

    Useful for extracting prices from yellow tags, promotions from red backgrounds, etc.

    Examples:
        # Extract text from red and yellow backgrounds in a grocery flyer
        python test_ocr.py test-colors -i flyer.png -c red -c yellow

        # Use Tesseract engine with higher tolerance
        python test_ocr.py test-colors -i flyer.png -c red -e tesseract -t 1.5

        # Save the filtered image to see what was detected
        python test_ocr.py test-colors -i flyer.png -c yellow --save-filtered

    Available colors: red, yellow, green, blue, orange, purple
    """
    if not image.exists():
        console.print(f"Error: Image not found: {image}", style="bold red")
        raise typer.Exit(1)

    try:
        from pdf2img.utils import apply_color_filter, extract_colored_regions
    except ImportError as e:
        console.print(f"Error: {e}", style="bold red")
        console.print("\nInstall opencv-python: pip install opencv-python", style="yellow")
        raise typer.Exit(1)

    # Get engine
    try:
        eng = get_engine(engine)
    except Exception as e:
        console.print(f"Error loading engine '{engine}': {e}", style="bold red")
        raise typer.Exit(1)

    console.print("\n[bold]Color-Filtered OCR Test[/bold]")
    console.print(f"Image: {image.name}")
    console.print(f"Engine: {engine}")
    console.print(f"Colors: {', '.join(colors)}")
    console.print(f"Tolerance: {tolerance}x\n")

    # Extract colored regions
    console.print("🎨 Detecting colored regions...", style="cyan")
    regions = extract_colored_regions(image, colors, tolerance=tolerance, min_area=100)

    if not regions:
        console.print("\n⚠️  No regions found with specified colors!", style="yellow")
        console.print("Try increasing tolerance: --tolerance 1.5 or --tolerance 2.0", style="dim")
        raise typer.Exit(0)

    console.print(f"✓ Found {len(regions)} colored regions\n", style="green")

    # Show region breakdown
    from collections import Counter

    region_counts = Counter(r["color"] for r in regions)
    for color, count in region_counts.items():
        console.print(f"  • {color}: {count} regions")

    # Run OCR on each region
    console.print(f"\n📝 Running {engine} OCR on colored regions...\n", style="cyan")

    results_by_color = {color: [] for color in colors}
    all_text = []

    for i, region in enumerate(regions, 1):
        console.print(
            f"  [{i}/{len(regions)}] Processing {region['color']} region "
            f"({region['bbox'][2]}x{region['bbox'][3]} px)...",
            style="dim",
        )

        # Save region as temporary image
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            region["image"].save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Run OCR on region
            result = eng._safe_extract(tmp_path)

            if result.success and result.text.strip():
                results_by_color[region["color"]].append(result.text.strip())
                all_text.append(result.text.strip())
        finally:
            # Clean up temp file
            tmp_path.unlink()

    # Display results
    console.print(f"\n{'='*80}")
    console.print("[bold]COLOR-FILTERED OCR RESULTS[/bold]")
    console.print(f"{'='*80}\n")

    total_chars = sum(len(t) for t in all_text)

    if not all_text:
        console.print("❌ No text extracted from colored regions", style="red")
    else:
        # Show results by color
        for color in colors:
            texts = results_by_color[color]
            if texts:
                console.print(
                    f"[bold {color}]■ {color.upper()} regions ({len(texts)} found):[/bold {color}]"
                )
                for text in texts:
                    # Show first 100 chars of each text block
                    preview = text[:100] + "..." if len(text) > 100 else text
                    console.print(f"  {preview}", style="dim")
                console.print()

        console.print(f"{'='*80}")
        console.print(f"Total text blocks: {len(all_text)}")
        console.print(f"Total characters: {total_chars}")
        console.print(f"{'='*80}\n")

        # Save combined results
        output_file = image.parent / f"{image.stem}_colors_{'_'.join(colors)}.txt"
        output_file.write_text("\n\n".join(all_text))
        console.print(f"✓ Saved combined results to: {output_file}", style="green")

        # Save results by color
        for color in colors:
            if results_by_color[color]:
                color_file = image.parent / f"{image.stem}_color_{color}.txt"
                color_file.write_text("\n\n".join(results_by_color[color]))
                console.print(f"✓ Saved {color} results to: {color_file}", style="green")

    # Save filtered image if requested
    if save_filtered:
        from PIL import Image as PILImage

        filtered_img = apply_color_filter(PILImage.open(image), colors, tolerance)
        filtered_path = image.parent / f"{image.stem}_filtered_{'_'.join(colors)}.png"
        filtered_img.save(filtered_path)
        console.print(f"\n🎨 Saved filtered image to: {filtered_path}", style="cyan")

    console.print()


if __name__ == "__main__":
    app()
