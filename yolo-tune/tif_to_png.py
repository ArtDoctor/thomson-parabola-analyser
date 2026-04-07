import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT_DIR / "data"
DEFAULT_TARGET_DIR = ROOT_DIR / "data-png"


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr

    arr_float = arr.astype(np.float32, copy=False)
    max_value = float(arr_float.max())
    if max_value <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if max_value <= 255.0 and arr_float.min() >= 0.0:
        return arr_float.astype(np.uint8)

    scaled = arr_float / max_value * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_single_image(filename: str, source_dir: str, target_dir: str) -> str | None:
    source_path = Path(source_dir) / filename
    if source_path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    try:
        target_path = Path(target_dir) / f"{source_path.stem}.png"
        with Image.open(source_path) as image:
            array = np.asarray(image)
            converted = Image.fromarray(_normalize_to_uint8(array))
            converted.save(target_path, "PNG")
        return f"Converted: {source_path.name} -> {target_path.name}"
    except Exception as exc:
        return f"Failed to convert {source_path.name}: {exc}"


def convert_tiff_to_png_parallel(
    source_dir: Path,
    target_dir: Path,
    max_workers: int | None = None,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    filenames = sorted(path.name for path in source_dir.iterdir())

    if max_workers == 1:
        for filename in filenames:
            result = process_single_image(filename, str(source_dir), str(target_dir))
            if result:
                print(result)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                filename,
                str(source_dir),
                str(target_dir),
            )
            for filename in filenames
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TIFF detector images into PNG files for YOLO workflows.",
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing .tif/.tiff files.",
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=str(DEFAULT_TARGET_DIR),
        help="Directory where converted PNG files should be written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional process count override.",
    )
    args = parser.parse_args()

    convert_tiff_to_png_parallel(
        Path(args.source_dir).expanduser().resolve(),
        Path(args.target_dir).expanduser().resolve(),
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
