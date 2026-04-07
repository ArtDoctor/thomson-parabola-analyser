import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image
import synthetic_data.generate_eval_synth as generate_eval_synth
import synthetic_data.generate_synthetic_dataset as generate_synthetic_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(file_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_synthetic_data_modules_import_from_repo_root() -> None:
    lim_x, lim_y = generate_synthetic_dataset.detector_limits_from_spot(400, 1100)
    assert lim_x[0] < lim_x[1]
    assert lim_y[0] < lim_y[1]

    species = generate_eval_synth.sample_species(generate_eval_synth.random.Random(123))
    assert species


def test_unet_training_script_is_import_safe() -> None:
    module = _load_module(
        "unet_denoiser_train",
        REPO_ROOT / "unet-denoiser" / "train.py",
    )

    assert callable(module.main)


def test_unet_benchmark_script_is_import_safe() -> None:
    module = _load_module(
        "unet_denoiser_benchmark",
        REPO_ROOT / "unet-denoiser" / "benchmark.py",
    )

    assert callable(module.main)


def test_yolo_tiff_converter_preserves_uint8_input(tmp_path: Path) -> None:
    module = _load_module(
        "yolo_tif_to_png",
        REPO_ROOT / "yolo-tune" / "tif_to_png.py",
    )

    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    expected = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    tif_path = source_dir / "sample.tif"
    Image.fromarray(expected, mode="L").save(tif_path)

    result = module.process_single_image("sample.tif", str(source_dir), str(target_dir))
    assert result is not None

    converted = np.array(Image.open(target_dir / "sample.png"))
    np.testing.assert_array_equal(converted, expected)
