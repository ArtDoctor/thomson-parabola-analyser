import argparse
import ast
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from oblisk.processing.pipeline import run
from oblisk.processing.preprocessing import PreprocessedImage, preprocess_image
from oblisk.reporting.batch_summary import (
    BatchImageStats,
    stats_from_run,
    write_batch_summary_md,
)
from oblisk.reporting.pipeline_log import (
    dump_entries,
    load_log_entries_from_json,
)
from oblisk.config import Settings
from oblisk.runtime import preload_unet_denoiser


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Thomson parabola detector images.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        metavar="IMAGE",
        help="Path(s) to input image(s)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output directory. One image: this path is used as-is. "
            "Multiple images: each run uses <output-dir>/<image_stem>/ "
            "(default: outputs/<image_stem>/ per image)"
        ),
    )
    parser.add_argument(
        "--add-plots",
        action="store_true",
        help=(
            "Save diagnostic plots to output_dir/plots/ "
            "(subset: e.g. rotated view omits with-background + RMSE curve)"
        ),
    )
    parser.add_argument(
        "--add-plots-full",
        action="store_true",
        dest="add_plots_full",
        help=(
            "Save plots like --add-plots but include every figure from each step "
            "(e.g. rotated lines with background, no background, and fit RMSE vs "
            "iteration)"
        ),
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        default=True,
        help="Apply morphological denoising (default: True)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_false",
        dest="denoise",
        help="Disable morphological denoising",
    )
    parser.add_argument(
        "--denoise-kernel-size",
        type=int,
        default=5,
        help="Denoise kernel size (default: 5)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Moving average window for peak detection (default: 15)",
    )
    parser.add_argument(
        "--prominence",
        type=int,
        default=5,
        help="Peak prominence (default: 5)",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=10,
        help="Min distance between peaks (default: 10)",
    )
    parser.add_argument(
        "--max-peak-distance",
        type=int,
        default=30,
        help="Max horizontal distance to connect peaks (default: 30)",
    )
    parser.add_argument(
        "--min-line-length-1",
        type=int,
        default=10,
        help="Min line length after first filter (default: 10)",
    )
    parser.add_argument(
        "--min-line-length-2",
        type=int,
        default=90,
        help="Min line length after merge (default: 90)",
    )
    parser.add_argument(
        "--max-x-gap",
        type=int,
        default=10,
        help="Max row gap for line building (default: 10)",
    )
    parser.add_argument(
        "--direction-tol-px",
        type=int,
        default=3,
        help="Direction tolerance in pixels (default: 3)",
    )
    parser.add_argument(
        "--inner-margin-crop-px",
        type=int,
        default=50,
        metavar="N",
        help=(
            "After detector ROI, trim N pixels from each image edge "
            "(default: 50; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--no-inner-margin-crop",
        action="store_true",
        help=(
            "Disable per-edge trim after detector ROI "
            "(same as --inner-margin-crop-px 0)"
        ),
    )
    parser.add_argument(
        "--use-experimental",
        action="store_true",
        dest="use_experimental_a_for_hydrogen",
        help=(
            "Always use physics-based hydrogen a; default is smart: "
            "min(good_a) only if the lowest a is isolated (>> next peak), "
            "else analytical"
        ),
    )
    denoise_method = parser.add_mutually_exclusive_group()
    denoise_method.add_argument(
        "--denoise-unet",
        action="store_true",
        dest="use_denoise_unet",
        help="Use UNet denoiser (default)",
    )
    denoise_method.add_argument(
        "--no-denoise-unet",
        action="store_false",
        dest="use_denoise_unet",
        help="Use morphological opening instead of UNet denoiser",
    )
    parser.set_defaults(use_denoise_unet=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel worker processes for batch runs "
            "(default: CPU count; use 1 to disable parallelism)"
        ),
    )
    return parser.parse_args()


def _output_dir_for_image(
    image_path: Path,
    base_output: Path | None,
    multi: bool,
) -> Path:
    if base_output is None:
        return Path("outputs") / image_path.stem
    if multi:
        return base_output / image_path.stem
    return base_output


def _save_preprocessed(pp: PreprocessedImage, output_dir: Path) -> Path:
    """Persist preprocessed arrays + metadata to *output_dir*/.preproc.npz."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".preproc.npz"
    meta = {
        "image_path": str(pp.image_path),
        "brightest_spot": pp.brightest_spot,
        "detector_config": pp.detector_config,
        "m_per_px_img": pp.m_per_px_img,
        "orig_w": pp.orig_w,
        "orig_h": pp.orig_h,
        "denoise_title": pp.denoise_title,
        "unet_resize_scale": pp.unet_resize_scale,
        "log_entries": json.dumps(dump_entries(pp.log_entries)),
        "timings": json.dumps(pp.timings),
    }
    np.savez(
        path,
        cropped=pp.cropped,
        opened=pp.opened,
        **{
            f"meta_{key}": np.void(
                value if isinstance(value, bytes) else str(value).encode()
            )
            for key, value in meta.items()
        },
    )
    return path


def _load_preprocessed(path: Path) -> PreprocessedImage:
    """Reload a PreprocessedImage from disk (.preproc.npz)."""

    data = np.load(path, allow_pickle=False)

    def _meta(key: str) -> str:
        return bytes(data[f"meta_{key}"]).decode()

    brightest_spot_raw = ast.literal_eval(_meta("brightest_spot"))
    brightest_spot = (int(brightest_spot_raw[0]), int(brightest_spot_raw[1]))
    return PreprocessedImage(
        image_path=Path(_meta("image_path")),
        cropped=data["cropped"],
        opened=data["opened"],
        brightest_spot=brightest_spot,
        detector_config=_meta("detector_config"),
        m_per_px_img=float(_meta("m_per_px_img")),
        orig_w=int(_meta("orig_w")),
        orig_h=int(_meta("orig_h")),
        denoise_title=_meta("denoise_title"),
        unet_resize_scale=(
            None
            if _meta("unet_resize_scale") == "None"
            else float(_meta("unet_resize_scale"))
        ),
        log_entries=load_log_entries_from_json(json.loads(_meta("log_entries"))),
        timings=json.loads(_meta("timings")),
    )


def _run_one_cpu(
    preproc_path: Path,
    output_dir: Path,
    add_plots: bool,
    settings: Settings,
    use_experimental_a_for_hydrogen: bool,
    use_denoise_unet: bool,
    add_plots_full: bool,
) -> tuple[Path, Path, dict[str, Any] | None, float, str | None]:
    """CPU-only analysis phase for a single image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        preprocessed = _load_preprocessed(preproc_path)
        result = run(
            image_path=preprocessed.image_path,
            output_dir=output_dir,
            add_plots=add_plots,
            settings=settings,
            use_experimental_a_for_hydrogen=use_experimental_a_for_hydrogen,
            use_denoise_unet=use_denoise_unet,
            add_plots_full=add_plots_full,
            preprocessed=preprocessed,
        )
        elapsed = time.perf_counter() - t0
        res_path = output_dir / "res.json"
        with open(res_path, "w") as handle:
            json.dump(result, handle, indent=2)
        preproc_path.unlink(missing_ok=True)
        return (preprocessed.image_path, output_dir, result, elapsed, None)
    except Exception:
        elapsed = time.perf_counter() - t0
        return (
            Path(str(preproc_path.parent)),
            output_dir,
            None,
            elapsed,
            traceback.format_exc(),
        )


def main() -> None:
    args = _parse_args()
    image_paths: list[Path] = args.images
    for image_path in image_paths:
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

    multi = len(image_paths) > 1
    inner_margin_crop_px = (
        0 if args.no_inner_margin_crop else args.inner_margin_crop_px
    )
    settings = Settings(
        denoise=args.denoise,
        denoise_kernel_size=args.denoise_kernel_size,
        window=args.window,
        prominence=args.prominence,
        distance=args.distance,
        max_peak_distance=args.max_peak_distance,
        min_line_length_1=args.min_line_length_1,
        min_line_length_2=args.min_line_length_2,
        max_x_gap=args.max_x_gap,
        direction_tol_px=args.direction_tol_px,
        inner_margin_crop_px=inner_margin_crop_px,
    )

    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // 2)
    n_workers = max(1, min(n_workers, len(image_paths)))

    summary_parent = (
        args.output_dir if args.output_dir is not None else Path("outputs")
    )

    use_parallel = multi and n_workers > 1
    batch_stats: list[BatchImageStats] = []
    failed: list[tuple[Path, str]] = []

    if settings.denoise and args.use_denoise_unet:
        preload_unet_denoiser(
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    if use_parallel:
        print(f"Processing {len(image_paths)} images with {n_workers} workers")
        t_batch_start = time.perf_counter()

        prepped: list[tuple[Path | None, Path, Path, str | None]] = []
        for image_path in image_paths:
            output_dir = _output_dir_for_image(
                image_path,
                args.output_dir,
                multi,
            )
            try:
                preprocessed = preprocess_image(
                    image_path,
                    settings,
                    args.use_denoise_unet,
                )
                preproc_path = _save_preprocessed(preprocessed, output_dir)
                prepped.append((preproc_path, output_dir, image_path, None))
            except Exception:
                error = traceback.format_exc()
                prepped.append((None, output_dir, image_path, error))
                failed.append((image_path, error))
                print(f"FAILED preprocess {image_path}:\n{error}")

        t_preprocess = time.perf_counter() - t_batch_start
        print(f"Preprocessing done in {t_preprocess:.1f}s")

        spawn_ctx = multiprocessing.get_context("spawn")
        futures_map = {}
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=spawn_ctx,
        ) as pool:
            for prepared_path, output_dir, image_path, err_text in prepped:
                if err_text is not None:
                    continue
                assert prepared_path is not None
                fut = pool.submit(
                    _run_one_cpu,
                    preproc_path=prepared_path,
                    output_dir=output_dir,
                    add_plots=args.add_plots,
                    settings=settings,
                    use_experimental_a_for_hydrogen=(
                        args.use_experimental_a_for_hydrogen
                    ),
                    use_denoise_unet=args.use_denoise_unet,
                    add_plots_full=args.add_plots_full,
                )
                futures_map[fut] = image_path

            for fut in as_completed(futures_map):
                img_path, out_dir, result, elapsed, err_text = fut.result()
                if err_text is not None:
                    failed.append((img_path, err_text))
                    print(f"FAILED {img_path} ({elapsed:.2f}s):\n{err_text}")
                else:
                    assert result is not None
                    batch_stats.append(stats_from_run(img_path, out_dir, result))
                    res_path = out_dir / "res.json"
                    print(
                        f"--- {img_path} --- "
                        f"saved {res_path}  ({elapsed:.2f}s)"
                    )
        t_batch_wall = time.perf_counter() - t_batch_start
        print(
            f"\nBatch done: {len(batch_stats)} OK, {len(failed)} failed "
            f"in {t_batch_wall:.1f}s wall-clock"
        )
    else:
        for image_path in image_paths:
            output_dir = _output_dir_for_image(
                image_path,
                args.output_dir,
                multi,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                result = run(
                    image_path=image_path,
                    output_dir=output_dir,
                    add_plots=args.add_plots,
                    settings=settings,
                    use_experimental_a_for_hydrogen=(
                        args.use_experimental_a_for_hydrogen
                    ),
                    use_denoise_unet=args.use_denoise_unet,
                    add_plots_full=args.add_plots_full,
                )
                elapsed = time.perf_counter() - t0
                res_path = output_dir / "res.json"
                with open(res_path, "w") as handle:
                    json.dump(result, handle, indent=2)
                if multi:
                    batch_stats.append(stats_from_run(image_path, output_dir, result))
                    print(f"--- {image_path} ---")
                print(f"Results saved to {res_path}")
                print(f"Processing time: {elapsed:.2f}s")
            except Exception:
                elapsed = time.perf_counter() - t0
                error = traceback.format_exc()
                failed.append((image_path, error))
                print(f"FAILED {image_path} ({elapsed:.2f}s):\n{error}")

    if multi and batch_stats:
        summary_path = write_batch_summary_md(summary_parent, batch_stats)
        print(f"Batch summary written to {summary_path}")
    if failed:
        print(f"\n{len(failed)} image(s) failed:")
        for img_path, _ in failed:
            print(f"  {img_path}")
