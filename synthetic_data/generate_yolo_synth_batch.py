import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_data.generate_synthetic_dataset import (
    apply_clean_transform,
    build_base_image_plan,
    describe_constants,
    sample_clean_transform,
    transform_spot_position,
)
from synthetic_data.utils.hits_to_img import detector_limits_from_spot, hits_to_img


ROOT_DIR = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT_DIR / "synthetic_data"
GENERATE_IMAGE_SH = SYNTH_DIR / "generate_image.sh"
HEADER_PATH = SYNTH_DIR / "thomson_shared.h"
HITS_PATH = SYNTH_DIR / "hits.txt"

FULL_IMAGE_SIZE = 1200
SAFE_MARGIN = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate clean 1200x1200 synthetic detector images for YOLO using "
            "synthetic_data/generate_image.sh with H0 always included, plus "
            "random tilt/rotation/scale."
        )
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "synth_yolo",
    )
    parser.add_argument("--seed", type=int, default=20260403)
    return parser.parse_args()


def format_float(value: float) -> str:
    return f"{value:.12g}"


def replace_constant(source: str, name: str, c_type: str, value: str) -> str:
    pattern = rf"constexpr\s+{c_type}\s+{name}\s*=\s*[^;]+;"
    replaced, count = re.subn(
        pattern,
        f"constexpr {c_type} {name} = {value};",
        source,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Failed to replace constant {name} in {HEADER_PATH}.")
    return replaced


def build_header_text(original_header: str, constants: object) -> str:
    header_text = original_header
    replacements = [
        ("K_MIN_MEV", "double", format_float(constants.k_min_mev)),
        ("K_MAX_REF_MEV", "double", format_float(constants.k_max_ref_mev)),
        ("KT_MEV", "double", format_float(constants.kt_mev)),
        ("COUNT_POWER", "double", format_float(constants.count_power)),
        ("SPREAD_EXTRA", "double", format_float(constants.spread_extra)),
        ("ENERGY_EXPONENT", "double", format_float(constants.energy_exponent)),
        ("ANGLE_SIGMA", "double", format_float(constants.angle_sigma)),
        ("BEAM_SIGMA", "double", format_float(constants.beam_sigma)),
        ("N_PARTICLES", "int", str(constants.n_particles)),
    ]
    for name, c_type, value in replacements:
        header_text = replace_constant(header_text, name, c_type, value)
    return header_text


def particle_args(species: list[str]) -> list[str]:
    args: list[str] = []
    for particle in species:
        args.extend(["-particle", particle])
    return args


def with_required_species(species: list[str]) -> list[str]:
    return sorted({*species, "H0"})


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def render_clean_image(plan: object, brightness_scale: float) -> Image.Image:
    lim_x, lim_y = detector_limits_from_spot(plan.spot_col, plan.spot_row)
    return hits_to_img(
        str(HITS_PATH),
        brightness_scale=brightness_scale,
        lim_x=lim_x,
        lim_y=lim_y,
    )


def apply_scale_transform(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = image.resize((scaled_width, scaled_height), resample=Image.Resampling.BICUBIC)

    if scale >= 1.0:
        left = max(0, (scaled_width - width) // 2)
        top = max(0, (scaled_height - height) // 2)
        return resized.crop((left, top, left + width, top + height)).convert("L")

    canvas = Image.new("L", (width, height), 0)
    left = (width - scaled_width) // 2
    top = (height - scaled_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def transform_point(
    base_pos: tuple[int, int],
    base_size: tuple[int, int],
    transform: object,
    scale: float,
) -> tuple[int, int]:
    transformed = transform_spot_position(base_pos, base_size, transform)
    marker = Image.new("L", base_size, 0)
    marker.putpixel(transformed, 255)
    marker = apply_scale_transform(marker, scale)
    marker_array = np.array(marker)
    out_pos = np.unravel_index(np.argmax(marker_array), marker_array.shape)
    return (int(out_pos[1]), int(out_pos[0]))


def sample_scale(rng: random.Random) -> float:
    return rng.uniform(0.85, 1.15)


def sample_valid_transform(
    rng: random.Random,
    base_spot_pos: tuple[int, int],
    base_size: tuple[int, int],
) -> tuple[object, float, tuple[int, int]]:
    safe_min = SAFE_MARGIN
    safe_max_x = base_size[0] - SAFE_MARGIN
    safe_max_y = base_size[1] - SAFE_MARGIN

    for _ in range(64):
        transform = sample_clean_transform(rng)
        scale = sample_scale(rng)
        transformed_spot = transform_point(base_spot_pos, base_size, transform, scale)
        if safe_min <= transformed_spot[0] <= safe_max_x and safe_min <= transformed_spot[1] <= safe_max_y:
            return transform, scale, transformed_spot

    transform = sample_clean_transform(rng)
    scale = 1.0
    transformed_spot = transform_point(base_spot_pos, base_size, transform, scale)
    return transform, scale, transformed_spot


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    prepare_output_dir(args.output_dir)
    manifest_path = args.output_dir / "manifest.jsonl"

    original_header = HEADER_PATH.read_text(encoding="utf-8")

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_handle:
            for index in range(args.count):
                plan = build_base_image_plan(rng)
                species = with_required_species(plan.species)
                brightness_scale = rng.uniform(0.2, 1.0)
                spot_position_base = (plan.spot_col, plan.spot_row)
                transform, scale, transformed_spot = sample_valid_transform(
                    rng,
                    spot_position_base,
                    (FULL_IMAGE_SIZE, FULL_IMAGE_SIZE),
                )

                HEADER_PATH.write_text(
                    build_header_text(original_header, plan.constants),
                    encoding="utf-8",
                )
                subprocess.run(
                    ["bash", str(GENERATE_IMAGE_SH), *particle_args(species)],
                    cwd=ROOT_DIR,
                    check=True,
                )

                image = render_clean_image(plan, brightness_scale)
                image = apply_clean_transform(image, transform)
                image = apply_scale_transform(image, scale)

                filename = f"img_{index:04d}.png"
                image.save(args.output_dir / filename)

                record = {
                    "index": index,
                    "file": filename,
                    "script": str(GENERATE_IMAGE_SH),
                    "noise": False,
                    "species": species,
                    "constants": {
                        "k_min_mev": plan.constants.k_min_mev,
                        "k_max_ref_mev": plan.constants.k_max_ref_mev,
                        "kt_mev": plan.constants.kt_mev,
                        "count_power": plan.constants.count_power,
                        "spread_extra": plan.constants.spread_extra,
                        "energy_exponent": plan.constants.energy_exponent,
                        "angle_sigma": plan.constants.angle_sigma,
                        "beam_sigma": plan.constants.beam_sigma,
                        "n_particles": plan.constants.n_particles,
                    },
                    "brightness_scale": brightness_scale,
                    "transform": {
                        "rotation_degrees": transform.rotation_degrees,
                        "tilt_x_degrees": transform.tilt_x_degrees,
                        "tilt_y_degrees": transform.tilt_y_degrees,
                        "scale": scale,
                    },
                    "detector_spot": {
                        "col": plan.spot_col,
                        "row": plan.spot_row,
                    },
                    "output_spot": {
                        "col": transformed_spot[0],
                        "row": transformed_spot[1],
                    },
                }
                manifest_handle.write(json.dumps(record) + "\n")

                print(
                    f"[{index + 1}/{args.count}] {filename} | "
                    f"species={','.join(species)} | "
                    f"{describe_constants(plan.constants)} | "
                    f"brightness={brightness_scale:.2f} | "
                    f"rotation={transform.rotation_degrees:.2f} | "
                    f"tilt_x={transform.tilt_x_degrees:.2f} | "
                    f"tilt_y={transform.tilt_y_degrees:.2f} | "
                    f"scale={scale:.3f} | "
                    f"spot=({transformed_spot[0]},{transformed_spot[1]})"
                )
    finally:
        HEADER_PATH.write_text(original_header, encoding="utf-8")


if __name__ == "__main__":
    main()
