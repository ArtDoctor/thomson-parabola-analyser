import sys
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_data.spot_utils import create_randomized_instance as create_spot
from synthetic_data.synth_dataset_models import (
    BaseImagePlan,
    CleanTransform,
    ConstantReplacement,
    FEATURE_FLAGS,
    SIMULATION_RANGES,
    SimulationConstants,
    TRANSFORM_LIMITS,
    NoisePlan,
)
from synthetic_data.synth_dataset_noise_runtime import (
    apply_named_noise_effect,
    capture_noise_state,
    configure_noise_module,
    restore_noise_state,
    sample_noise_plan,
)
from synthetic_data.utils.hits_to_img import detector_limits_from_spot, hits_to_img


ROOT_DIR = Path(__file__).resolve().parent
THOMSON_OPTIMIZED_CPP = ROOT_DIR / "thomson_optimized.cpp"
THOMSON_SHARED_HEADER = ROOT_DIR / "thomson_shared.h"

# Edit these to control dataset generation:
NUM_SYNTHETIC_IMAGES = 90
CLEAN_IMAGES_PER_SYNTHETIC = 3
NOISY_IMAGES_PER_CLEAN = 2
OUTPUT_DIR = ROOT_DIR / "generated_dataset"
SEED = 42

AVAILABLE_SPECIES: tuple[str, ...] = (
    "H1",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "Si1",
    "Si2",
    "Si3",
    "Si4",
    "Si5",
    "Si6",
    "Si7",
    "Si8",
    "Si9",
    "Si10",
    "Si11",
    "Si12",
    "Si13",
    "Si14",
)


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    clean_dir = output_dir / "clean"
    noisy_dir = output_dir / "noisy"
    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    return clean_dir, noisy_dir


def sample_species(rng: random.Random) -> list[str]:
    species_count = SIMULATION_RANGES.species_count.sample(rng)
    sampled_species = rng.sample(list(AVAILABLE_SPECIES), species_count)
    sampled_species.sort()
    return sampled_species


def sample_constants(rng: random.Random) -> SimulationConstants:
    return SimulationConstants(
        k_min_mev=SIMULATION_RANGES.k_min_mev.sample(rng),
        k_max_ref_mev=SIMULATION_RANGES.k_max_ref_mev.sample(rng),
        kt_mev=SIMULATION_RANGES.kt_mev.sample(rng),
        count_power=SIMULATION_RANGES.count_power.sample(rng),
        spread_extra=SIMULATION_RANGES.spread_extra.sample(rng),
        energy_exponent=SIMULATION_RANGES.energy_exponent.sample(rng),
        angle_sigma=SIMULATION_RANGES.angle_sigma.sample(rng),
        beam_sigma=SIMULATION_RANGES.beam_sigma.sample(rng),
        n_particles=SIMULATION_RANGES.n_particles.sample(rng),
    )


def build_base_image_plan(rng: random.Random) -> BaseImagePlan:
    # Place the bright spot in the lower-left fan so the synthetic traces spread
    # across the detector area used by the runtime pipeline.
    dist = rng.randint(300, 400)
    return BaseImagePlan(
        species=sample_species(rng), 
        constants=sample_constants(rng),
        spot_col=750 - dist,
        spot_row=750 + dist,
    )


def format_float(value: float) -> str:
    return f"{value:.12g}"


def replace_constant(source: str, replacement: ConstantReplacement) -> str:
    pattern = rf"constexpr\s+{replacement.c_type}\s+{replacement.name}\s*=\s*[^;]+;"
    replaced, count = re.subn(
        pattern,
        f"constexpr {replacement.c_type} {replacement.name} = {replacement.value};",
        source,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Failed to replace constant {replacement.name} in thomson_shared.h.")
    return replaced


def build_modified_header(constants: SimulationConstants) -> str:
    header_text = THOMSON_SHARED_HEADER.read_text(encoding="utf-8")
    replacements = [
        ConstantReplacement(name="K_MIN_MEV", c_type="double", value=format_float(constants.k_min_mev)),
        ConstantReplacement(name="K_MAX_REF_MEV", c_type="double", value=format_float(constants.k_max_ref_mev)),
        ConstantReplacement(name="KT_MEV", c_type="double", value=format_float(constants.kt_mev)),
        ConstantReplacement(name="COUNT_POWER", c_type="double", value=format_float(constants.count_power)),
        ConstantReplacement(name="SPREAD_EXTRA", c_type="double", value=format_float(constants.spread_extra)),
        ConstantReplacement(name="ENERGY_EXPONENT", c_type="double", value=format_float(constants.energy_exponent)),
        ConstantReplacement(name="ANGLE_SIGMA", c_type="double", value=format_float(constants.angle_sigma)),
        ConstantReplacement(name="BEAM_SIGMA", c_type="double", value=format_float(constants.beam_sigma)),
        ConstantReplacement(name="N_PARTICLES", c_type="int", value=str(constants.n_particles)),
    ]
    updated_text = header_text
    for replacement in replacements:
        updated_text = replace_constant(updated_text, replacement)
    return updated_text


def compile_simulator(work_dir: Path, constants: SimulationConstants) -> Path:
    cpp_path = work_dir / "thomson_optimized.cpp"
    header_path = work_dir / "thomson_shared.h"
    binary_path = work_dir / "thomson_optimized"

    shutil.copy2(THOMSON_OPTIMIZED_CPP, cpp_path)
    header_path.write_text(build_modified_header(constants), encoding="utf-8")

    compile_cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-fopenmp",
        str(cpp_path.name),
        "-o",
        str(binary_path.name),
    ]
    subprocess.run(compile_cmd, cwd=work_dir, check=True)
    return binary_path


def run_simulator(binary_path: Path, species: Sequence[str], work_dir: Path) -> Path:
    hits_path = work_dir / "hits.txt"
    args: list[str] = [f"./{binary_path.name}", *FEATURE_FLAGS.to_args()]
    for particle in species:
        args.extend(["-particle", particle])

    with hits_path.open("w", encoding="utf-8") as output_handle:
        subprocess.run(args, cwd=work_dir, stdout=output_handle, check=True)
    return hits_path


def render_clean_base_image(hits_path: Path, brightness_scale: float, plan: BaseImagePlan) -> Image.Image:
    lim_x, lim_y = detector_limits_from_spot(plan.spot_col, plan.spot_row)
    return hits_to_img(
        str(hits_path),
        brightness_scale=brightness_scale,
        lim_x=lim_x,
        lim_y=lim_y,
    )


def sample_clean_transform(rng: random.Random) -> CleanTransform:
    return CleanTransform(
        rotation_degrees=rng.uniform(-TRANSFORM_LIMITS.rotation_degrees, TRANSFORM_LIMITS.rotation_degrees),
        tilt_x_degrees=rng.uniform(-TRANSFORM_LIMITS.tilt_degrees, TRANSFORM_LIMITS.tilt_degrees),
        tilt_y_degrees=rng.uniform(-TRANSFORM_LIMITS.tilt_degrees, TRANSFORM_LIMITS.tilt_degrees),
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_tilt_quad(
    width: int,
    height: int,
    transform: CleanTransform,
) -> tuple[float, float, float, float, float, float, float, float]:
    max_shift_x = width * 0.16
    max_shift_y = height * 0.16
    shift_x = math.sin(math.radians(transform.tilt_x_degrees)) * max_shift_x
    shift_y = math.sin(math.radians(transform.tilt_y_degrees)) * max_shift_y

    left_top_x = clamp(0.0 + shift_x, 0.0, width - 1.0)
    left_top_y = clamp(0.0 + shift_y, 0.0, height - 1.0)
    right_top_x = clamp(width - 1.0 - shift_x, 0.0, width - 1.0)
    right_top_y = clamp(0.0 - shift_y, 0.0, height - 1.0)
    right_bottom_x = clamp(width - 1.0 + shift_x, 0.0, width - 1.0)
    right_bottom_y = clamp(height - 1.0 - shift_y, 0.0, height - 1.0)
    left_bottom_x = clamp(0.0 - shift_x, 0.0, width - 1.0)
    left_bottom_y = clamp(height - 1.0 + shift_y, 0.0, height - 1.0)

    return (
        left_top_x,
        left_top_y,
        right_top_x,
        right_top_y,
        right_bottom_x,
        right_bottom_y,
        left_bottom_x,
        left_bottom_y,
    )


def apply_clean_transform(image: Image.Image, transform: CleanTransform) -> Image.Image:
    rotated = image.rotate(
        transform.rotation_degrees,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )
    quad = build_tilt_quad(rotated.width, rotated.height, transform)
    transformed = rotated.transform(
        rotated.size,
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )
    return transformed.convert("L")


def transform_spot_position(
    base_pos: tuple[int, int],
    base_size: tuple[int, int],
    transform: CleanTransform,
) -> tuple[int, int]:
    """Map a point from the base image to the transformed image.

    Uses a marker image to trace the point through the same transform as
    apply_clean_transform, guaranteeing exact correspondence.
    """
    base_w, base_h = base_size
    bx, by = base_pos
    marker = Image.new("L", (base_w, base_h), 0)
    marker.putpixel((bx, by), 255)
    transformed = apply_clean_transform(marker, transform)
    arr = np.array(transformed)
    out_pos = np.unravel_index(np.argmax(arr), arr.shape)
    return (int(out_pos[1]), int(out_pos[0]))


def overlay_spot(
    base_array: np.ndarray, spot_array: np.ndarray, position: tuple[int, int]
) -> np.ndarray:
    """Additively composite a spot onto a base image at the given (x, y) position.

    The position specifies where the *center* of the spot should land.
    Out-of-bounds regions are silently clipped.
    Uses a soft radial mask so the patch fades at edges and avoids a visible rectangle.
    """
    return _overlay_spot_impl(
        base_array, spot_array, position,
        subtract_background=0.0, soft_edge=True,
    )


def overlay_spot_noisy(
    base_array: np.ndarray, spot_array: np.ndarray, position: tuple[int, int]
) -> np.ndarray:
    """Overlay a noisy spot, subtracting its background to avoid a visible rectangle.

    The full spot has a ~45-level background blob; we only add the part above
    that threshold so the edges fade smoothly into the base image.
    """
    return _overlay_spot_impl(base_array, spot_array, position, subtract_background=50.0)


def _overlay_spot_impl(
    base_array: np.ndarray,
    spot_array: np.ndarray,
    position: tuple[int, int],
    subtract_background: float,
    soft_edge: bool = False,
) -> np.ndarray:
    bh, bw = base_array.shape[:2]
    sh, sw = spot_array.shape[:2]
    cx, cy = position

    sx_start = max(0, sw // 2 - cx)
    sy_start = max(0, sh // 2 - cy)
    dx_start = max(0, cx - sw // 2)
    dy_start = max(0, cy - sh // 2)

    w = min(sw - sx_start, bw - dx_start)
    h = min(sh - sy_start, bh - dy_start)
    if w <= 0 or h <= 0:
        return base_array

    spot_patch = spot_array[sy_start : sy_start + h, sx_start : sx_start + w].astype(
        np.float32
    )
    contribution = np.maximum(spot_patch - subtract_background, 0.0)

    if soft_edge:
        center_y, center_x = h / 2.0 - 0.5, w / 2.0 - 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        max_dist = min(center_x, center_y, w - 1 - center_x, h - 1 - center_y)
        fade_radius = max(max_dist * 0.85, 1.0)
        mask = np.maximum(0.0, 1.0 - dist / fade_radius).astype(np.float32)
        contribution *= mask

    result = base_array.copy()
    base_patch = result[dy_start : dy_start + h, dx_start : dx_start + w].astype(
        np.float32
    )
    result[dy_start : dy_start + h, dx_start : dx_start + w] = np.clip(
        base_patch + contribution, 0, 255
    ).astype(base_array.dtype)
    return result


def create_noisy_image(
    clean_image: Image.Image,
    plan: NoisePlan,
    seed: int,
    spot_noisy: np.ndarray | None = None,
    spot_position: tuple[int, int] | None = None,
) -> Image.Image:
    original_state = capture_noise_state()
    np.random.seed(seed)
    try:
        configure_noise_module(plan)
        noisy_array = np.array(clean_image, dtype=np.float32)
        for effect_name in plan.order:
            noisy_array = apply_named_noise_effect(effect_name, noisy_array, plan)
        final_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        # Overlay the noisy spot onto the noisy image (masked to avoid rectangle)
        if spot_noisy is not None and spot_position is not None:
            final_array = overlay_spot_noisy(final_array, spot_noisy, spot_position)
        return Image.fromarray(final_array, mode="L")
    finally:
        restore_noise_state(original_state)


CROP_PIXELS = 150


def save_image(image: Image.Image, path: Path) -> None:
    if image.width > 2 * CROP_PIXELS and image.height > 2 * CROP_PIXELS:
        image = image.crop(
            (CROP_PIXELS, CROP_PIXELS, image.width - CROP_PIXELS, image.height - CROP_PIXELS)
        )
    image.save(path)


def describe_constants(constants: SimulationConstants) -> str:
    return (
        f"k_min={constants.k_min_mev:.4f}, "
        f"k_max_ref={constants.k_max_ref_mev:.4f}, "
        f"kt={constants.kt_mev:.4f}, "
        f"count_power={constants.count_power:.4f}, "
        f"spread_extra={constants.spread_extra:.6f}, "
        f"energy_exponent={constants.energy_exponent:.4f}, "
        f"angle_sigma={constants.angle_sigma:.6f}, "
        f"beam_sigma={constants.beam_sigma:.6f}, "
        f"n_particles={constants.n_particles}"
    )


def generate_base_image(plan: BaseImagePlan, brightness_scale: float) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="synthetic_dataset_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        binary_path = compile_simulator(temp_dir, plan.constants)
        hits_path = run_simulator(binary_path, plan.species, temp_dir)
        return render_clean_base_image(hits_path, brightness_scale, plan)


# Spot position is dynamically fetched from BaseImagePlan


def main() -> None:
    clean_dir, noisy_dir = ensure_output_dirs(OUTPUT_DIR)
    rng = random.Random(SEED)

    clean_count = 0
    noisy_count = 0

    for base_index in range(NUM_SYNTHETIC_IMAGES):
        plan = build_base_image_plan(rng)
        brightness_scale = rng.uniform(0.2, 1.0)
        print(
            f"[base {base_index + 1}/{NUM_SYNTHETIC_IMAGES}] "
            f"species={','.join(plan.species)} | {describe_constants(plan.constants)} | brightness={brightness_scale:.2f}"
        )
        base_image = generate_base_image(plan, brightness_scale)

        for clean_index in range(CLEAN_IMAGES_PER_SYNTHETIC):
            transform = sample_clean_transform(rng)

            # Generate a beam-origin spot and align its bright core with the
            # tracked source position used through later transforms.
            spot_noisy, spot_clean, spot_center_local = create_spot()
            spot_position_base = (plan.spot_col, plan.spot_row)

            patch_center = (spot_clean.shape[1] // 2, spot_clean.shape[0] // 2)
            dx = spot_center_local[0] - patch_center[0]
            dy = spot_center_local[1] - patch_center[1]
            patch_place_pos = (
                spot_position_base[0] - dx,
                spot_position_base[1] - dy
            )
            
            base_with_spot = overlay_spot(
                np.array(base_image), spot_clean, patch_place_pos
            )
            clean_image = apply_clean_transform(
                Image.fromarray(base_with_spot, mode="L"), transform
            )

            spot_position = transform_spot_position(
                spot_position_base,
                (base_image.width, base_image.height),
                transform,
            )
            patch_place_pos_noisy = (
                spot_position[0] - dx,
                spot_position[1] - dy
            )

            clean_name = f"img_{base_index:04d}_clean_{clean_index:02d}.png"
            clean_path = clean_dir / clean_name
            save_image(clean_image, clean_path)
            clean_count += 1

            for noisy_index in range(NOISY_IMAGES_PER_CLEAN):
                noise_plan = sample_noise_plan(rng)
                noisy_seed = rng.randint(0, 2**31 - 1)
                noisy_image = create_noisy_image(
                    clean_image, noise_plan, noisy_seed,
                    spot_noisy=spot_noisy, spot_position=patch_place_pos_noisy,
                )
                noisy_name = (
                    f"img_{base_index:04d}_clean_{clean_index:02d}_noise_{noisy_index:02d}.png"
                )
                noisy_path = noisy_dir / noisy_name
                save_image(noisy_image, noisy_path)
                noisy_count += 1

    print(f"Generated {clean_count} clean images in {clean_dir}")
    print(f"Generated {noisy_count} noisy images in {noisy_dir}")


if __name__ == "__main__":
    main()
