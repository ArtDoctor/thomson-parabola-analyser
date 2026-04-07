"""Generate the eval_synth dataset: 30 synthetic Thomson parabola images with ground truth.

Usage:
    cd synthetic_data
    python generate_eval_synth.py

Output goes to <repo_root>/eval_synth/ with one PNG + one JSON per image.

Rendered PNGs are larger than the raw 1200×1200 histogram: a black margin is added
before rotation/tilt so the synthetic bright spot (near a detector edge after
histogram placement) stays inside the frame. Final size is roughly
(1200 + 2*PRE_TRANSFORM_MARGIN_PX - 2*EVAL_SAVE_BORDER_CROP_PX) per side.
"""
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_data.generate_synthetic_dataset import (
    CleanTransform,
    NoisePlan,
    apply_clean_transform,
    create_noisy_image,
    overlay_spot,
    sample_noise_plan,
    transform_spot_position,
)
from synthetic_data.spot_utils import create_randomized_instance as create_spot
from synthetic_data.utils.hits_to_img import detector_limits_from_spot, hits_to_img

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "eval_synth"

THOMSON_CPP = ROOT_DIR / "thomson_optimized.cpp"
THOMSON_HEADER = ROOT_DIR / "thomson_shared.h"

NUM_IMAGES = 30
SEED = 123

# Padding before rotation/tilt keeps the beam-origin spot in-frame (it is placed
# using histogram coords that skew toward one detector corner; PIL rotate clips
# when expand=False).
PRE_TRANSFORM_MARGIN_PX: int = 320
# Lighter border crop than training `save_image` (150px); margin already reduces
# edge clipping from the warp.
EVAL_SAVE_BORDER_CROP_PX: int = 80

AVAILABLE_SPECIES: tuple[str, ...] = (
    "H1",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "Si1", "Si2", "Si3", "Si4", "Si5", "Si6", "Si7",
    "Si8", "Si9", "Si10", "Si11", "Si12", "Si13", "Si14",
)

# Mass numbers used by the simulator (must match thomson_shared.h lookup_mass_number)
MASS_NUMBER: dict[str, int] = {"H": 1, "C": 12, "Si": 28}

# For m/q computation (matching utils/species.py)
M_U = 1.66053906660e-27  # unified atomic mass unit, kg
M_P = 1.67262192369e-27  # proton mass, kg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sim_label_to_pipeline(label: str) -> tuple[str, str, int, int]:
    """Convert simulator label like 'C4' to pipeline info.

    Returns (pipeline_label, symbol, charge, mass_number).
    E.g. 'C4' -> ('C^4+', 'C', 4, 12).
    """
    # Find where digits start
    for i, ch in enumerate(label):
        if ch.isdigit():
            symbol = label[:i]
            charge = int(label[i:])
            break
    else:
        symbol = label
        charge = 1

    mass_num = MASS_NUMBER[symbol]
    pipeline_label = f"{symbol}^{charge}+"
    return pipeline_label, symbol, charge, mass_num


def compute_mq(mass_number: int, charge: int, symbol: str) -> float:
    """Compute mass-over-charge ratio matching utils/species.py convention."""
    if symbol == "H" and mass_number == 1:
        return 1.0  # reference: H+ is defined as 1.0
    return (mass_number * M_U) / (charge * M_P)


def compile_simulator(work_dir: Path) -> Path:
    """Compile the Thomson simulator once with default (unmodified) constants."""
    cpp_dst = work_dir / "thomson_optimized.cpp"
    header_dst = work_dir / "thomson_shared.h"
    binary = work_dir / "thomson_optimized"

    shutil.copy2(THOMSON_CPP, cpp_dst)
    shutil.copy2(THOMSON_HEADER, header_dst)

    subprocess.run(
        ["g++", "-O3", "-march=native", "-fopenmp",
         cpp_dst.name, "-o", binary.name],
        cwd=work_dir, check=True,
    )
    return binary


def run_simulator(binary: Path, species: list[str], work_dir: Path) -> Path:
    """Run the simulator for given species, return path to hits.txt."""
    hits_path = work_dir / "hits.txt"
    args = [f"./{binary.name}"]
    for sp in species:
        args.extend(["-particle", sp])

    with hits_path.open("w", encoding="utf-8") as fh:
        subprocess.run(args, cwd=work_dir, stdout=fh, check=True)
    return hits_path


def parse_hits_energy_stats(hits_path: Path) -> dict[str, dict[str, float]]:
    """Parse hits.txt and return per-species energy statistics."""
    df = pd.read_csv(
        hits_path, sep=r"\s+",
        names=["Y", "X", "Energy", "ParticleName"],
    )
    stats: dict[str, dict[str, float]] = {}
    for name, group in df.groupby("ParticleName"):
        energies = group["Energy"].values
        stats[str(name)] = {
            "energy_min_mev": float(np.min(energies)),
            "energy_max_mev": float(np.max(energies)),
            "energy_mean_mev": float(np.mean(energies)),
            "energy_median_mev": float(np.median(energies)),
        }
    return stats


def sample_species(rng: random.Random) -> list[str]:
    """Pick 5-15 random species (no H0)."""
    count = rng.randint(5, 15)
    selected = rng.sample(list(AVAILABLE_SPECIES), count)
    selected.sort()
    return selected


def sample_transform(rng: random.Random) -> CleanTransform:
    """Sample rotation (±20°) and tilt (3-15° magnitude, random sign)."""
    rotation = rng.uniform(-20.0, 20.0)

    tilt_x_mag = rng.uniform(3.0, 15.0)
    tilt_y_mag = rng.uniform(3.0, 15.0)
    tilt_x = tilt_x_mag * rng.choice([-1, 1])
    tilt_y = tilt_y_mag * rng.choice([-1, 1])

    return CleanTransform(
        rotation_degrees=rotation,
        tilt_x_degrees=tilt_x,
        tilt_y_degrees=tilt_y,
    )


def sample_spot_position(rng: random.Random) -> tuple[int, int]:
    """Sample spot position matching generate_synthetic_dataset convention."""
    dist = rng.randint(300, 400)
    return 750 - dist, 750 + dist  # (col, row)


def scale_noise_plan_intensity(plan: NoisePlan, factor: float) -> NoisePlan:
    """Scale all noise strengths by a shared factor."""
    plan.white_spots.num_spots = max(1, int(plan.white_spots.num_spots * factor))
    plan.white_spots.brightness_min = max(1, int(plan.white_spots.brightness_min * factor))
    plan.white_spots.brightness_max = max(
        plan.white_spots.brightness_min + 1,
        int(plan.white_spots.brightness_max * factor),
    )
    plan.black_spots.num_spots = max(1, int(plan.black_spots.num_spots * factor))
    plan.vertical_lines.intensity *= factor
    plan.perlin_darkening.darken_strength *= factor
    plan.perlin_whitening.whiten_strength *= factor
    plan.general_noise.std *= factor
    plan.pedestal.level *= factor
    plan.pedestal.variation_strength *= factor
    plan.gradient_blobs.intensity_min *= factor
    plan.gradient_blobs.intensity_max *= factor
    return plan


def _pad_grayscale_array(arr: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return arr
    return np.pad(
        arr,
        ((margin_px, margin_px), (margin_px, margin_px)),
        mode="constant",
        constant_values=0,
    )


def save_eval_synth_image(image: Image.Image, path: Path) -> None:
    crop = EVAL_SAVE_BORDER_CROP_PX
    if image.width > 2 * crop and image.height > 2 * crop:
        image = image.crop((crop, crop, image.width - crop, image.height - crop))
    image.save(path)


def build_ground_truth_json(
    species_sim: list[str],
    energy_stats: dict[str, dict[str, float]],
    transform: CleanTransform,
    noisy: bool,
    spot_col: int,
    spot_row: int,
    seed: int,
    noise_intensity: float | None = None,
) -> dict:
    """Build the ground truth JSON dict for one image."""
    species_list = []
    for sim_label in species_sim:
        pipeline_label, symbol, charge, mass_num = sim_label_to_pipeline(sim_label)
        mq = compute_mq(mass_num, charge, symbol)
        entry: dict = {
            "label": pipeline_label,
            "symbol": symbol,
            "charge": charge,
            "mass_number": mass_num,
            "m_over_q": round(mq, 6),
        }
        if sim_label in energy_stats:
            entry.update(energy_stats[sim_label])
        species_list.append(entry)

    # Sort by m/q for readability
    species_list.sort(key=lambda s: s["m_over_q"])

    result: dict = {
        "species": species_list,
        "transform": {
            "rotation_degrees": round(transform.rotation_degrees, 2),
            "tilt_x_degrees": round(transform.tilt_x_degrees, 2),
            "tilt_y_degrees": round(transform.tilt_y_degrees, 2),
        },
        "noisy": noisy,
        "spot_col": spot_col,
        "spot_row": spot_row,
        "seed": seed,
    }
    if noise_intensity is not None:
        result["noise_intensity"] = round(noise_intensity, 3)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    rng = random.Random(SEED)

    # Decide which images are noisy (roughly half)
    noisy_flags = [False] * NUM_IMAGES
    noisy_indices = rng.sample(range(NUM_IMAGES), NUM_IMAGES // 2)
    for i in noisy_indices:
        noisy_flags[i] = True

    with tempfile.TemporaryDirectory(prefix="eval_synth_") as tmp:
        tmp_dir = Path(tmp)
        print("Compiling simulator (once)...")
        binary = compile_simulator(tmp_dir)

        for idx in range(NUM_IMAGES):
            is_noisy = noisy_flags[idx]
            tag = "noisy" if is_noisy else "clean"
            img_name = f"synth_{idx + 1:04d}_{tag}"

            # 1) Sample species
            species = sample_species(rng)

            # 2) Run simulator
            print(f"[{idx + 1}/{NUM_IMAGES}] species={','.join(species)}  ({tag})")
            hits_path = run_simulator(binary, species, tmp_dir)

            # 3) Parse energy stats from hits
            energy_stats = parse_hits_energy_stats(hits_path)

            # 4) Render base image
            spot_col, spot_row = sample_spot_position(rng)
            lim_x, lim_y = detector_limits_from_spot(spot_col, spot_row)
            brightness = rng.uniform(0.4, 1.0)
            base_image = hits_to_img(
                str(hits_path), brightness_scale=brightness,
                lim_x=lim_x, lim_y=lim_y,
            )

            # 5) Overlay synthetic spot (clean version)
            spot_noisy_arr, spot_clean_arr, spot_center_local = create_spot()
            spot_pos_base = (spot_col, spot_row)

            # Compensate for spot core offset (same logic as generate_synthetic_dataset)
            patch_center = (spot_clean_arr.shape[1] // 2, spot_clean_arr.shape[0] // 2)
            dx = spot_center_local[0] - patch_center[0]
            dy = spot_center_local[1] - patch_center[1]
            patch_place_pos = (spot_pos_base[0] - dx, spot_pos_base[1] - dy)

            base_with_spot = overlay_spot(
                np.array(base_image), spot_clean_arr, patch_place_pos,
            )

            margin = PRE_TRANSFORM_MARGIN_PX
            padded_arr = _pad_grayscale_array(base_with_spot, margin)
            spot_for_transform = (
                spot_pos_base[0] + margin,
                spot_pos_base[1] + margin,
            )
            padded_w = int(padded_arr.shape[1])
            padded_h = int(padded_arr.shape[0])

            # 6) Apply rotation + tilt
            transform = sample_transform(rng)
            clean_image = apply_clean_transform(
                Image.fromarray(padded_arr, mode="L"), transform,
            )

            # Track spot position through transform
            spot_position = transform_spot_position(
                spot_for_transform,
                (padded_w, padded_h),
                transform,
            )

            # 7) Apply noise if flagged
            if is_noisy:
                noise_intensity = rng.uniform(0.3, 0.8)
                noise_plan = sample_noise_plan(rng)
                noise_plan = scale_noise_plan_intensity(noise_plan, noise_intensity)
                noise_seed = rng.randint(0, 2**31 - 1)

                # Noisy spot overlay position
                patch_place_pos_noisy = (
                    spot_position[0] - dx,
                    spot_position[1] - dy,
                )
                final_image = create_noisy_image(
                    clean_image, noise_plan, noise_seed,
                    spot_noisy=spot_noisy_arr,
                    spot_position=patch_place_pos_noisy,
                )
            else:
                noise_intensity = None
                final_image = clean_image

            # 8) Save image (gentle border crop; see EVAL_SAVE_BORDER_CROP_PX)
            img_path = OUTPUT_DIR / f"{img_name}.png"
            save_eval_synth_image(final_image, img_path)

            # 9) Save ground truth JSON
            gt = build_ground_truth_json(
                species_sim=species,
                energy_stats=energy_stats,
                transform=transform,
                noisy=is_noisy,
                spot_col=spot_col,
                spot_row=spot_row,
                seed=SEED + idx,
                noise_intensity=noise_intensity,
            )
            json_path = OUTPUT_DIR / f"{img_name}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(gt, f, indent=2)

    n_clean = sum(1 for f in noisy_flags if not f)
    n_noisy = sum(1 for f in noisy_flags if f)
    print(f"\nDone! Generated {NUM_IMAGES} images ({n_clean} clean, {n_noisy} noisy)")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
