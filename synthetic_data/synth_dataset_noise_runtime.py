import random

import numpy as np

from synthetic_data.synth_dataset_models import (
    BlackSpotsConfig,
    GeneralNoiseConfig,
    GradientBlobsConfig,
    NoiseModuleState,
    NoisePlan,
    PedestalConfig,
    PerlinDarkeningConfig,
    PerlinWhiteningConfig,
    VerticalLinesConfig,
    WhiteSpotsConfig,
)
from synthetic_data.utils import noise as noise_module


def sample_light_vector(rng: random.Random) -> tuple[float, float, float]:
    return (
        rng.uniform(-1.0, 1.0),
        rng.uniform(-1.0, 1.0),
        rng.uniform(0.35, 1.0),
    )


def sample_noise_plan(rng: random.Random) -> NoisePlan:
    effect_order = [
        "white_spots",
        "black_spots",
        "vertical_lines",
        "perlin_darkening",
        "perlin_whitening",
        "general_noise",
        "gradient_blobs",
    ]
    rng.shuffle(effect_order)
    effect_order.append("pedestal")
    brightness_min = rng.randint(10, 50)
    brightness_max = rng.randint(max(brightness_min + 10, 70), 180)
    area_min = rng.randint(120, 800)
    area_max = rng.randint(max(area_min + 500, 4000), 13000)
    gb_sigma_min = rng.uniform(200, 400)
    gb_sigma_max = rng.uniform(max(gb_sigma_min + 50, 400), 600)
    gb_intensity_min = rng.uniform(5, 15)
    gb_intensity_max = rng.uniform(max(gb_intensity_min + 5, 20), 40)
    return NoisePlan(
        order=effect_order,
        white_spots=WhiteSpotsConfig(
            num_spots=rng.randint(1600, 3600),
            size_max=rng.randint(1, 3),
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            light_vector=sample_light_vector(rng),
        ),
        black_spots=BlackSpotsConfig(
            num_spots=rng.randint(20, 70),
            area_min=area_min,
            area_max=area_max,
            darken_factor=rng.uniform(0.25, 0.7),
            edge_smoothness=rng.uniform(3.0, 8.0),
        ),
        vertical_lines=VerticalLinesConfig(
            period=rng.randint(8, 18),
            thickness=rng.randint(1, 6),
            intensity=rng.uniform(6.0, 24.0),
            smoothness=rng.uniform(3.0, 10.0),
        ),
        perlin_darkening=PerlinDarkeningConfig(
            blur_sigma=rng.uniform(16.0, 38.0),
            darken_strength=rng.uniform(0.2, 0.8),
        ),
        perlin_whitening=PerlinWhiteningConfig(
            blur_sigma=rng.uniform(18.0, 45.0),
            whiten_strength=rng.uniform(8.0, 28.0),
        ),
        general_noise=GeneralNoiseConfig(
            mean=rng.uniform(0.0, 5.0),
            std=rng.uniform(5.0, 35.0),
        ),
        pedestal=PedestalConfig(
            level=rng.uniform(15.0, 50.0),
            variation_sigma=rng.uniform(60.0, 120.0),
            variation_strength=rng.uniform(3.0, 12.0),
        ),
        gradient_blobs=GradientBlobsConfig(
            num_blobs=rng.randint(1, 3),
            sigma_min=gb_sigma_min,
            sigma_max=gb_sigma_max,
            intensity_min=gb_intensity_min,
            intensity_max=gb_intensity_max,
        ),
    )


def capture_noise_state() -> NoiseModuleState:
    return NoiseModuleState(
        ws_num_spots=noise_module.WS_NUM_SPOTS,
        ws_size_max=noise_module.WS_SIZE_MAX,
        ws_brightness_range=noise_module.WS_BRIGHTNESS_RANGE,
        bs_num_spots=noise_module.BS_NUM_SPOTS,
        bs_area_range=noise_module.BS_AREA_RANGE,
        bs_darken_factor=noise_module.BS_DARKEN_FACTOR,
        bs_edge_smoothness=noise_module.BS_EDGE_SMOOTHNESS,
        vl_period=noise_module.VL_PERIOD,
        vl_thickness=noise_module.VL_THICKNESS,
        vl_intensity=noise_module.VL_INTENSITY,
        vl_smoothness=noise_module.VL_SMOOTHNESS,
        pl_blur_sigma=noise_module.PL_BLUR_SIGMA,
        pl_darken_strength=noise_module.PL_DARKEN_STRENGTH,
        pl_white_blur_sigma=noise_module.PL_WHITE_BLUR_SIGMA,
        pl_whiten_strength=noise_module.PL_WHITEN_STRENGTH,
        gn_mean=noise_module.GN_MEAN,
        gn_std=noise_module.GN_STD,
        bp_level=noise_module.BP_LEVEL,
        bp_variation_sigma=noise_module.BP_VARIATION_SIGMA,
        bp_variation_strength=noise_module.BP_VARIATION_STRENGTH,
        gb_num_blobs=noise_module.GB_NUM_BLOBS,
        gb_sigma_range=noise_module.GB_SIGMA_RANGE,
        gb_intensity_range=noise_module.GB_INTENSITY_RANGE,
    )


def restore_noise_state(state: NoiseModuleState) -> None:
    noise_module.WS_NUM_SPOTS = state.ws_num_spots
    noise_module.WS_SIZE_MAX = state.ws_size_max
    noise_module.WS_BRIGHTNESS_RANGE = state.ws_brightness_range
    noise_module.BS_NUM_SPOTS = state.bs_num_spots
    noise_module.BS_AREA_RANGE = state.bs_area_range
    noise_module.BS_DARKEN_FACTOR = state.bs_darken_factor
    noise_module.BS_EDGE_SMOOTHNESS = state.bs_edge_smoothness
    noise_module.VL_PERIOD = state.vl_period
    noise_module.VL_THICKNESS = state.vl_thickness
    noise_module.VL_INTENSITY = state.vl_intensity
    noise_module.VL_SMOOTHNESS = state.vl_smoothness
    noise_module.PL_BLUR_SIGMA = state.pl_blur_sigma
    noise_module.PL_DARKEN_STRENGTH = state.pl_darken_strength
    noise_module.PL_WHITE_BLUR_SIGMA = state.pl_white_blur_sigma
    noise_module.PL_WHITEN_STRENGTH = state.pl_whiten_strength
    noise_module.GN_MEAN = state.gn_mean
    noise_module.GN_STD = state.gn_std
    noise_module.BP_LEVEL = state.bp_level
    noise_module.BP_VARIATION_SIGMA = state.bp_variation_sigma
    noise_module.BP_VARIATION_STRENGTH = state.bp_variation_strength
    noise_module.GB_NUM_BLOBS = state.gb_num_blobs
    noise_module.GB_SIGMA_RANGE = state.gb_sigma_range
    noise_module.GB_INTENSITY_RANGE = state.gb_intensity_range


def configure_noise_module(plan: NoisePlan) -> None:
    noise_module.WS_NUM_SPOTS = plan.white_spots.num_spots
    noise_module.WS_SIZE_MAX = plan.white_spots.size_max
    noise_module.WS_BRIGHTNESS_RANGE = (
        plan.white_spots.brightness_min,
        plan.white_spots.brightness_max,
    )
    noise_module.BS_NUM_SPOTS = plan.black_spots.num_spots
    noise_module.BS_AREA_RANGE = (plan.black_spots.area_min, plan.black_spots.area_max)
    noise_module.BS_DARKEN_FACTOR = plan.black_spots.darken_factor
    noise_module.BS_EDGE_SMOOTHNESS = plan.black_spots.edge_smoothness
    noise_module.VL_PERIOD = plan.vertical_lines.period
    noise_module.VL_THICKNESS = plan.vertical_lines.thickness
    noise_module.VL_INTENSITY = plan.vertical_lines.intensity
    noise_module.VL_SMOOTHNESS = plan.vertical_lines.smoothness
    noise_module.PL_BLUR_SIGMA = plan.perlin_darkening.blur_sigma
    noise_module.PL_DARKEN_STRENGTH = plan.perlin_darkening.darken_strength
    noise_module.PL_WHITE_BLUR_SIGMA = plan.perlin_whitening.blur_sigma
    noise_module.PL_WHITEN_STRENGTH = plan.perlin_whitening.whiten_strength
    noise_module.GN_MEAN = plan.general_noise.mean
    noise_module.GN_STD = plan.general_noise.std
    noise_module.BP_LEVEL = plan.pedestal.level
    noise_module.BP_VARIATION_SIGMA = plan.pedestal.variation_sigma
    noise_module.BP_VARIATION_STRENGTH = plan.pedestal.variation_strength
    noise_module.GB_NUM_BLOBS = plan.gradient_blobs.num_blobs
    noise_module.GB_SIGMA_RANGE = (plan.gradient_blobs.sigma_min, plan.gradient_blobs.sigma_max)
    noise_module.GB_INTENSITY_RANGE = (plan.gradient_blobs.intensity_min, plan.gradient_blobs.intensity_max)


def apply_named_noise_effect(effect_name: str, array: np.ndarray, plan: NoisePlan) -> np.ndarray:
    if effect_name == "white_spots":
        return noise_module.apply_white_spots(array, light_vector=plan.white_spots.light_vector)
    if effect_name == "black_spots":
        return noise_module.apply_black_spots(array)
    if effect_name == "vertical_lines":
        return noise_module.apply_vertical_lines(array)
    if effect_name == "perlin_darkening":
        return noise_module.apply_perlin_darkening(array)
    if effect_name == "perlin_whitening":
        return noise_module.apply_perlin_whitening(array)
    if effect_name == "general_noise":
        return noise_module.apply_general_noise(array)
    if effect_name == "pedestal":
        return noise_module.apply_baseline_pedestal(array)
    if effect_name == "gradient_blobs":
        return noise_module.apply_gradient_blobs(array)
    raise ValueError(f"Unsupported noise effect: {effect_name}")
