import random

from pydantic import BaseModel, Field


class FloatRange(BaseModel):
    minimum: float
    maximum: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.minimum, self.maximum)


class IntRange(BaseModel):
    minimum: int
    maximum: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.minimum, self.maximum)


class SimulationConstants(BaseModel):
    k_min_mev: float
    k_max_ref_mev: float
    kt_mev: float
    count_power: float
    spread_extra: float
    energy_exponent: float
    angle_sigma: float
    beam_sigma: float
    n_particles: int


class SimulationRanges(BaseModel):
    k_min_mev: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.008, maximum=0.010))
    k_max_ref_mev: FloatRange = Field(default_factory=lambda: FloatRange(minimum=1.65, maximum=1.75))
    kt_mev: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.34, maximum=0.38))
    count_power: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.09, maximum=0.11))
    spread_extra: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.00065, maximum=0.00105))
    energy_exponent: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.29, maximum=0.31))
    angle_sigma: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.00022, maximum=0.00055))
    beam_sigma: FloatRange = Field(default_factory=lambda: FloatRange(minimum=0.00022, maximum=0.00055))
    n_particles: IntRange = Field(default_factory=lambda: IntRange(minimum=190000, maximum=210000))
    species_count: IntRange = Field(default_factory=lambda: IntRange(minimum=5, maximum=20))


class FeatureFlags(BaseModel):
    beam: bool = False
    adaptive: bool = False
    fringe: bool = False
    relativistic: bool = False

    def to_args(self) -> list[str]:
        args: list[str] = []
        if self.beam:
            args.append("-beam")
        if self.adaptive:
            args.append("-adaptive")
        if self.fringe:
            args.append("-fringe")
        if self.relativistic:
            args.append("-relativistic")
        return args


class TransformLimits(BaseModel):
    rotation_degrees: float = 20.0
    tilt_degrees: float = 10.0


class WhiteSpotsConfig(BaseModel):
    num_spots: int
    size_max: int
    brightness_min: int
    brightness_max: int
    light_vector: tuple[float, float, float]


class BlackSpotsConfig(BaseModel):
    num_spots: int
    area_min: int
    area_max: int
    darken_factor: float
    edge_smoothness: float


class VerticalLinesConfig(BaseModel):
    period: int
    thickness: int
    intensity: float
    smoothness: float


class PerlinDarkeningConfig(BaseModel):
    blur_sigma: float
    darken_strength: float


class PerlinWhiteningConfig(BaseModel):
    blur_sigma: float
    whiten_strength: float


class GeneralNoiseConfig(BaseModel):
    mean: float
    std: float


class PedestalConfig(BaseModel):
    level: float
    variation_sigma: float
    variation_strength: float


class GradientBlobsConfig(BaseModel):
    num_blobs: int
    sigma_min: float
    sigma_max: float
    intensity_min: float
    intensity_max: float


class NoisePlan(BaseModel):
    order: list[str]
    white_spots: WhiteSpotsConfig
    black_spots: BlackSpotsConfig
    vertical_lines: VerticalLinesConfig
    perlin_darkening: PerlinDarkeningConfig
    perlin_whitening: PerlinWhiteningConfig
    general_noise: GeneralNoiseConfig
    pedestal: PedestalConfig
    gradient_blobs: GradientBlobsConfig


class NoiseModuleState(BaseModel):
    ws_num_spots: int
    ws_size_max: int
    ws_brightness_range: tuple[int, int]
    bs_num_spots: int
    bs_area_range: tuple[int, int]
    bs_darken_factor: float
    bs_edge_smoothness: float
    vl_period: int
    vl_thickness: int
    vl_intensity: float
    vl_smoothness: float
    pl_blur_sigma: float
    pl_darken_strength: float
    pl_white_blur_sigma: float
    pl_whiten_strength: float
    gn_mean: float
    gn_std: float
    bp_level: float
    bp_variation_sigma: float
    bp_variation_strength: float
    gb_num_blobs: int
    gb_sigma_range: tuple[float, float]
    gb_intensity_range: tuple[float, float]


class BaseImagePlan(BaseModel):
    species: list[str]
    constants: SimulationConstants
    spot_col: int
    spot_row: int


class CleanTransform(BaseModel):
    rotation_degrees: float
    tilt_x_degrees: float
    tilt_y_degrees: float


class ConstantReplacement(BaseModel):
    name: str
    c_type: str
    value: str


SIMULATION_RANGES = SimulationRanges()
FEATURE_FLAGS = FeatureFlags()
TRANSFORM_LIMITS = TransformLimits()
