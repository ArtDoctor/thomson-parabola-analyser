import numpy as np
import scipy.ndimage as ndimage

# Use FFT-based blur for large sigma (O(n log n) vs O(n*sigma) for spatial)
_GAUSSIAN_FFT_SIGMA_THRESHOLD = 12.0


def _gaussian_filter_fast(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= _GAUSSIAN_FFT_SIGMA_THRESHOLD:
        return ndimage.gaussian_filter(arr, sigma=sigma)
    f = np.fft.fft2(arr)
    f = ndimage.fourier_gaussian(f, sigma)
    out = np.fft.ifft2(f).real
    return out.astype(arr.dtype, copy=False)


# ==========================================
#         EDITABLE PARAMETERS
# ==========================================

# 1. Small White Spots Parameters
WS_NUM_SPOTS = 2800
WS_SIZE_MAX = 2                  # Maximum pixel radius 
WS_BRIGHTNESS_RANGE = (20, 150)  # Range of added brightness

# 2. Big Black Spots Parameters
BS_NUM_SPOTS = 45
BS_AREA_RANGE = (200, 10000)    # Target area in pixels
BS_DARKEN_FACTOR = 0.45         # 1.0 = pure black, 0.0 = no effect
BS_EDGE_SMOOTHNESS = 5.0        # Sigma for blurring the spot edges

# 3. Vertical Lines Parameters
VL_PERIOD = 10                  # Total distance between the start of each line (pixels)
VL_THICKNESS = 4                # Thickness of the core line (pixels)
VL_INTENSITY = 15.0             # Additive intensity
VL_SMOOTHNESS = 7.0             # Sigma for smoothing the transition between line and dark space

# 4. Perlin-like Noise Parameters
PL_BLUR_SIGMA = 25.0            # Higher sigma = larger dark noise blobs
PL_DARKEN_STRENGTH = 0.5        # 1.0 = strong darkening, 0.0 = no darkening
PL_WHITE_BLUR_SIGMA = 30.0      # Higher sigma = larger bright noise blobs
PL_WHITEN_STRENGTH = 15.0       # Max brightness added by the white noise

# 5. General Sensor Noise Parameters
GN_MEAN = 0.0                   # Mean of the noise (keep at 0 to avoid shifting overall brightness)
GN_STD = 16.0                   # Standard deviation (Higher = stronger grain)

# 6. Baseline Pedestal Parameters
BP_LEVEL = 25.0                 # Constant offset added to the whole image (simulates sensor read-noise floor)
BP_VARIATION_SIGMA = 80.0       # Sigma for slow spatial variation of the pedestal
BP_VARIATION_STRENGTH = 8.0     # Amplitude of the slow spatial variation

# 7. Gradient Blob Parameters
GB_NUM_BLOBS = 2                # Number of large Gaussian blobs to add
GB_SIGMA_RANGE: tuple[float, float] = (200.0, 600.0)   # Range for blob sigma (pixels)
GB_INTENSITY_RANGE: tuple[float, float] = (10.0, 40.0) # Range for blob peak intensity (gray levels)
# ==========================================


def apply_white_spots(arr: np.ndarray, light_vector: tuple[float, float, float] = (-0.5, -0.5, 1.0)) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    h, w = out.shape
    lv = np.array(light_vector, dtype=np.float32) / np.linalg.norm(light_vector)

    n = WS_NUM_SPOTS
    x = np.random.randint(0, w, size=n)
    y = np.random.randint(0, h, size=n)
    radius = np.random.randint(1, WS_SIZE_MAX + 1, size=n)
    brightness = np.random.randint(
        WS_BRIGHTNESS_RANGE[0], WS_BRIGHTNESS_RANGE[1] + 1, size=n
    ).astype(np.float32)

    for r in range(1, WS_SIZE_MAX + 1):
        mask_r = radius == r
        n_r = int(mask_r.sum())
        if n_r == 0:
            continue
        ys = y[mask_r]
        xs = x[mask_r]
        bs = brightness[mask_r]
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        dist_sq = xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2
        circle = dist_sq <= r * r
        dys = yy[circle]
        dxs = xx[circle]
        zz = np.sqrt(np.maximum(0.0, r * r - dist_sq[circle]))
        dot = (dxs * lv[0] + dys * lv[1] + zz * lv[2]) / r
        shade = np.clip(dot, 0.0, 1.0)
        falloff = 1.0 - np.sqrt(dist_sq[circle]) / r
        weights = (shade * falloff).astype(np.float32)
        rows = (ys[:, None] + dys).ravel()
        cols = (xs[:, None] + dxs).ravel()
        vals = (bs[:, None] * weights).ravel()
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        np.add.at(out, (rows[valid], cols[valid]), vals[valid])

    return out


def apply_black_spots(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    mask = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(BS_NUM_SPOTS):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        
        area = np.random.uniform(BS_AREA_RANGE[0], BS_AREA_RANGE[1])
        radius = int(np.sqrt(area / np.pi))
        
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
        
        Y, X = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (X - x)**2 + (Y - y)**2
        
        local_mask = dist_sq <= radius**2
        mask[y_min:y_max, x_min:x_max] = np.maximum(mask[y_min:y_max, x_min:x_max], local_mask)

    smoothed_mask = ndimage.gaussian_filter(mask, sigma=BS_EDGE_SMOOTHNESS)
    darken_map = 1.0 - (smoothed_mask * BS_DARKEN_FACTOR)
    return arr * darken_map


def apply_vertical_lines(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    pattern = ((np.arange(w, dtype=np.int32) % VL_PERIOD) < VL_THICKNESS).astype(
        np.float32
    )
    smoothed_pattern = ndimage.gaussian_filter1d(pattern, sigma=VL_SMOOTHNESS)
    
    if smoothed_pattern.max() > 0:
        smoothed_pattern = (smoothed_pattern / smoothed_pattern.max()) * VL_INTENSITY
        
    return arr + smoothed_pattern


def apply_perlin_darkening(arr: np.ndarray) -> np.ndarray:
    noise = np.random.rand(*arr.shape).astype(np.float32)
    smoothed_noise = _gaussian_filter_fast(noise, PL_BLUR_SIGMA)
    
    noise_min, noise_max = smoothed_noise.min(), smoothed_noise.max()
    normalized_noise = (smoothed_noise - noise_min) / (noise_max - noise_min)
    
    darken_map = 1.0 - (normalized_noise * PL_DARKEN_STRENGTH)
    return arr * darken_map


def apply_perlin_whitening(arr: np.ndarray) -> np.ndarray:
    noise = np.random.rand(*arr.shape).astype(np.float32)
    smoothed_noise = _gaussian_filter_fast(noise, PL_WHITE_BLUR_SIGMA)
    
    noise_min, noise_max = smoothed_noise.min(), smoothed_noise.max()
    normalized_noise = (smoothed_noise - noise_min) / (noise_max - noise_min)
    
    whiten_map = normalized_noise * PL_WHITEN_STRENGTH
    return arr + whiten_map


def apply_general_noise(arr: np.ndarray) -> np.ndarray:
    # Generates standard Gaussian noise matching the image dimensions
    noise = np.random.normal(GN_MEAN, GN_STD, arr.shape)
    return arr + noise


def apply_baseline_pedestal(arr: np.ndarray) -> np.ndarray:
    """Add a constant + slowly-varying offset to simulate detector read-noise floor.

    Real experimental images never have pure-black pixels; there is always a
    baseline brightness from sensor read noise. This adds a constant pedestal
    plus a smooth spatial variation so the floor isn't perfectly uniform.
    """
    h, w = arr.shape
    # Constant offset
    pedestal = np.full_like(arr, BP_LEVEL, dtype=np.float32)
    # Slow spatial variation
    if BP_VARIATION_STRENGTH > 0:
        variation = np.random.randn(h, w).astype(np.float32)
        variation = _gaussian_filter_fast(variation, BP_VARIATION_SIGMA)
        # Normalise to [-1, 1] then scale
        v_min, v_max = variation.min(), variation.max()
        if v_max - v_min > 0:
            variation = (variation - v_min) / (v_max - v_min) * 2.0 - 1.0
        variation *= BP_VARIATION_STRENGTH
        pedestal = pedestal + variation
    return arr + pedestal


def apply_gradient_blobs(arr: np.ndarray) -> np.ndarray:
    """Add large-scale smooth Gaussian brightness blobs.

    Simulates non-uniform detector sensitivity, scattered X-ray fogging,
    or uneven illumination during readout.  Generates 1-N blobs with random
    positions, sizes, and intensities and additively blends them.
    """
    h, w = arr.shape
    blob_map = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    for _ in range(GB_NUM_BLOBS):
        cx = np.random.uniform(0, w)
        cy = np.random.uniform(0, h)
        sigma = np.random.uniform(GB_SIGMA_RANGE[0], GB_SIGMA_RANGE[1])
        intensity = np.random.uniform(GB_INTENSITY_RANGE[0], GB_INTENSITY_RANGE[1])
        blob = intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        blob_map += blob

    return arr + blob_map
