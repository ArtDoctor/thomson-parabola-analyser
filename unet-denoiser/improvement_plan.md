# UNet Denoiser Improvement Plan

## Current Pipeline Summary

| Component | Details |
|---|---|
| **Architecture** | 4-level UNet (`DeeperUNet`), encoder 64→128→256→512, `LeakyReLU(0.2)`, `BatchNorm` on inner blocks, output `ReLU()` (positive-only), single-channel grayscale in/out |
| **Loss** | Composite: weighted L1 (15× weight where target > 0.05) + 2× MSE + (1 − SSIM) |
| **Training data** | ~89 base synthetic images × 3 clean transforms × 2 noise variants = **~534 noisy + ~267 clean** pairs, 1200×1200 px |
| **Noise model** | White spots, black spots, vertical lines, Perlin darkening, Perlin whitening, Gaussian noise, baseline pedestal, gradient blobs — applied in random order |
| **Clean images** | C++ Thomson simulator → `hits_to_img` histogram (1200 bins, linear normalisation to 0-255) |
| **Inference** | Patch-based sliding window (512×512, stride 256), Hann windowed blending |
| **Training** | AdamW 1e-4, ReduceLROnPlateau (factor 0.5, patience 10), early stopping patience 10, batch size 8, crop 512, max 200 epochs |

### Key Domain Gap Observations

| Property | Synthetic | Experimental |
|---|---|---|
| Background mean | ≈ 12–18 (noisy), ≈ 0.9 (clean) | ≈ 27–47 |
| Background std | ≈ 9 | ≈ 15–17 |
| Min pixel value | 0 | 7–10 (never zero) |
| % pixels > 0 | ~85% (noisy) / ~7% (clean) | ~100% |
| Image size | 1200×1200 (fixed) | ~1193×1242 (variable) |
| Dynamic range | max ≈ 164 (noisy) | max ≈ 255 |
| Noise character | Structured (spots, lines, Perlin) but "clean black" background | Continuous sensor noise floor, never truly dark |


## Improvement Recommendations

### 🟢 TIER 1 — LOW-HANGING FRUIT (High impact, low effort)

#### 1. Use a log or sqrt intensity mapping in `hits_to_img`
**What:** Currently `hits_to_img` uses linear normalisation: `H / H.max() * 255`. This compresses faint parabolas into very few gray levels and wastes dynamic range on the brightest bin. Use `np.log1p(H)` or `np.sqrt(H)` before normalising.
**Why it helps:** Real detector images (imaging plates, phosphor screens) have a sub-linear response. The experimental images clearly show faint tails that are visible — something the current linear mapping does not reproduce well. This gives the clean targets a more realistic intensity profile and lets the model learn to preserve faint structures.
**Expected improvement: ★★★★☆**
**Effort: ★ (3 lines in `hits_to_img.py` — the comment in the code even mentions this!)**

---

### 🟡 TIER 2 — MODERATE EFFORT, HIGH REWARD

#### 2. Match the intensity histogram between synthetic and real images
**What:** Compute the histogram of a set of real experimental images (your own TIFs or PNGs from the instrument). During training, randomly apply histogram matching to warp each synthetic noisy image's intensity distribution to look like a randomly selected real image.
**Why it helps:** Directly closes the intensity distribution gap. Even an approximate match greatly improves generalisation.
**Expected improvement: ★★★★☆**
**Effort: ★★ (scikit-image `match_histograms` is one function call; integrate into dataset)**

#### 3. Replace BatchNorm with GroupNorm or InstanceNorm
**What:** BatchNorm statistics are computed per-batch. With small batches (8) and very sparse images (>90% black), the batch statistics are noisy and shift at inference. Use `GroupNorm(num_groups=8, num_channels=C)` or `InstanceNorm2d` instead.
**Why it helps:** GroupNorm is independent of batch size and works much better for image restoration tasks. This is a well-known improvement for dense-prediction networks.
**Expected improvement: ★★★☆☆**
**Effort: ★★ (swap `BatchNorm2d` → `GroupNorm` in model.py)**

#### 4. Use Charbonnier loss instead of L1+MSE
**What:** Replace the weighted L1 + MSE combo with a Charbonnier (smooth L1) loss: `loss = sqrt((pred - target)^2 + eps^2)`. This is differentiable everywhere and less sensitive to outliers than MSE while still penalising large errors more than L1.
**Why it helps:** Avoids the MSE's tendency to over-smooth and L1's sharp corners. Very common in modern image restoration (used in SwinIR, EDSR, etc.)
**Expected improvement: ★★☆☆☆**
**Effort: ★★ (implement a 5-line loss function)**

---

### 🔴 TIER 3 — HIGHER EFFORT, POTENTIALLY TRANSFORMATIVE

#### 5. Fine-tune on real experimental images (self-supervised / Noise2Noise)
**What:** Use **Noise2Noise** or **Noise2Self** training on real experimental images. Since you likely don't have paired clean/noisy real images, split a real noisy image into two "pseudo-pairs" (e.g. using random pixel subsets or consecutive shots of the same setup) and fine-tune the pre-trained model.
**Why it helps:** Directly teaches the model what real noise looks like. Even 10–20 real images used for fine-tuning can dramatically close the domain gap. This is the gold standard for domain adaptation in image denoising.
**Expected improvement: ★★★★★**
**Effort: ★★★ (needs some real data curation and a Noise2Noise training loop)**

#### 6. Add a deeper bottleneck or use attention modules
**What:** Add one more encoder/decoder level (e.g. 512→1024→512) or integrate channel-attention (SE blocks) or spatial-attention (CBAM) into the bottleneck.
**Why it helps:** The current 4-level UNet has a receptive field of ~64 pixels at the deepest level (with 3×3 convolutions). Thomson parabola features can span hundreds of pixels. More depth or attention helps the model capture long-range structure.
**Expected improvement: ★★★☆☆**
**Effort: ★★★ (architecture change + retraining)**

#### 7. Use a perceptual / feature-matching loss
**What:** Add a VGG-based perceptual loss term. Extract features from a pretrained VGG-16 at layers `relu1_2`, `relu2_2`, `relu3_3` and compare pred vs target in feature space.
**Why it helps:** Encourages the model to preserve structural features (like parabola curves) even when pixel-level similarity is ambiguous. Reduces blurriness.
**Expected improvement: ★★☆☆☆**
**Effort: ★★★ (load VGG, implement feature extraction, add loss term)**

#### 8. Style-transfer / CycleGAN for domain adaptation
**What:** Train a CycleGAN to translate synthetic noisy images → realistic-looking experimental images without paired supervision. Then retrain the denoiser on this "more realistic" synthetic data.
**Why it helps:** Learns the precise noise characteristics of the real detector without needing paired data.
**Expected improvement: ★★★★☆**
**Effort: ★★★★ (full CycleGAN training pipeline)**

#### 9. Inference-time test augmentation (TTA)
**What:** At inference, run the model on the original image AND 3 rotated/flipped versions. Average the (inverse-transformed) outputs.
**Why it helps:** Reduces prediction variance and artifact patterns with zero retraining.
**Expected improvement: ★★☆☆☆**
**Effort: ★ (modify `denoise_large_image` to loop over transforms)**

---

### 🔵 TIER 4 — SYNTHETIC DATA REALISM (Closing the visual domain gap)

#### 10. Realistic bright spot (H0) modelling
**What:** The current H0 "bright spot" is a perfect symmetric circle placed at the origin. Real experimental images show bright spots that are:
- **Larger and more diffuse** with a sharp bright core fading into a wider halo
- **Asymmetric** — elongated or irregularly shaped due to beam profile and detector response
- **Accompanied by ghost spots** — secondary dimmer reflections offset from the main spot
- **Radial streaks/rays** — faint lines extending outward from the spot center (blooming, diffraction, or saturation artifacts)

Replace the simple circle with a composite model: a 2D asymmetric Gaussian core (random ellipticity and orientation) + a wider low-intensity halo (second Gaussian or exponential decay) + optional ghost spots (1–3 dimmer copies with random offsets) + optional radial streaks (thin lines at random angles with exponential intensity falloff).
**Why it helps:** The bright spot is one of the most prominent features in every experimental image and the current synthetic version looks nothing like the real thing. Any model seeing a perfect circle in training will struggle with the messy real spot, potentially misclassifying the halo/ghosts/streaks as noise.
**Expected improvement: ★★★☆☆**
**Effort: ★★ (modify `hits_to_img.py` or add a post-processing step to composite the spot)**

#### 11. Asymmetric angular dispersion (directional velocity bias)
**What:** Currently, the simulator draws particle velocities from a symmetric distribution (`angle_sigma`, `beam_sigma`), producing parabolas that spread evenly in all directions. In real experiments, parabolas often show asymmetric spread — one side of a parabola is wider/brighter than the other, especially for low-q/m species. Implement this by:
- Adding a random directional bias to the velocity sampling (e.g., shift the angular mean slightly off-center per species)
- Using different sigma values for the left vs right half of the angular distribution
- Optionally making the energy distribution slightly asymmetric (more particles at one end of the energy spectrum)

**Why it helps:** Symmetric parabolas look "too perfect" compared to real data. The asymmetry comes from real physics: non-uniform target conditions, off-axis laser incidence, field inhomogeneities, and detector placement. Training on only symmetric parabolas may cause the model to treat asymmetric features as noise rather than signal.
**Expected improvement: ★★★☆☆**
**Effort: ★★ (modify velocity sampling in the C++ simulator — add a per-species angular offset and split-sigma)**

#### 12. Additional noise types (hot pixels, readout streaks, vignetting, dust/scratches)
**What:** Add several minor but realistic noise types that appear in real detector images:
- **Hot/dead pixel clusters:** Small clusters (1–5 px) of stuck-bright or stuck-dark pixels, randomly placed. ~20–100 per image.
- **Readout streaks:** Faint horizontal or vertical streaks spanning the full image width/height, caused by CCD readout artifacts. 1–5 per image, intensity ~3–10 gray levels.
- **Edge vignetting:** Gradual darkening toward the image edges (radial gradient, ~5–15% intensity drop at corners).
- **Dust/scratch marks:** Random thin curved lines or small irregular dark patches simulating physical detector surface defects.

**Why it helps:** Real detector images accumulate many small imperfections that collectively define their visual character. The current noise model covers the dominant effects well (Gaussian noise, Perlin variation, spots, lines, pedestal, gradient blobs) but misses these secondary artifacts. Adding them increases the diversity of the training noise, making the model more robust to the varied and messy reality of experimental data.
**Expected improvement: ★★☆☆☆**
**Effort: ★★ (implement 4 small noise functions in `noise.py`)**

---

## Prioritised Action Plan

| Priority | Action | Impact | Effort | Notes |
|---|---|---|---|---|
| **1** | Use log/sqrt intensity mapping (#1) | ★★★★☆ | ★ | Already suggested in code comments! |
| **2** | Histogram matching to real images (#2) | ★★★★☆ | ★★ | Direct domain gap reduction |
| **3** | Realistic bright spot modelling (#10) | ★★★☆☆ | ★★ | H0 spot is visually very wrong currently |
| **4** | Asymmetric angular dispersion (#11) | ★★★☆☆ | ★★ | Makes parabolas look realistic, prevents model treating asymmetry as noise |
| **5** | Replace BatchNorm with GroupNorm (#3) | ★★★☆☆ | ★★ | Better for small-batch sparse images |
| **6** | Additional noise types (#12) | ★★☆☆☆ | ★★ | Hot pixels, streaks, vignetting, dust — secondary but adds diversity |
| **7** | Fine-tune on real data with Noise2Noise (#5) | ★★★★★ | ★★★ | Ultimate domain adaptation |
| **8** | Inference TTA (#9) | ★★☆☆☆ | ★ | Free improvement, no retraining |
| **9** | Charbonnier loss (#4) | ★★☆☆☆ | ★★ | Minor but principled improvement |
| **10** | Deeper bottleneck / attention (#6) | ★★★☆☆ | ★★★ | Larger receptive field |
| **11** | Perceptual loss (#7) | ★★☆☆☆ | ★★★ | Helps preserve structure |
| **12** | CycleGAN domain adaptation (#8) | ★★★★☆ | ★★★★ | Heavy but powerful |

---

## Summary

The core issue is **domain gap**, not model capacity. The clean targets are correct (signal-only, mostly black). But the synthetic *noisy inputs* were fundamentally different from real experimental noisy images. Several fixes are now in place: baseline pedestal, widened Gaussian noise range, gradient blobs, data augmentation (flips/rotations), and longer training (200 epochs, patience 10).

Remaining gaps:

1. **Dynamic range / tone mapping** — linear histogram normalisation in `hits_to_img` doesn't match real detector response curves (fix #1).
2. **Intensity histogram matching** — warp synthetic noisy images to match real image intensity distributions (#2).
3. **Visual realism of synthetic features** — the bright spot (H0) is a perfect circle instead of the messy asymmetric blob with halos/ghosts seen in real data (#10); parabolas are perfectly symmetric when real ones show directional bias (#11); and the noise model could add hot pixels, readout streaks, vignetting, and dust marks (#12).

Implementing #1 (log/sqrt mapping) and #2 (histogram matching) should produce a **dramatic** improvement. The Tier 4 items (#10–#12) further close the visual domain gap by making synthetic images *look* more like real experimental data.
