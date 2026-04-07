import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


EVAL_SYNTH_DIR = Path("eval_synth")


def _load_ground_truth(image_stem: str) -> dict[str, Any] | None:
    """Load ground truth JSON for a synthetic image, if it exists."""
    gt_path = EVAL_SYNTH_DIR / f"{image_stem}.json"
    if gt_path.exists():
        with gt_path.open() as f:
            return json.load(f)
    return None


def _compare_species(
    res: dict[str, Any],
    gt: dict[str, Any],
) -> dict[str, Any]:
    """Compare detected species in res.json against ground truth.

    Returns dict with precision, recall, true/false positives, etc.
    """
    gt_labels = {sp["label"] for sp in gt["species"]}
    gt_mq = {sp["label"]: sp["m_over_q"] for sp in gt["species"]}

    detected_labels: set[str] = set()
    mq_errors: list[float] = []

    for entry in res.get("classified", []):
        label = entry.get("label", "")
        detected_labels.add(label)
        if label in gt_mq:
            mq_meas = entry.get("mq_meas", 0.0)
            mq_true = gt_mq[label]
            if mq_true > 0:
                mq_errors.append(abs(mq_meas - mq_true) / mq_true)

    true_pos = detected_labels & gt_labels
    false_pos = detected_labels - gt_labels
    false_neg = gt_labels - detected_labels

    n_tp = len(true_pos)
    n_fp = len(false_pos)
    n_fn = len(false_neg)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0

    return {
        "true_positives": sorted(true_pos),
        "false_positives": sorted(false_pos),
        "false_negatives": sorted(false_neg),
        "precision": precision,
        "recall": recall,
        "mean_mq_rel_error": float(np.mean(mq_errors)) if mq_errors else None,
        "n_gt_species": len(gt_labels),
        "n_detected": len(detected_labels),
    }


def run(output_dir: str = "outputs") -> None:
    files = glob.glob(f"{output_dir}/*/res.json")
    total_parabolas = 0
    unidentified = 0
    rmses: list[float] = []

    synth_results: list[dict[str, Any]] = []
    clean_results: list[dict[str, Any]] = []
    noisy_results: list[dict[str, Any]] = []

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        ev = data.get("eval", {})
        if not ev:
            continue

        summary = ev.get("species_summary", {})
        total_parabolas += summary.get("n_classified_parabolae", 0)
        unidentified += summary.get("n_unidentified", 0)

        rmse = ev.get("global_rmse_y_eff_px")
        if rmse is not None:
            rmses.append(rmse)

        image_stem = Path(f).parent.name
        gt = _load_ground_truth(image_stem)
        if gt is not None:
            comparison = _compare_species(data, gt)
            comparison["image"] = image_stem
            comparison["noisy"] = gt.get("noisy", False)
            comparison["rmse"] = rmse
            synth_results.append(comparison)
            if gt.get("noisy", False):
                noisy_results.append(comparison)
            else:
                clean_results.append(comparison)

    identified = total_parabolas - unidentified
    avg_rmse = np.mean(rmses) if rmses else 0.0

    print(f"Total Parabolas Detected: {total_parabolas}")
    print(f"Unidentified Species: {unidentified}")
    print(f"Identified Species: {identified}")
    print(f"Average RMSE: {avg_rmse:.4f} px")

    if synth_results:
        print(f"\n{'=' * 60}")
        print("SYNTHETIC GROUND TRUTH EVALUATION")
        print(f"{'=' * 60}")
        _print_synth_summary("All synthetic", synth_results)
        if clean_results:
            _print_synth_summary("Clean images", clean_results)
        if noisy_results:
            _print_synth_summary("Noisy images", noisy_results)

        print("\n--- Per-image breakdown ---")
        print(f"{'Image':<30} {'Prec':>6} {'Rec':>6} {'mq_err':>8} "
              f"{'TP':>3} {'FP':>3} {'FN':>3} {'RMSE':>8} {'Tag':>5}")
        for r in sorted(synth_results, key=lambda x: x["image"]):
            mq_str = f"{r['mean_mq_rel_error']:.4f}" if r["mean_mq_rel_error"] is not None else "   N/A"
            rmse_str = f"{r['rmse']:.2f}" if r["rmse"] is not None else "  N/A"
            tag = "noisy" if r["noisy"] else "clean"
            print(
                f"{r['image']:<30} {r['precision']:>6.2f} {r['recall']:>6.2f} "
                f"{mq_str:>8} {len(r['true_positives']):>3} "
                f"{len(r['false_positives']):>3} {len(r['false_negatives']):>3} "
                f"{rmse_str:>8} {tag:>5}"
            )


def _print_synth_summary(label: str, results: list[dict[str, Any]]) -> None:
    """Print aggregate metrics for a group of synthetic results."""
    n = len(results)
    avg_prec = np.mean([r["precision"] for r in results])
    avg_rec = np.mean([r["recall"] for r in results])
    mq_errs = [r["mean_mq_rel_error"] for r in results if r["mean_mq_rel_error"] is not None]
    avg_mq = np.mean(mq_errs) if mq_errs else float("nan")
    total_tp = sum(len(r["true_positives"]) for r in results)
    total_fp = sum(len(r["false_positives"]) for r in results)
    total_fn = sum(len(r["false_negatives"]) for r in results)

    print(f"\n  {label} ({n} images):")
    print(f"    Avg precision:       {avg_prec:.3f}")
    print(f"    Avg recall:          {avg_rec:.3f}")
    print(f"    Avg m/q rel error:   {avg_mq:.4f}")
    print(f"    Total TP / FP / FN:  {total_tp} / {total_fp} / {total_fn}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    run(out_dir)
