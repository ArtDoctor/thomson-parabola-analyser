import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from utils.config import TrainingConfig
from utils.dataset import DenoisingDataset
from utils.inference import denoise_large_image
from utils.loss import composite_loss
from utils.model import DeeperUNet


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _save_train_generator_preview(
    train_ds: torch.utils.data.Dataset,
    out_dir: Path,
    num_images: int,
) -> None:
    if len(train_ds) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        idx = i % len(train_ds)
        noisy, clean = train_ds[idx]
        n_hw = noisy.squeeze(0).detach().cpu().numpy()
        c_hw = clean.squeeze(0).detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(n_hw, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("Noisy (train aug)")
        axes[1].imshow(c_hw, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Clean (same aug)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Train sample {i + 1}/{num_images} (dataset idx {idx})")
        plt.tight_layout()
        out_path = out_dir / f"train_generator_sample_{i + 1:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _autocast(device: torch.device) -> torch.amp.autocast_mode.autocast:
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def _maybe_init_wandb(config: TrainingConfig) -> object | None:
    if wandb is None:
        print("wandb is not installed; continuing without experiment logging.")
        return None

    init_kwargs: dict[str, object] = {}
    mode = os.environ.get("WANDB_MODE")
    if mode:
        init_kwargs["mode"] = mode

    try:
        return wandb.init(
            project="thomson-denoising",
            config=config.model_dump(),
            **init_kwargs,
        )
    except Exception as exc:  # pragma: no cover - depends on local wandb setup
        print(f"wandb init failed ({exc}); continuing without experiment logging.")
        return None


def _maybe_log(run: object | None, payload: dict[str, object]) -> None:
    if run is not None:
        run.log(payload)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TrainingConfig()
    run = _maybe_init_wandb(config)

    clean_dir = BASE_DIR / config.clean_dir
    noisy_dir = BASE_DIR / config.noisy_dir
    checkpoints_dir = BASE_DIR / "checkpoints"
    output_dir = BASE_DIR / "output"
    test_dir = BASE_DIR / "test"

    dataset = DenoisingDataset(
        str(clean_dir),
        str(noisy_dir),
        crop_size=config.crop_size,
    )
    if len(dataset) < 2:
        raise ValueError(
            "The UNet training dataset needs at least two noisy/clean pairs. "
            f"Found {len(dataset)} items under {noisy_dir}."
        )

    train_size = max(1, int(0.9 * len(dataset)))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    checkpoints_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    _save_train_generator_preview(train_dataset, output_dir, num_images=6)

    fixed_val_noisy, fixed_val_clean = next(iter(val_loader))
    fixed_val_noisy = fixed_val_noisy.to(device)
    fixed_val_clean = fixed_val_clean.to(device)

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    model = DeeperUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    epochs_no_improve = 0
    best_val_loss = float("inf")
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    checkpoint_path = checkpoints_dir / "unet_denoise_best.pth"

    for epoch in range(config.epochs):
        model.train()
        epoch_train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()

            with _autocast(device):
                preds = model(noisy)
                loss = composite_loss(preds, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy_val, clean_val in val_loader:
                noisy_val = noisy_val.to(device)
                clean_val = clean_val.to(device)
                with _autocast(device):
                    preds_val = model(noisy_val)
                    val_loss = composite_loss(preds_val, clean_val)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        _maybe_log(
            run,
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch + 1,
            },
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved new best model (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(
                "    No improvement. Early stopping counter: "
                f"{epochs_no_improve}/{config.patience}"
            )
            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_history, label="Training Loss (L1)", color="blue")
    plt.plot(val_loss_history, label="Validation Loss (L1)", color="orange", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Denoising Model Training & Validation Progression")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(output_dir / "loss_history.png", dpi=150, bbox_inches="tight")
    plt.close()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded best model from {checkpoint_path} for validation and test inference")
    model.eval()
    with torch.no_grad():
        with _autocast(device):
            fixed_preds = model(fixed_val_noisy)

        fixed_preds = torch.clamp(fixed_preds, 0.0, 1.0)
        fixed_gt = fixed_val_clean

        num_images_to_log = min(3, fixed_val_noisy.size(0))
        logged_visuals = []

        if run is not None and wandb is not None:
            for i in range(num_images_to_log):
                combined_image = torch.cat(
                    [fixed_val_noisy[i], fixed_gt[i], fixed_preds[i]], dim=2
                )
                logged_visuals.append(
                    wandb.Image(
                        combined_image,
                        caption=f"Epoch {epoch + 1}: Noisy | GT | Pred (Sample {i})",
                    )
                )

        if logged_visuals:
            _maybe_log(
                run,
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "epoch": epoch + 1,
                    "validation_samples": logged_visuals,
                },
            )

    with torch.no_grad():
        noisy_viz, clean_viz = next(iter(train_loader))
        noisy_viz = noisy_viz.to(device)
        clean_viz = clean_viz.to(device)
        preds_viz = torch.clamp(model(noisy_viz), 0.0, 1.0)

        num_preview = min(3, noisy_viz.size(0))
        fig, axes = plt.subplots(num_preview, 3, figsize=(12, 4 * num_preview))
        axes_grid = np.atleast_2d(axes)
        for i in range(num_preview):
            n_img = noisy_viz[i].cpu().squeeze().numpy()
            c_img = clean_viz[i].cpu().squeeze().numpy()
            p_img = preds_viz[i].cpu().squeeze().numpy()

            axes_grid[i, 0].imshow(n_img, cmap="gray")
            axes_grid[i, 0].set_title("Noisy Input")
            axes_grid[i, 1].imshow(c_img, cmap="gray")
            axes_grid[i, 1].set_title("Clean Ground Truth")
            axes_grid[i, 2].imshow(p_img, cmap="gray")
            axes_grid[i, 2].set_title("Model Prediction")

            for ax in axes_grid[i]:
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    test_images = glob.glob(str(test_dir / "*.*"))
    test_log_images = []

    for img_path_str in test_images:
        img_path = Path(img_path_str)
        if not img_path.is_file():
            continue

        print(f"Running inference on {img_path.name}...")
        try:
            noisy_png, denoised_png = denoise_large_image(
                str(img_path), model, device, patch_size=config.patch_size, stride=256
            )
        except Exception as exc:
            print(f"Failed to process {img_path.name}: {exc}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(noisy_png, cmap="gray")
        axes[0].set_title(f"Original Noisy Image: {img_path.name}")
        axes[0].axis("off")
        axes[1].imshow(denoised_png, cmap="gray")
        axes[1].set_title("Denoised Image (Seamless)")
        axes[1].axis("off")

        base_name = img_path.stem
        comp_path = output_dir / f"denoised_comparison_{base_name}.png"
        out_path = output_dir / f"denoised_{base_name}.png"

        Image.fromarray((denoised_png * 255).astype(np.uint8)).rotate(-90).transpose(
            Image.FLIP_LEFT_RIGHT
        ).convert("RGB").save(out_path)

        plt.tight_layout()
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close()

        if run is not None and wandb is not None:
            test_log_images.append(
                wandb.Image(
                    str(comp_path),
                    caption=f"Test Image: {img_path.name}",
                )
            )

    if test_log_images:
        _maybe_log(run, {"test_inferences": test_log_images})

    if run is not None and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
