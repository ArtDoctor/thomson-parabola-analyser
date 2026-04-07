from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-4
    patch_size: int = 512
    architecture: str = "DeeperUNet-Sparse-Optimized"
    loss_function: str = "L1 + SSIM"
    patience: int = 15
    clean_dir: str = "clean/"
    noisy_dir: str = "noisy/"
    crop_size: int = 512
    num_workers: int = 4
