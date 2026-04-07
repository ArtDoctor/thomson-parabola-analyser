import time
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from utils.inference import denoise_large_image
from utils.model import DeeperUNet


def main() -> None:
    device = torch.device("cpu")
    model = DeeperUNet().to(device)
    model.eval()

    image = Image.fromarray(
        np.random.randint(0, 255, (1200, 1200), dtype=np.uint8),
        mode="L",
    )
    dummy_path = Path(__file__).resolve().with_name("dummy_benchmark.png")
    image.save(dummy_path)

    try:
        start = time.time()
        denoise_large_image(str(dummy_path), model, device)
        print(f"Patch-based inference took {time.time() - start:.2f} seconds")

        start = time.time()
        with torch.no_grad():
            img_tensor = T.functional.to_tensor(image).unsqueeze(0).to(device)
            h, w = img_tensor.shape[2:]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            out = model(padded)
            out = out[:, :, :h, :w]
        print(f"Single pass inference took {time.time() - start:.2f} seconds")
    finally:
        dummy_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
