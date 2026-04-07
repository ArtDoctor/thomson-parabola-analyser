import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1_loss = F.l1_loss(pred, target, reduction="none")
    weight_mask = torch.where(target > 0.05, 15.0, 1.0)
    weighted_l1 = (l1_loss * weight_mask).mean()
    mse_loss = F.mse_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True, win_size=7)
    return weighted_l1 + (2.0 * mse_loss) + (1.0 - ssim_val)
