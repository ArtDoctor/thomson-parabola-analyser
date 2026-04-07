import torch
import torch.nn as nn


class DeeperUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        def double_conv(in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.d1 = double_conv(in_channels, 64, norm=False)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)
        self.d4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u1 = double_conv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u2 = double_conv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u3 = double_conv(128, 64, norm=False)

        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.u1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.u2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.u3(x)

        return self.out(x)
