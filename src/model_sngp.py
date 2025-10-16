import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from SNGP.gaussian_process import RandomFeatureGaussianProcess
from SNGP.sngp_head import SNGP  # assuming you save your class here

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET_SNGP(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        features=[64, 128, 256, 512],
        reduction_dim=128,
        random_features=1024,
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final feature projection before GP
        self.feature_projection = nn.utils.spectral_norm(
            nn.Conv2d(features[0], reduction_dim, kernel_size=1)
        )

        # SNGP head (takes flattened spatial features)
        self.sngp_head = SNGP(
            in_features=reduction_dim,
            num_classes=num_classes,
            reduction_dim=reduction_dim,
            classif_dropout=0.2,
            random_features=random_features,
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # Project feature maps down to reduction_dim
        x = self.feature_projection(x)  # [B, R, H, W]

        # Flatten spatial dimensions for GP
        B, R, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, R)  # [B*H*W, R]

        # Pass through SNGP head
        gp_out = self.sngp_head(x_flat)  # returns (logits, cov) tuple
        logits, cov = gp_out[0], gp_out[1]

        # Reshape back to image shape
        logits = logits.view(B, 1, H, W)

        return logits, cov
