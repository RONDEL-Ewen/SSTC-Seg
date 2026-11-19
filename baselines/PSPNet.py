import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module as in PSPNet.
    Pools the input feature map at several grid scales, applies a 1x1 convolution to reduce channels,
    then upsamples to original feature map size.
    Uses GroupNorm instead of BatchNorm so it works with batch size = 1.
    """
    def __init__(self, in_channels, pool_sizes, reduction_dim, gn_groups=16):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(size, size)),
                    nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(gn_groups, reduction_dim),
                    nn.ReLU(inplace=True)
                )
            )
        # remove batch-dependence in final fusion as well
        self.fusion_norm = nn.GroupNorm(gn_groups, in_channels + len(pool_sizes) * reduction_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h, w = x.size(2), x.size(3)
        outputs = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
            outputs.append(y)
        out = torch.cat(outputs, dim=1)
        out = self.fusion_norm(out)
        return out


class PSPNet(nn.Module):
    """
    PSPNet for semantic segmentation with a ResNet-50 backbone and Pyramid Pooling Module.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_sizes=(1, 2, 3, 6),
        ppm_reduction_dim: int = 512,
        backbone_pretrained: bool = True,
        dropout: float = 0.1,
        gn_groups: int = 16
    ):
        super(PSPNet, self).__init__()
        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=backbone_pretrained)
        # If input channels differ, adjust the first conv
        if in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        else:
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        # ResNet layers
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Pyramid pooling module
        self.ppm = PyramidPoolingModule(
            in_channels=2048,
            pool_sizes=pool_sizes,
            reduction_dim=ppm_reduction_dim,
            gn_groups=gn_groups
        )

        # Final fusion conv and classifier
        fusion_channels = 2048 + len(pool_sizes) * ppm_reduction_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(
        self,
        x:        torch.Tensor,
        training: bool = False,
        memory_frames  = None,
        memory_masks   = None
    ):
        # Input spatial size
        h0, w0 = x.size(2), x.size(3)
        # Backbone feature extraction
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Pyramid pooling
        x = self.ppm(x)
        # Fusion and classification
        x = self.fusion(x)
        x = self.classifier(x)
        # Upsample to input resolution
        x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=True)
        return x