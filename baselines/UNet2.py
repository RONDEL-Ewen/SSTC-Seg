import torch
import torch.nn as nn
import torch.nn.functional as F

# Module de double convolution identique à deux couches conv + ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.2):
        """
        Args:
            in_channels (int): number of input channels (e.g., 3 for RGB).
            num_classes (int): number of segmentation classes.
            dropout (float): dropout rate for encoder blocks.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (down path)
        self.conv1 = DoubleConv(in_channels, 64)
        self.drop1 = nn.Dropout2d(dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.drop2 = nn.Dropout2d(dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.drop3 = nn.Dropout2d(dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.drop4 = nn.Dropout2d(dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(dropout)

        # Decoder (up path)
        self.up6   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512 + 512, 512)

        self.up7   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256 + 256, 256)

        self.up8   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128 + 128, 128)

        self.up9   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(64 + 64, 64)

        # Final classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()

    @staticmethod
    def _center_crop_to(x, target_hw):
        """
        Center-crop tensor x (B, C, H, W) to spatial size target_hw=(Ht, Wt).
        If already equal, returns x unchanged.
        """
        _, _, H, W = x.size()
        Ht, Wt = target_hw
        if H == Ht and W == Wt:
            return x
        dh = max(0, H - Ht)
        dw = max(0, W - Wt)
        top = dh // 2
        left = dw // 2
        return x[:, :, top:top + Ht, left:left + Wt]

    def forward(self, x, training: bool = False, memory_frames=None, memory_masks=None):
        # ---- Encoder ----
        c1 = self.conv1(x)
        c1 = self.drop1(c1)
        d1 = self.pool1(c1)

        c2 = self.conv2(d1)
        c2 = self.drop2(c2)
        d2 = self.pool2(c2)

        c3 = self.conv3(d2)
        c3 = self.drop3(c3)
        d3 = self.pool3(c3)

        c4 = self.conv4(d3)
        c4 = self.drop4(c4)
        d4 = self.pool4(c4)

        # ---- Bottleneck ----
        c5 = self.conv5(d4)
        c5 = self.drop5(c5)

        # ---- Decoder with center-cropped skip connections ----
        up6 = self.up6(c5)                                        # B x 512 x H x W
        c4c = self._center_crop_to(c4, up6.size()[2:])            # crop encoder feat to match
        m6  = torch.cat([c4c, up6], dim=1)                        # B x 1024 x H x W
        c6  = self.conv6(m6)

        up7 = self.up7(c6)                                        # B x 256 x H x W
        c3c = self._center_crop_to(c3, up7.size()[2:])
        m7  = torch.cat([c3c, up7], dim=1)                        # B x 512 x H x W
        c7  = self.conv7(m7)

        up8 = self.up8(c7)                                        # B x 128 x H x W
        c2c = self._center_crop_to(c2, up8.size()[2:])
        m8  = torch.cat([c2c, up8], dim=1)                        # B x 256 x H x W
        c8  = self.conv8(m8)

        up9 = self.up9(c8)                                        # B x 64 x H x W
        c1c = self._center_crop_to(c1, up9.size()[2:])
        m9  = torch.cat([c1c, up9], dim=1)                        # B x 128 x H x W
        c9  = self.conv9(m9)

        out = self.final_conv(c9)                                 # B x num_classes x H x W
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

    def _initialize_weights(self):
        # He (Kaiming) normal init for conv and convtranspose layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
