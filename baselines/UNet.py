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
            in_channels (int): nombre de canaux en entrée (par exemple, 3 pour RGB).
            num_classes (int): nombre de classes à segmenter.
            dropout (float): taux de dropout utilisé dans les blocs.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Voie descendante
        self.conv1 = DoubleConv(in_channels, 64)
        self.drop1 = nn.Dropout2d(dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = DoubleConv(64, 128)
        self.drop2 = nn.Dropout2d(dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = DoubleConv(128, 256)
        self.drop3 = nn.Dropout2d(dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = DoubleConv(256, 512)
        self.drop4 = nn.Dropout2d(dropout)  # Ici, la sortie du dropout sera utilisée comme skip connection (cf. version Keras)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(dropout)

        # Voie montante avec upsampling puis concaténation des skip connections
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512 + 512, 512)  # concatène la sortie de up6 et le skip branch issue de drop4

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256 + 256, 256)  # concatène up7 et la sortie de conv3 (avant dropout)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128 + 128, 128)  # concatène up8 et la sortie de conv2

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(64 + 64, 64)     # concatène up9 et la sortie de conv1

        # Couche finale pour ramener à num_classes canaux (pas d'activation ici car CrossEntropyLoss attend des logits)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, training=False, memory_frames=None, memory_masks=None):
        # Voie descendante
        c1 = self.conv1(x)      # c1: B x 64 x H x W
        d1 = self.drop1(c1)
        p1 = self.pool1(d1)

        c2 = self.conv2(p1)     # c2: B x 128 x H/2 x W/2
        d2 = self.drop2(c2)
        p2 = self.pool2(d2)

        c3 = self.conv3(p2)     # c3: B x 256 x H/4 x W/4
        d3 = self.drop3(c3)
        p3 = self.pool3(d3)

        c4 = self.conv4(p3)     # c4: B x 512 x H/8 x W/8
        d4 = self.drop4(c4)     # Utilisation de la version dropout pour la skip connection
        p4 = self.pool4(d4)

        c5 = self.conv5(p4)     # c5: B x 1024 x H/16 x W/16
        d5 = self.drop5(c5)

        # Voie montante
        up6 = self.up6(d5)                          # up6: B x 512 x H/8 x W/8
        merge6 = torch.cat([d4, up6], dim=1)         # Concaténation le long de la dimension canaux (512+512)
        c6 = self.conv6(merge6)

        up7 = self.up7(c6)                          # up7: B x 256 x H/4 x W/4
        merge7 = torch.cat([c3, up7], dim=1)         # Utilise c3 (avant dropout) comme skip connection
        c7 = self.conv7(merge7)

        up8 = self.up8(c7)                          # up8: B x 128 x H/2 x W/2
        merge8 = torch.cat([c2, up8], dim=1)
        c8 = self.conv8(merge8)

        up9 = self.up9(c8)                          # up9: B x 64 x H x W
        merge9 = torch.cat([c1, up9], dim=1)
        c9 = self.conv9(merge9)

        out = self.final_conv(c9)  # Output shape: B x num_classes x H x W
        return out

    def _initialize_weights(self):
        # Initialisation "he_normal" pour les couches convolutionnelles
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)