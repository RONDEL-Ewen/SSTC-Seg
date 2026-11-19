import torch
import torch.nn as nn
import torch.nn.functional as F

from .frame_encoder import MSMCConv, DeformMSMCConv



# ===============================[ DecoderBlock ]===============================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        deformable: bool = True
    ):
        super(DecoderBlock, self).__init__()

        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size = 2, stride = 2)

        # MSMCConv
        if deformable:
            self.msmc = DeformMSMCConv(output_dim, output_dim)
        else:
            self.msmc = MSMCConv(output_dim, output_dim)

        # Final Conv3x3 layer
        self.conv3x3 = nn.Conv2d(output_dim, output_dim, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace = True)

    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor
    ):

        # Upsample the input
        x = self.upsample(x)


        # TEST
        # Align spatial dimensions if needed
        if x.shape[-2:] != skip_connection.shape[-2:]:
            x = F.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=True)

        # Add skip connection
        x = x + skip_connection

        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)

        out = self.msmc(x)

        return out



# ===============================[ MaskDecoder ]===============================

class SharedHeavyMaskDecoder(nn.Module):
    """
    Mask decoder with heavy (upsampling + MSMCConv) layers shared,
    and separate 1x1 heads per class.
    Compatible with the original MaskDecoder interface.

    Args:
        final_dim (int): Depth of the incoming feature map from encoder.
        num_classes (int): Number of output classes.
        num_blocks (int): Number of DecoderBlock layers.
        base_dim (int): Base channel dimension for decoding blocks.
        deformable (bool): Flag for using deformable convolutions.
    """
    def __init__(
        self,
        final_dim:   int,
        num_classes: int = 2,
        num_blocks:  int = 4,
        base_dim:    int = 64,
        deformable:  bool = True
    ):
        super().__init__()
        # Build shared decoder blocks (same as original MaskDecoder)
        self.blocks = nn.ModuleList()
        current_dim = final_dim
        for i in range(num_blocks - 1, -1, -1):
            next_dim = base_dim * (2 ** i)
            self.blocks.append(
                DecoderBlock(current_dim, next_dim, deformable=deformable)
            )
            current_dim = next_dim

        # Separate 1x1 conv head per class
        self.heads = nn.ModuleList([
            nn.Conv2d(base_dim, 1, kernel_size=1)
            for _ in range(num_classes)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map from the encoder (shape [B, final_dim, H', W']).
            encoder_outputs (list[Tensor]): Skip connections from each encoder stage.

        Returns:
            Tensor: Stack of per-class masks of shape [B, num_classes, H, W].
        """
        # Shared decoding
        for i, block in enumerate(self.blocks):
            x = block(x, encoder_outputs[-(i + 1)])

        # Per-class head
        masks = [head(x) for head in self.heads]  # each [B,1,H,W]
        # Stack into [B, num_classes, H, W]
        return torch.cat(masks, dim=1)