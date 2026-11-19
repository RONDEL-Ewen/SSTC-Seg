import torch
import torch.nn as nn

from model.components.frame_encoder          import FrameEncoder
from model.components.mask_decoder           import MaskDecoder



# ===============================[ MFCPNet ]===============================

class MFCPNet(nn.Module):

    def __init__(
        self,
        input_dim:          int,
        num_classes:        int,
        num_encoder_blocks: int  = 5,
        base_dim:           int  = 64,
        num_heads:          int  = 8,
        k:                  int  = 5,
        deformable:         bool = False
    ):
        super(MFCPNet, self).__init__()

        self.base_dim = base_dim
        self.num_encoder_blocks = num_encoder_blocks

        # Frame Encoder
        self.image_encoder = FrameEncoder(
            input_dim  = input_dim,
            num_blocks = num_encoder_blocks,
            base_dim   = base_dim,
            num_heads  = num_heads,
            k          = k,
            deformable = False
        )

        # Mask Decoder
        self.mask_decoder = MaskDecoder(
            final_dim   = base_dim * (2 ** (num_encoder_blocks - 1)),
            num_classes = num_classes,
            num_blocks  = num_encoder_blocks - 1,
            base_dim    = base_dim,
            deformable  = False
        )

    def forward(
        self,
        x:        torch.Tensor,
        training: bool = False,
        memory_frames  = None,
        memory_masks   = None
    ):

        # STEP 1: Pass through ImageEncoder

        encoder_output, skip_connections = self.image_encoder(x)

        # STEP 2: Pass through MaskDecoder

        predicted_masks = self.mask_decoder(encoder_output, skip_connections)

        return predicted_masks