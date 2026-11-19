import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_convolution import DeformConv2D

class HierarchicalDenseCRF(nn.Module):
    def __init__(
        self,
        num_classes:        int = 8,
        num_iterations:     int = 2,
        spatial_ksize:      int = 3,
        bilateral_ksize:    int = 3,
        num_encoder_blocks: int = 5,
        base_dim:           int = 64,
        skip_channels           = None
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.num_iterations = num_iterations

        # --- deformable spatial filtering per class ---
        self.spatial_conv = DeformConv2D(
            in_channels  = num_classes,
            out_channels = num_classes,
            kernel_size  = spatial_ksize,
            padding      = spatial_ksize // 2,
            groups       = num_classes,
            bias         = False
        )

        # --- deformable bilateral filtering (uses image + skip guidance + edge) ---
        # guidance channels: class-scores + RGB (3) + fused skips (3) + edge map (1)
        self.bilateral_conv = DeformConv2D(
            in_channels  = num_classes + 3 + 3 + 1,
            out_channels = num_classes,
            kernel_size  = bilateral_ksize,
            padding      = bilateral_ksize // 2,
            groups       = 1,
            bias         = False
        )

        # --- learnable compatibility transform ---
        self.compat_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)
        self._init_compat()

        # --- skip-feature projections ---
        if skip_channels is None:
            skip_channels = [base_dim * (2 ** i) for i in range(num_encoder_blocks - 1)]
        self.skip_proj = nn.ModuleList([nn.Conv2d(ch, 3, kernel_size=1) for ch in skip_channels])

        # --- Sobel edge kernels for boundary guidance ---
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1,-2,-1],
                                  [ 0, 0, 0],
                                  [ 1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _init_compat(self):
        # Initialize compatibility matrix: diag = -1, off-diag = +0.1
        W = torch.full((self.num_classes, self.num_classes), 0.1)
        W.fill_diagonal_(-1.0)
        self.compat_conv.weight.data.copy_(W.view(self.num_classes, self.num_classes, 1, 1))
        if self.compat_conv.bias is not None:
            nn.init.zeros_(self.compat_conv.bias)

    def fuse_skip_features(self, skip_features, target_size):
        proj_feats = []
        for feat, proj in zip(skip_features, self.skip_proj):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=True)
            proj_feats.append(p)
        return torch.stack(proj_feats, dim=0).mean(dim=0)

    def compute_edge_map(self, image):
        # Convert to grayscale
        gray = image.mean(dim=1, keepdim=True)
        # Sobel gradients
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        # Edge magnitude
        edge = torch.sqrt(gx**2 + gy**2 + 1e-6)
        # Normalize per image
        N, C, H, W = edge.shape
        max_val = edge.view(N, -1).max(dim=1)[0].view(N,1,1,1)
        edge = edge / (max_val + 1e-6)
        return edge

    def mean_field_update(self, unary, guidance, final_logits=False):
        Q = F.softmax(unary, dim=1)
        for i in range(self.num_iterations):
            spatial_msg = self.spatial_conv(Q)
            bilateral_in = torch.cat([Q, guidance], dim=1)
            bilateral_msg = self.bilateral_conv(bilateral_in)
            message = self.compat_conv(spatial_msg + bilateral_msg)
            updated = unary - message
            if i == self.num_iterations - 1 and final_logits:
                Q = updated
            else:
                Q = F.softmax(updated, dim=1)
        return Q

    def forward(self, logits, image, skip_features=None, return_logits=False):
        target_size = logits.shape[2:]
        # Fuse skip-level features
        if skip_features is not None:
            fused_skip = self.fuse_skip_features(skip_features, target_size)
        else:
            fused_skip = torch.zeros((image.shape[0], 3, *target_size), device=image.device)
        # Compute edge guidance
        edge_map = self.compute_edge_map(image)
        # Build guidance tensor: RGB + fused skips + edge
        guidance = torch.cat([image, fused_skip, edge_map], dim=1)

        # --- Full-resolution update ---
        Q_full = self.mean_field_update(logits, guidance, final_logits=return_logits)

        # --- Half-resolution branch ---
        logits_h = F.interpolate(logits, scale_factor=0.5, mode='bilinear', align_corners=True)
        guidance_h = F.interpolate(guidance, scale_factor=0.5, mode='bilinear', align_corners=True)
        Q_h = self.mean_field_update(logits_h, guidance_h, final_logits=return_logits)
        Q_h_up = F.interpolate(Q_h, size=target_size, mode='bilinear', align_corners=True)

        # --- Quarter-resolution branch ---
        logits_q = F.interpolate(logits, scale_factor=0.25, mode='bilinear', align_corners=True)
        guidance_q = F.interpolate(guidance, scale_factor=0.25, mode='bilinear', align_corners=True)
        Q_q = self.mean_field_update(logits_q, guidance_q, final_logits=return_logits)
        Q_q_up = F.interpolate(Q_q, size=target_size, mode='bilinear', align_corners=True)

        # Fuse three scales equally
        return (Q_full + Q_h_up + Q_q_up) / 3.0