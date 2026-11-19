import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, 
                      padding="same", activation="relu", groups=1):
    """
    Construit une séquence [Conv2d -> BatchNorm2d -> Activation] avec gestion du padding.
    """
    if padding == "same":
        pad = dilation * (kernel_size - 1) // 2
    elif padding == "valid":
        pad = 0
    else:
        pad = padding
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=pad, dilation=dilation,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if activation is not None:
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class CFPModule(nn.Module):
    """
    Module CFP (Context Fusion and Propagation) adapté de la version Keras.
    On suppose ici que "filters" vaut 64. Ainsi :
      - La réduction initiale convertit l'entrée en filters//4 (16) canaux.
      - Chaque branche produit 8 canaux.
      - Pour que le BatchNorm attend 16 canaux, on concatène deux tenseurs (8+8).
    """
    def __init__(self, in_channels, filters, d_size):
        super(CFPModule, self).__init__()
        # Réduction initiale des canaux
        self.conv_inp = conv2d_bn_layer(in_channels, filters // 4, kernel_size=1,
                                          stride=1, dilation=1, padding="same",
                                          activation="relu")
        
        # Branche 1
        self.conv1_1 = conv2d_bn_layer(filters // 4, filters // 16, kernel_size=3,
                                       stride=1, dilation=1, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv1_2 = conv2d_bn_layer(filters // 16, filters // 16, kernel_size=3,
                                       stride=1, dilation=1, padding="same",
                                       activation="relu", groups=filters // 16)
        # Pour éviter le problème de divisibilité, on utilise groups = filters // 16
        self.conv1_3 = conv2d_bn_layer(filters // 16, filters // 8, kernel_size=3,
                                       stride=1, dilation=1, padding="same",
                                       activation="relu", groups=filters // 16)
        
        # Branche 2
        d_rate2 = d_size // 4 + 1
        self.conv2_1 = conv2d_bn_layer(filters // 4, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate2, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv2_2 = conv2d_bn_layer(filters // 16, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate2, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv2_3 = conv2d_bn_layer(filters // 16, filters // 8, kernel_size=3,
                                       stride=1, dilation=d_rate2, padding="same",
                                       activation="relu", groups=filters // 16)
        
        # Branche 3
        d_rate3 = d_size // 2 + 1
        self.conv3_1 = conv2d_bn_layer(filters // 4, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate3, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv3_2 = conv2d_bn_layer(filters // 16, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate3, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv3_3 = conv2d_bn_layer(filters // 16, filters // 8, kernel_size=3,
                                       stride=1, dilation=d_rate3, padding="same",
                                       activation="relu", groups=filters // 16)
        
        # Branche 4
        d_rate4 = d_size + 1
        self.conv4_1 = conv2d_bn_layer(filters // 4, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate4, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv4_2 = conv2d_bn_layer(filters // 16, filters // 16, kernel_size=3,
                                       stride=1, dilation=d_rate4, padding="same",
                                       activation="relu", groups=filters // 16)
        self.conv4_3 = conv2d_bn_layer(filters // 16, filters // 8, kernel_size=3,
                                       stride=1, dilation=d_rate4, padding="same",
                                       activation="relu", groups=filters // 16)
        
        # BatchNorm pour chaque branche – ils attendent filters//4 canaux (16 canaux si filters=64)
        self.bn_o1 = nn.BatchNorm2d(filters // 4)
        self.bn_o2 = nn.BatchNorm2d(filters // 4)
        self.bn_o3 = nn.BatchNorm2d(filters // 4)
        self.bn_o4 = nn.BatchNorm2d(filters // 4)
        
        # Fusion finale
        self.bn_out = nn.BatchNorm2d(filters)
        self.conv_out = conv2d_bn_layer(filters, filters, kernel_size=1,
                                        stride=1, dilation=1, padding="valid",
                                        activation="relu")

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x_inp = self.conv_inp(x)
        
        # Branche 1
        x1 = self.conv1_1(x_inp)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)  # x1 shape: [B, 8, H, W] (pour filters=64)
        # Branche 2
        x2 = self.conv2_1(x_inp)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)  # [B, 8, H, W]
        # Branche 3
        x3 = self.conv3_1(x_inp)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)  # [B, 8, H, W]
        # Branche 4
        x4 = self.conv4_1(x_inp)
        x4 = self.conv4_2(x4)
        x4 = self.conv4_3(x4)  # [B, 8, H, W]
        
        # Pour obtenir 16 canaux, concaténation de deux copies de x1 pour la branche 1
        o1 = self.bn_o1(torch.cat([x1, x1], dim=1))  # [B, 16, H, W]
        o2 = self.bn_o2(torch.cat([x2, x2], dim=1))    # [B, 16, H, W]
        o3 = self.bn_o3(torch.cat([x1, x3], dim=1))     # [B, 16, H, W]
        o4 = self.bn_o4(torch.cat([x1, x4], dim=1))     # [B, 16, H, W]
        
        ad1 = o1
        ad2 = ad1 + o2
        ad3 = ad2 + o3
        ad4 = ad3 + o4
        # Concaténation finale : 4 * 16 = 64 canaux
        out_cat = torch.cat([ad1, ad2, ad3, ad4], dim=1)
        out_cat = self.bn_out(out_cat)
        out_conv = self.conv_out(out_cat)
        # Le résidu final suppose que x possède 64 canaux (sinon, il faut le projeter)
        out = out_conv + x
        return out


class CFPNetM(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): Nombre de canaux de l'image d'entrée (ex: 3 pour RGB).
            num_classes (int): Nombre de classes à segmenter.
        """
        super(CFPNetM, self).__init__()
        # Extraction initiale
        self.conv1 = conv2d_bn_layer(in_channels, 32, kernel_size=3,
                                     stride=1, dilation=1, padding="same", activation="relu")
        self.conv2 = conv2d_bn_layer(32, 32, kernel_size=3,
                                     stride=1, dilation=1, padding="same", activation="relu")
        self.conv3 = conv2d_bn_layer(32, 32, kernel_size=3,
                                     stride=1, dilation=1, padding="same", activation="relu")
        # Skip connection pour décodage : fusion de conv3 et de l'entrée
        # Pour in_channels=3, cela donne 32 + 3 = 35 canaux.
        # On la calculera dans forward.
        
        # Injection 1
        self.inj1_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.inj1_bn = nn.BatchNorm2d(in_channels)
        self.inj1_relu = nn.ReLU(inplace=True)
        self.inj1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Fusion opt_cat1 pour CFP1 : concaténer conv3 et inj1
        self.opt_cat1_conv = conv2d_bn_layer(32 + in_channels, 64, kernel_size=3,
                                             stride=2, dilation=1, padding="same", activation="relu")
        
        # Bloc CFP1
        self.cfp1 = CFPModule(64, 64, d_size=2)
        self.cfp2 = CFPModule(64, 64, d_size=2)
        
        # Injection 2
        self.inj2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.inj2_bn = nn.BatchNorm2d(in_channels)
        self.inj2_relu = nn.ReLU(inplace=True)
        self.opt_cat2_conv = conv2d_bn_layer(64 + 64 + in_channels, 128, kernel_size=3,
                                             stride=2, dilation=1, padding="same", activation="relu")
        
        # Bloc CFP2
        self.cfp3 = CFPModule(128, 128, d_size=4)
        self.cfp4 = CFPModule(128, 128, d_size=4)
        self.cfp5 = CFPModule(128, 128, d_size=8)
        self.cfp6 = CFPModule(128, 128, d_size=8)
        self.cfp7 = CFPModule(128, 128, d_size=16)
        self.cfp8 = CFPModule(128, 128, d_size=16)
        
        # Injection 3
        self.inj3_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.inj3_bn = nn.BatchNorm2d(in_channels)
        self.inj3_relu = nn.ReLU(inplace=True)
        
        # Décodage
        # opt_cat3 est obtenu en concaténant :
        #   - cfp8_out : 128 canaux
        #   - opt_cat2_out : 128 canaux (issu de opt_cat2_conv)
        #   - inj3 : in_channels (pour RGB, 3 canaux)
        # Total = 128 + 128 + 3 = 259 canaux
        self.deconv4 = nn.ConvTranspose2d(259, 128, kernel_size=2, stride=2)
        # Après deconv4, on concatène avec opt_cat2_out (128 canaux) : 128+128 = 256 canaux
        self.deconv5 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        # Pour la dernière étape, on concatène la sortie de deconv5 (64 canaux)
        # avec la skip connection initiale (skip1, 35 canaux) : 64+35 = 99 canaux.
        # On supprime la dernière transposed convolution pour ne pas doubler la résolution.
        self.final_conv = conv2d_bn_layer(99, num_classes, kernel_size=1,
                                          stride=1, dilation=1, padding="valid", activation=None)

    def forward(self, x, training=False, memory_frames=None, memory_masks=None):
        # Extraction initiale
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # [B, 32, H, W]
        # Skip connection : concaténer conv3 et l'entrée x -> [B, 32 + in_channels, H, W]
        skip1 = torch.cat([conv3, x], dim=1)  # pour in_channels=3, [B,35,H,W]
        
        # Injection 1
        inj1 = self.inj1_pool(x)
        inj1 = self.inj1_bn(inj1)
        inj1 = self.inj1_relu(inj1)
        inj1 = self.inj1_upsample(inj1)  # [B, in_channels, H, W]
        
        # Fusion opt_cat1 pour CFP1
        opt_cat1 = torch.cat([conv3, inj1], dim=1)  # [B, 32+in_channels, H, W]
        opt_cat1_out = self.opt_cat1_conv(opt_cat1)   # [B,64, H/2, W/2]
        
        # Bloc CFP1
        cfp1_out = self.cfp1(opt_cat1_out)
        cfp2_out = self.cfp2(cfp1_out)  # [B,64,H/2,W/2]
        
        # Injection 2
        inj2 = self.inj2_pool(inj1)  # [B, in_channels, H/2, W/2]
        inj2 = self.inj2_bn(inj2)
        inj2 = self.inj2_relu(inj2)
        
        # Fusion opt_cat2
        opt_cat2 = torch.cat([cfp2_out, opt_cat1_out, inj2], dim=1)  # [B,64+64+in_channels, H/2, W/2]
        opt_cat2_out = self.opt_cat2_conv(opt_cat2)   # [B,128, H/4, W/4]
        
        # Bloc CFP2
        cfp3_out = self.cfp3(opt_cat2_out)
        cfp4_out = self.cfp4(cfp3_out)
        cfp5_out = self.cfp5(cfp4_out)
        cfp6_out = self.cfp6(cfp5_out)
        cfp7_out = self.cfp7(cfp6_out)
        cfp8_out = self.cfp8(cfp7_out)  # [B,128,H/4,W/4]
        
        # Injection 3
        inj3 = self.inj3_pool(inj2)  # [B, in_channels, H/4, W/4]
        inj3 = self.inj3_bn(inj3)
        inj3 = self.inj3_relu(inj3)
        
        # Fusion opt_cat3 : concaténation de cfp8_out (128), opt_cat2_out (128) et inj3 (in_channels=3) → 259 canaux
        opt_cat3 = torch.cat([cfp8_out, opt_cat2_out, inj3], dim=1)  # [B,259,H/4,W/4]
        
        # Décodage
        deconv4 = self.deconv4(opt_cat3)  # [B,128, H/2, W/2]
        opt_cat2_up = F.interpolate(opt_cat2_out, size=deconv4.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([deconv4, opt_cat2_up], dim=1)  # [B,128+128=256, H/2, W/2]
        deconv5 = self.deconv5(up1)  # [B,64, H, W]
        skip1_up = F.interpolate(skip1, size=deconv5.shape[2:], mode='bilinear', align_corners=False)  # [B,35,H,W]
        up2 = torch.cat([deconv5, skip1_up], dim=1)  # [B,64+35=99,H,W]
        out = self.final_conv(up2)  # [B, num_classes, H, W] (ici H et W devraient correspondre à la résolution cible, ex. 80×80)
        return out