from torch import nn


class ConvBlock(nn.Sequential):
    """Conv + Activation"""
    def __init__(self, in_channels, mid_channels, out_channels=None, nb_conv=2, kernel_size=3):
        out_channels = out_channels or mid_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(inplace=False),
        )

    def forward_skip(x): #x(16)
        """Use this function when we want to save the last pre-activation value"""
        second_to_last = None
        for layer in self:
            second_to_last = x
            x = layer(x)
        return x, second_to_last


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
        )
        self.bottleneck = ConvBlock(32, 64, 32)
        self.decoder = nn.Sequential(
            ConvBlock(64, 16),
            ConvBlock(32, 8),
            ConvBlock(16, 8),
        )
        self.conv_last = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(x): # x(16)
        skips = []
        for layer in self.encoder:
            x, skip = layer.forward_skip(x)
            x = self.maxpool(x)
            skips += [skip]
            
        x = self.bottleneck(x)
        
        for layer, skip in zip(self.decoder, skips):
            x = torch.cat([self.upsample(x), skip], dim=1)
            x = layer(x)
        x = self.conv_last(x)
        return x
            
