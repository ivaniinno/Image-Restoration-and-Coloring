import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.identity_map = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = inputs
        out = self.layer(x)
        residual = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class Generator(nn.Module):
    """ResUNet-like generator that maps L channel to ab channels.

    input: (batch, 1, H, W) grayscale L channel normalized to [0,1]
    output: (batch, 2, H, W) ab channels normalized to [0,1]
    """
    def __init__(self, input_channel=1, output_channel=2, dropout_rate=0.2):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel, 64)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, inputs):
        # Encoder
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)

        # Bridge
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)

        # Decoder
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)

        output = self.output(d1)
        output = torch.sigmoid(output)
        return output


class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, ab, l):
        img_input = torch.cat((ab, l), 1)
        output = self.model(img_input)
        return output


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def build_models(device=None, gen_weights: str = None, crit_weights: str = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(input_channel=1, output_channel=2)
    critic = Critic(in_channels=3)

    generator.apply(weights_init)
    critic.apply(weights_init)

    if gen_weights:
        generator.load_state_dict(torch.load(gen_weights, map_location=device))
    if crit_weights:
        critic.load_state_dict(torch.load(crit_weights, map_location=device))

    generator.to(device)
    critic.to(device)

    return generator, critic
