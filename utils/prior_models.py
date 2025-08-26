import torch
import torch.nn as nn

class UnetScoreNetwork(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        channels = [64, 128, 256, 256]

        self.first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_conv1 = self._down_block(channels[0], channels[1], padding=0)
        self.down_conv2 = self._down_block(channels[1], channels[2], padding=1)
        self.down_conv3 = self._down_block(channels[2], channels[3], padding=0)

        self.up_conv3 = self._up_block(channels[3], channels[2], padding=1)
        self.up_conv2 = self._up_block(channels[2] * 2, channels[1], padding=0)
        self.up_conv1 = self._up_block(channels[1] * 2, channels[0], padding=1)

        self.last_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, t=None, eps=1e-3, device="cpu"):
        x = x.reshape(x.shape[0], 1, 14, 14)
        if t is None:
            t = eps * torch.ones(x.shape[0], 1, device=device)
        t = t[..., None, None].expand(t.shape[0], 1, 14, 14)

        enc1 = self.first_conv(torch.cat((x, t), dim=1))
        enc2 = self.down_conv1(enc1)
        enc3 = self.down_conv2(enc2)

        bottleneck = self.up_conv3(self.down_conv3(enc3))

        dec3 = torch.cat((bottleneck, enc3), dim=1)
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.up_conv1(dec2)
        dec1 = self.last_conv(dec1)

        return dec1.reshape(dec1.shape[0], -1)


    @staticmethod
    def _down_block(in_channels, out_channels, padding):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.LogSigmoid())

    @staticmethod
    def _up_block(in_channels, out_channels, padding):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=padding),
            torch.nn.LogSigmoid())


class UnetScoreNetwork_SR(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        channels = [32, 64, 128, 256, 256]

        self.first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_conv1 = self._down_block(channels[0], channels[1], padding=0)
        self.down_conv2 = self._down_block(channels[1], channels[2], padding=0)
        self.down_conv3 = self._down_block(channels[2], channels[3], padding=1)
        self.down_conv4 = self._down_block(channels[3], channels[4], padding=0)

        self.up_conv4 = self._up_block(channels[4], channels[3], padding=1)
        self.up_conv3 = self._up_block(channels[3] * 2, channels[2], padding=0)
        self.up_conv2 = self._up_block(channels[2] * 2, channels[1], padding=1)
        self.up_conv1 = self._up_block(channels[1] * 2, channels[0], padding=1)

        self.last_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, t=None, eps=1e-2, device="cpu"):
        x = x.reshape(x.shape[0], 1, 28, 28)
        if t is None:
            t = eps * torch.ones(x.shape[0], 1, device=device)
        t = t[..., None, None].expand(t.shape[0], 1, 28, 28)

        enc1 = self.first_conv(torch.cat((x, t), dim=1))
        enc2 = self.down_conv1(enc1)
        enc3 = self.down_conv2(enc2)
        enc4 = self.down_conv3(enc3)

        bottleneck = self.up_conv4(self.down_conv4(enc4))

        dec4 = torch.cat((bottleneck, enc4), dim=1)
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.up_conv1(dec2)
        dec1 = self.last_conv(dec1)

        return dec1.reshape(dec1.shape[0], -1)


    @staticmethod
    def _down_block(in_channels, out_channels, padding):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=padding),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.LogSigmoid())

    @staticmethod
    def _up_block(in_channels, out_channels, padding):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=padding),
            torch.nn.LogSigmoid())