import torch
import torch.nn as nn
import torch.nn.functional as F
from Residual_block import ResidualBlock

class Generator(nn.Module):
    def __init__(self, ngf, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, 7),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU()]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)