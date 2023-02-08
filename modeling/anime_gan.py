import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from modeling.conv_blocks import ConvNormLReLU
from modeling.conv_blocks import InvertedResBlock
from utils.common import initialize_weights

class Generator(nn.Module):
    def __init__(self, dataset=''):
        super(Generator, self).__init__()
        self.name = f'generator_{dataset}'
        
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0,1,0,1)),
            ConvNormLReLU(64, 64)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0,1,0,1)),            
            ConvNormLReLU(128, 128)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )    
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        initialize_weights(self.out_layer)
        
    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)
        
        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", 
                                align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", 
                                align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", 
                                align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", 
                                align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out

class Discriminator(nn.Module):
    def __init__(self,  args):
        super(Discriminator, self).__init__()
        self.name = f'discriminator_{args.dataset}'
        self.bias = False
        channels = 32
        out = 32
        
        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(1, args.d_layers):
            layers += [
                nn.Conv2d(channels, out * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out * 2, out * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
                nn.LayerNorm(out * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4
            out *= 2

        layers += [
            nn.Conv2d(out * 2, out * 2, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LayerNorm(out * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out * 2, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if args.use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, img):
        return self.discriminate(img)