# Multi-Sequence Fourier Fusion
import random

import torch.nn as nn
import torch.nn.functional as F


class MSFF_basic(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=1, nb_channels: int=64, n_seqs: int=2) -> None:
        super().__init__()
        self.n_seqs = n_seqs
        self.nb_channels = nb_channels

        self.encoders = Encoder(in_channels, nb_channels)
        self.decoders = nn.ModuleList()
        for _ in range(n_seqs):
            self.decoders.append(Decoder(out_channels, nb_channels))
        
        self.dec_mask = MaskDecoder(3, nb_channels)
        self.dec_gif = MaskDecoder(209, nb_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, seqs):
        f_seqs = self.encoders(seqs)[0]
        
        outs = []
        for i in range(self.n_seqs):
            outs.append(self.decoders[i](f_seqs))

        seg = self.dec_mask(f_seqs)
        gif = self.dec_gif(f_seqs)

        return {
            'out': outs,
            'mask': seg,
            'gif': gif,
            'f_seqs': f_seqs,
        }



class MaskDecoder(nn.Module):
    def __init__(self, out_channels: int=1, nb_channels: int=64) -> None:
        super().__init__()

        self.res = nn.Sequential(
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convt1 = nn.Conv2d(nb_channels*4, nb_channels*2, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(nb_channels*2)
        self.act = nn.LeakyReLU(0.2, True)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convt2 = nn.Conv2d(nb_channels*2, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(nb_channels)
        self.convt3 = nn.Conv2d(in_channels=nb_channels, out_channels=out_channels, kernel_size=1, padding=0)
        #self.act_last = nn.Sigmoid()
    
    def forward(self, x):
        x = self.res(x)

        x = self.up1(x)
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.up2(x)
        x = self.convt2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.convt3(x)
        #x = self.act_last(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int=1, nb_channels: int=64) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nb_channels, kernel_size=7, padding=3, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(nb_channels)
        self.act = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels*2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(nb_channels*2)
        self.conv3 = nn.Conv2d(in_channels=nb_channels*2, out_channels=nb_channels*4, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(nb_channels*4)

        self.res = nn.Sequential(
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.act(x)
        x = self.conv2(x1)
        x = self.norm2(x)
        x2 = self.act(x)
        x = self.conv3(x2)
        x = self.norm3(x)
        x3 = self.act(x)

        x = self.res(x3)
        
        return x, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, out_channels: int=3, nb_channels: int=64) -> None:
        super().__init__()

        self.res = nn.Sequential(
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
            ResnetBlock(nb_channels*4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True),
        )

        self.convt1 = nn.ConvTranspose2d(nb_channels*4, nb_channels*2, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(nb_channels*2)
        self.act = nn.LeakyReLU(0.2, True)
        self.convt2 = nn.ConvTranspose2d(nb_channels*2, nb_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(nb_channels)
        self.convt3 = nn.Conv2d(in_channels=nb_channels, out_channels=out_channels, kernel_size=7, padding=3, padding_mode='reflect')
        self.act_last = nn.Tanh()
    
    def forward(self, x):
        x = self.res(x)

        x = self.convt1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.convt2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.convt3(x)
        x = self.act_last(x)
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, n_class, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, n_class, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        nw, nh = 224, 224
        w = random.randint(0, input.shape[-2]-nw-1)
        h = random.randint(0, input.shape[-1]-nh-1)
        return self.model(input[:,:,w:w+nw,h:h+nh])


class MSFKoosNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=256, head='mlp', feat_dim=128, num_classes=4):
        super(MSFKoosNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=dim_in, out_channels=dim_in*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=dim_in*2, out_channels=dim_in*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=dim_in*2, out_channels=dim_in*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=dim_in*2, out_channels=dim_in*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=dim_in*2, out_channels=dim_in*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(in_channels=dim_in*2, out_channels=dim_in*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_in*2),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        self.attention = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=dim_in, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(dim_in*2, num_classes)

        if head == 'linear':
            self.head1 = nn.Linear(dim_in*2, feat_dim)
            self.head2 = nn.Linear(dim_in*2, feat_dim)
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in*2, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in*2, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, pretrain=True):
        #x = torch.cat([x, mask], dim=1)
        #print(x.shape)
        x = x[:,:256] * self.attention(x[:,256:])
        feat = self.encoder(x)
        if pretrain:
            feat1 = F.normalize(self.head1(feat), dim=1)
            feat2 = F.normalize(self.head2(feat), dim=1)
            return feat1, feat2
        else:
            pred = self.classifier(feat)
            return pred