import torch
import torch.nn as nn


class Img2PatchEmbed(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super(Img2PatchEmbed, self).__init__()

        self.num_patches = (emb_dim // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=emb_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (batch_size, emb_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, emb_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, emb_dim)

        return x


class LearnablePositionalEmbedding(nn.Module):
    def __int__(self, num_patches, emb_dim):
        super(LearnablePositionalEmbedding, self).__init__()

        self.num_patches = num_patches
        self.emb_dim = emb_dim
        self.pos_emb = nn.Parameter(torch.randn(self.num_patches, emb_dim), requires_grad=True)

    def forward(self):

        return self.pos_emb




