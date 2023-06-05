import torch.nn as nn
import torch
from utils import Img2PatchEmbed, ViTBlock


class ViT(nn.Module):
    def __init__(self, in_channels, img_size, emb_dim, num_heads, patch_size, depth, mlp_ratio, num_classes):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.emb_dim = emb_dim
        self.depth = depth
        self.patch_size = patch_size
        self.img_emb = Img2PatchEmbed(in_channels, img_size, emb_dim, patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim), requires_grad=True)
        self.classifier = nn.Linear(emb_dim, num_classes)

        self.vit_blocks = nn.ModuleList([
            ViTBlock(emb_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, x):
        N, _, _, _ = x.shape
        x = self.img_emb(x)  # (N, num_patches, emb_dim)
        cls_token = self.cls_token.expand(
            N, -1, -1
        )  # (N, 1, emb_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_emb
        for vit_block in self.vit_blocks:
            x = vit_block(x)

        cls_token = x[:, 0]
        scores = self.classifier(cls_token)

        return scores
