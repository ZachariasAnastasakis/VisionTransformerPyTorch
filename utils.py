import torch
import torch.nn as nn


class Img2PatchEmbed(nn.Module):
    def __init__(self, in_channels, img_size, emb_dim, patch_size):
        super(Img2PatchEmbed, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
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


# class LearnablePositionalEmbedding(nn.Module):
#     def __int__(self, num_patches, emb_dim):
#         super(LearnablePositionalEmbedding, self).__init__()
#
#         self.num_patches = num_patches
#         self.emb_dim = emb_dim
#         self.pos_emb = nn.Parameter(torch.randn(self.num_patches, emb_dim), requires_grad=True)
#
#     def forward(self):
#         return self.pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.emb_idm = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_idm // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.num_heads * self.head_dim != self.emb_idm:
            raise ValueError("Embedding dim must be divisible by the number of heads.")

        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)

        self.head_projection = nn.Linear(self.num_heads * self.head_dim, self.emb_idm)

    def forward(self, x):
        # x -> (batch_size, num_patches, emb_dim)
        batch_size, num_patches, _ = x.shape
        query, key, value = x, x, x

        q_x = query.reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        k_x = key.reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        v_x = value.reshape(batch_size, num_patches, self.num_heads, self.head_dim)

        q_x = self.query(q_x)
        k_x = self.key(k_x)
        v_x = self.value(v_x)

        dot_product = torch.matmul(q_x, k_x.transpose(-1, -2))
        attention = nn.Softmax(-1)(dot_product * self.scale)

        scaled_dot_product_attention_per_head = torch.matmul(attention, v_x)
        scaled_dot_product_attention = scaled_dot_product_attention_per_head.flatten(2)

        multihead_attention = self.head_projection(scaled_dot_product_attention)

        return multihead_attention


class MLP(nn.Module):
    def __init__(self, emb_dim, mlp_ratio):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim), nn.GELU(),
            nn.Linear(emb_dim * mlp_ratio, emb_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(ViTBlock, self).__init__()

        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual_1 = self.multihead_attention(self.norm1(x)) + x
        residual_2 = self.mlp(self.norm2(x)) + residual_1

        return residual_2