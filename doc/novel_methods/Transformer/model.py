import torch
import torch.nn as nn
from functools import partial


class Embedding(nn.Module):
    def __init__(self, n_features=34, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(n_features, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        B, P, L = x.shape   # (batch_size, number of features per time step, sequence length)
        # transpose [B, P, L] -> [P, B, L]
        # flatten [P, B, L] -> [P, BL]
        # transpose [P, BL] -> [BL, P]
        x = x.transpose(1, 0).flatten(1).permute(1,0)

        # project [BL, P] -> [BL, embed_dim]
        # reshape [BL, embed_dim] -> [B, L, embed_dim]
        x = self.proj(x)
        x = self.norm(x)
        x = x.reshape(B, L, self.embed_dim)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.attn = None # added

    def forward(self, x):
        # [batch_size, length+1, embed_dim]
        B, L, P = x.shape

        # flatten [batch_size, length + 1, embed_dim] -> [batch_size * (length+1), embed_dim]
        # qkv [batch_size*(length+1), embed_dim] -> [batch_size*(length+1), 3*embed_dim]
        x = self.qkv(x.flatten(start_dim=0, end_dim=1))

        # reshape [batch_size*(length+1), 3*embed_dim] -> [batch_size*(length+1), 3, num_heads, embed_dim_per_head]
        # reshape [batch_size*(length+1), 3, num_heads, embed_dim_per_head] -> [batch_size, length+1, 3, num_heads, embed_dim_per_head]
        # permute [batch_size, length+1, 3, num_heads, embed_dim_per_head] -> [3, batch_size, num_heads, length+1, embed_dim_per_head]
        x = x.reshape(-1, 3, self.num_heads, P // self.num_heads)
        x = x.reshape(B, L, 3, self.num_heads, P // self.num_heads)
        x = x.permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        # transpose [batch_size, num_heads, length+1, embed_dim_per_head] -> [batch_size, num_heads, embed_dim_per_head, length+1]
        # @ -> [batch_size, num_heads, length+1, length+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn = attn   # for extracting frame attention in the future # Use this!

        # @ -> [batch_size, num_heads, length+1, embed_dim_per_head]
        # transpose [batch_size, num_heads, length+1, embed_dim_per_head] -> [batch_size, lenght+1, num_heads, embed_dim_per_head]
        # reshape [batch_size, lenght+1, num_heads, embed_dim_per_head] -> [batch_size, lenght+1, embed_dim]
        # reshape [batch_size, lenght+1, embed_dim] -> [batch_size*lenght+1, embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, L, P).flatten(start_dim=0, end_dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        # reshape [batch_size*lenght+1, embed_dim] -> [batch_size, length+1, embed_dim]
        x = x.reshape(B, L, P)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim, num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """This is a Transformer encoder"""
    def __init__(self, n_features=34, num_classes=5, embed_dim=128, depth=3,
                 num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 embed_layer=Embedding, act_layer=None, norm_layer=None):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.embedding = embed_layer(n_features, embed_dim)
        self.length = 100

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.length + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        # weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_transformer_weights)


    def forward_features(self, x):
        # [batch_size, n_features, length] -> [batch_size, length, embed_dim]
        x = self.embedding(x)

        #[1, 1, 128] -> [B, 1, 128]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)   # [B, 101, 128]

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_transformer_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def transformer_base(n_features=34):
    model = Transformer(n_features=n_features, embed_dim=128, depth=8, num_heads=8,
                        drop_ratio=0.2, attn_drop_ratio=0.2)
    return model

def transformer_large(n_features=34):
    model = Transformer(n_features=n_features, embed_dim=256, depth=12, num_heads=16,
                        drop_ratio=0.2, attn_drop_ratio=0.2)
    return model

def transformer_huge (n_features=34):
    model = Transformer(n_features=n_features, embed_dim=512, depth=16, num_heads=32,
                        drop_ratio=0.2, attn_drop_ratio=0.2)
    return model