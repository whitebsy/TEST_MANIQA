import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

'''真正的网络模型结构'''

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  #vit on small datasets中提出的LSA方法
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x): #(4,198*512)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SPT(nn.Module):
    """
    dim = 768
    pathc = 16
    image_size = 224
    """
    def __init__(self, *, dim, patch_size, channels=3, image_size, flateen = True):
        super().__init__()
        self.in_dim = channels
        self.dim = dim
        self.flatten = flateen

        #patch paritition
        self.conv = nn.Conv2d(self.in_dim,self.dim,kernel_size=patch_size,stride=patch_size,padding=0)

        self.to_patch_linear = nn.Sequential(
            # #x: [b, 768, 14, 70(14*5)]
            # Rearrange('b c h w  -> b (h w) c', h=image_size // patch_size, w=image_size // patch_size),  # ++？是干什么的
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        """
        x : [b, 3, 224, 224]  B, C, H, W
        """
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))

        # x: [b, 3, 224, 224]  -->  [b, 3, 224, 224*5]
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)

        # [b, 3, 224, 224*5] -->  [b, 768, 14, 70(14*5)]
        x_sample = self.conv(x_with_shifts)

        #[b, 768, 14, 70(14*5)] --> [b,  980, 768]  B,C,H,W --> B,(h*w),C
        x_sample = x_sample.flatten(2).transpose(1,2)

        # out: [b, 980, 768]
        out = self.to_patch_linear(x_sample)
        return out

#对于小型数据集适用的VIT模型
class ViT_base(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """
        image_size = 224
        patch_size = 16
        dim = 768
        depth = 6
        heads = 16
        mpl_dim = 768
        """
        super().__init__()
        image_height, image_width = pair(image_size) #224 224
        patch_height, patch_width = pair(patch_size) #16 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be ' \
                                                                                    'divisible by the patch size. '

        num_patches = (image_height // patch_height) * (image_width // patch_width)  #14 * 14
        patch_dim = channels * patch_height * patch_width   # 每个patch的通道数
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels, image_size=image_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, img):
        """
        img: [3, 224, 224]  c, h, w
        """
        # [b, 3, 224, 224] --> [b, 980, 768]   b,c,h,w --> b,(h*w),c
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 1:, :]  # ++

        '''结束'''
        x = rearrange(x, 'b (h w) c -> b c h w',
                      h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)  # ++
        return x
