import torch
import timm
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from Small_vit import vit_small

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


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.H = input_resolution
        self.W = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: B, H*W, C   [b, 7, 7, 1024]
        """
        H = self.H
        W = self.W
        x = rearrange(x, 'b c h w -> b (h w) c')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)            # 归一化
        x = self.reduction(x)       # 降维，通道降低的两倍
        x = self.dropout(x)
        x = rearrange(x, 'b (h w) c -> b c h w ', h=7, w=7)

        return x

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
        for _ in range(depth):    #depth:6   即在transformer中有6次 LSA 和 Feed Forward
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x    # LSA
            x = ff(x) + x    # Feed Forward
        return x


class teset_MANIQA(nn.Module):
    def __init__(self, embed_dim = 768, num_outputs = 1, patch_size = 16, drop = 0.2, # dim需要改变
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()

        self.img_size = img_size                                                            # 224
        self.patch_size = patch_size                                                        # 16
        self.input_size = img_size // patch_size                                            # 14
        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        # self.spt = SPT(dim=embed_dim, patch_size=patch_size, image_size=img_size)
        self.dropout = nn.Dropout(0.2)
        self.transformerEoder = Transformer(embed_dim,depth = 6,dim_head=64, heads=16, mlp_dim=2048)

        # 特征数需要修改
        self.feature_num = self.patch_size * self.patch_size * 3  # 14 * 14 * 5
        self.out1_feature = 512
        self.out2_feature = 256

        # 降通道
        self.conv = nn.Conv2d(15, 3, 1, 1, 0)
        self.conv1 = nn.Conv2d(self.feature_num * 3, self.feature_num, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.feature_num, self.out2_feature, 1, 1, 0)

        # part 2
        self.small_vit = vit_small(      # small_vit模型，只有LSA+FeedForward模块
            image_size=img_size,
            patch_size=16,
            # num_classes=1000,
            dim=self.out1_feature,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

        # part 3
        # 添加的patch_merging模块，用于降维
        self.patch_merging = PatchMerging(input_resolution=self.input_size,  dim=self.out1_feature
        )

        # part4
        self.fc_1 = nn.Sequential(
            nn.Linear(self.out2_feature, self.out2_feature),  # ++
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.out2_feature, 1),
        )


    def forward(self, x):

        # 如果直接使用vit预训练模型，需要输入为[3, 224, 224]不可以更改
        #[4, 3, 224, 224] -- > [4, 196, 768]
        x0 = self.vit(x[0]).cuda()   # [b, 14*14, 768*5]
        x1 = self.vit(x[1]).cuda()
        x2 = self.vit(x[2]).cuda()

        # x: [b, 196, 768]  -->  [b, 196, 768  * 3]
        x = torch.cat((x0, x1, x2), dim=2)

        # x : [b, 196, 768  * 3]  --> [b, 768 * 3 , 14, 14]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        # x : [b, 768 * 3 , 14, 14] -->  [b, 768, 14, 14]  -->  [b, 196, 768]
        x = self.conv1(x)   #
        x = self.dropout(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x = self.transformerEoder(x)

        #x: [b, 196, 768] ->   [b, 768, 14, 14]  -> [b, 256, 14, 14]  -> [b, 196, 256]
        x = rearrange(x, 'b (h w) c -> b c h w',h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        x = self.dropout(x)


        x = x[:, 0, :]

        score = self.fc_1(x)

        return score
