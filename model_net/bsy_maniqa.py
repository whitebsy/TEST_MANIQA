import torch
import timm
import small_vit
import torch.nn.functional as F
# from bsy_swin import SwinTransformer
from torch import nn
from small_vit import ViT
from einops import rearrange
from vit_base import ViT_base

class SPT(nn.Module):
    """
    Spatial Transformation
    """
    def __init__(self, *, dim, patch_size, channels=3, image_size):  #dim = 512(vit 1)
        super().__init__()

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        # x:[b, 3, 224, 224] -->  [b, 3*5, 224, 224]
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)  #横着拼

        # return self.to_patch_tokens(x_with_shifts)   #输出 (197,512)
        return x_with_shifts

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
        self.W = input_resolution * 5
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = self.H
        W = self.W
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

        x = self.norm(x)
        x = self.reduction(x)

        return x

class MANIQA(nn.Module):
    def __init__(self, embed_dim = 768, num_outputs = 1, patch_size = 16, drop = 0.1,#dim需要改变
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()

        self.img_size = img_size   #224
        self.patch_size = patch_size     #16
        self.input_size = img_size // patch_size     #14
        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.spt = SPT(dim = embed_dim, patch_size= patch_size, image_size= img_size)

        # self.vit_base = ViT_base(image_size = img_size,   #224
        #                          patch_size = patch_size,    #16
        #                          dim = embed_dim,     #768
        #                          depth = 6,
        #                          heads = 16,
        #                          mlp_dim = dim_mlp,
        #                          pool='cls',
        #                          channels=3)

        # 特征数需要修改
        self.feature_num = self.input_size * self.input_size * 5  # 14 * 14 * 5
        self.out1_feature = 512
        self.out2_feature = 256

        # part 1
        # self.spatial_transformation = SPT(dim = embed_dim, patch_size = patch_size, img_size = img_size)   #SPT变换
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)     #使用的预训练的VIT模型

        # 降通道
        self.conv1 = nn.Conv2d(self.feature_num, self.out1_feature, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.out1_feature * 2, self.out2_feature, 1, 1, 0)

        # part 2
        self.small_vit =  ViT(      #small_vit模型，只有LSA+FeedForward模块
            image_size=img_size,
            patch_size=16,
            num_classes=1000,
            dim=self.out1_feature,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

        #part 3
        #添加的patch_merging模块，用于降维
        self.patch_merging = PatchMerging(input_resolution=self.input_size,  dim=self.out1_feature
        )

        # part4
        self.fc_1 = nn.Sequential(
            nn.Linear(self.out2_feature, self.out2_feature),  # ++
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.out2_feature, 1),
        )


    def forward(self, x):   # x : [b, 3, 224 ,224]
        # 对输入进行SPT 变换后再输入vit模型
        # x : [b,3, 224, 224] ---> [b, 3*5, 224, 224]
        x_0 = self.spt(x[0])
        x_1 = self.spt(x[1])
        x_2 = self.spt(x[2])

        #x: [b, 15, 224, 224] --> [b, 14, 14, 15*16*16] --> [b, 196, 5*768]
        x0 = self.vit(x_0).cuda()
        x1 = self.vit(x_1).cuda()
        x2 = self.vit(x_2).cuda()

        # 如果直接使用vit 预训练模型，需要输入为[3, 224, 224]不可以更改
        # x0 = self.vit_base(x[0]).cuda()   # [b, 14*14, 768 * 5]
        # x1 = self.vit_base(x[1]).cuda()
        # x2 = self.vit_base(x[2]).cuda()

        #x: [b, 196, 5*768]  -->  [b, 196, 768 * 5 * 3]
        x = torch.cat((x0, x1, x2), dim=2)

        # stage 1
        # x: [b, 196, 768 * 5 * 3] --> [b, 768 * 5 * 3, 14, 14]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        # x: [b, 768 * 5 * 3, 14, 14] --> [b, 512, 14,14]
        x = self.conv1(x)   #是否还需要？

        # x: [b, 512, 14,14] --> [b, 196, 512]
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)

        # x: [b, 196, 512] --> [b, 512, 14,14]
        x = self.small_vit(x)

        # stage2   进行下采样
        x = self.patch_merging(x)   # [b, 7, 7, 1024]
        x = self.conv2(x)     # [b, 7, 7, 256]
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size // 2, w=self.input_size // 2)

        # _, _, h, w = x.shape
        # x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)  # [4, 196, 384]

        x = x[:, 0, :]

        score = self.fc_1(x)

        return score
