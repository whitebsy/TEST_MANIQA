import torch
from torch import nn
import torch.nn.functional as F
class patch_merging():
    def __int__(self, dim,norm_layer = nn.LayerNorm):
        super().__int__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim,2*dim,bias=False)
        self.norm = norm_layer(4*dim)

    def forwaed(self,x,H,W):
        """
        x: B, H*W ,C
        """
        B,L,C = x.shape
        assert L == H * W ,"input feature has wrong size"

        x = x.view(B,H,W,C)

        #padding ,如何输入feature map的H W不是2的整倍数，需要进行padding
        pad_input = (H % 2 ==1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x,(0,0,0,W%2,0,H%2))

        #进行分块
        x0 = x[:, 0::2, 0::2, :]    # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]    # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]    # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]    # B, H/2, W/2, C

        x = torch.cat([x0,x1,x2,x3],-1)  #在通道维度上进行concat   B, H/2, W/2, C*4
        # dim = -1 是指在最后一个维度上进行concat
        x = x.view(B,-1, 4*C)     #在高度和宽度方向上进行展平   B , H/2 * W/2, C*4
        # x.view(,-1,)表示在-1位置的数的自行计算
        x = self.norm(x)
        x = self.reduction(x)   #linear   B , H/2 * W/2, C*2

        return x