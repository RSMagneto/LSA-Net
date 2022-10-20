import pdb
import torch
import torch.nn as nn
from einops import rearrange
import transformer
import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),(q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out = self.proj(x).view(b, h, w, c)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class Long_Rnage_Attention_Module(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        b, c, hw = x.size()
        x = rearrange(x, 'b c (h w) -> b c h w', h=int(hw ** 0.5), w=int(hw ** 0.5))
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Short_Rnage_Attention_Module(nn.Module):
    def __init__(self, in_channels, sram_dim):
        super(Short_Rnage_Attention_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=sram_dim, kernel_size=7, stride=1, padding=3)
        self.conv7 = nn.Conv2d(in_channels=sram_dim*4, out_channels=sram_dim*2, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=sram_dim*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=1, bias=False),
            nn.Softmax(dim=2)
        )
        self.conv9 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = self.conv2(x)
        x2 = rearrange(x2, 'b c h w -> b c (h w)')
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        feature1 = torch.einsum('b n i, b j n -> b i j', x1, x2)
        feature1 = self.mlp(feature1)
        feature2 = torch.cat([x3, x4, x5, x6], dim=1)
        feature2 = self.conv7(feature2)
        feature2 = self.conv8(feature2)
        feature2 = rearrange(feature2, 'b c h w -> b c (h w)')
        out = feature1 * feature2
        b, c, hw = out.size()
        out = rearrange(out, 'b c (h w) -> b c h w', h = int(hw ** 0.5), w = int(hw ** 0.5))
        out = self.conv9(out)
        return out

class LSModule(nn.Module):
    def __init__(self, feature_dim, sram_dim):
        super(LSModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim*2, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=feature_dim*2, out_channels=feature_dim*2, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
        )
        self.long_range_attention_moudle = Long_Rnage_Attention_Module(dim=feature_dim*2,heads=4, dim_head=32)
        self.short_range_attention_moudle = Short_Rnage_Attention_Module(in_channels=feature_dim//2, sram_dim=sram_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.long_range_attention_moudle(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.short_range_attention_moudle(x1)
        x2 = self.short_range_attention_moudle(x2)
        x3 = self.short_range_attention_moudle(x3)
        x4 = self.short_range_attention_moudle(x4)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out

class LSAnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, feature_dim=16, sram_dim=8):
        super(LSAnet, self).__init__()
        self.pre_conv =  nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=feature_dim, kernel_size=3, padding=0, stride=1),
            nn.PReLU()
        )
        self.lsmodule = nn.Sequential(
            LSModule(feature_dim=feature_dim, sram_dim=sram_dim),
            LSModule(feature_dim=feature_dim * 2, sram_dim=sram_dim * 2)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim * 4, out_channels=feature_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=feature_dim * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

    def forward(self,x):
        x = self.pre_conv(x)
        x = self.lsmodule(x)
        x = self.refine(x)
        return x
