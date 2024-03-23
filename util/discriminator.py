import math
import torch
from torch import nn
from einops import rearrange
from inspect import isfunction


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, mult = 2, norm = True):
        super().__init__()


        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

class generator(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                #Upsample(dim_out) if not is_last else nn.Identity(),
                ConvNextBlock(dim_out, dim_out),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                ConvNextBlock(dim_out, dim_out),
                ConvNextBlock(dim_out, dim_out),
            ]))

        self.compress = nn.Conv2d(dims[-1],3,1)

    def forward(self, x):
        for convnext, convnext2, attn, convnext4, convnext5, attn2, convnext6, convnext7 in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = attn(x)
            # x = upsample(x)
            x = convnext4(x)
            x = convnext5(x)
            x = attn2(x)
            x = convnext6(x)
            x = convnext7(x)

        return self.compress(x)

class discriminator_v4(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
                ConvNextBlock(dim_out, dim_out, norm=ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            ]))

        self.compress = nn.Conv2d(dims[-1],1,1)
        # self.out = nn.Linear(256, 1)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        for convnext, convnext2, attn, downsample, convnect3, convnext4, attn2 in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = attn(x)
            x = downsample(x)
            x = convnect3(x)
            x = convnext4(x)
            x = attn2(x)

        x = self.compress(x)
        x = torch.flatten(x, start_dim=1)
        return self.out(x)
class discriminator_v3(nn.Module):
    def __init__(
        self,
        image_size,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
    ):
        super().__init__()
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        self.compress = nn.Conv2d(dims[-1],1,1)

        # last_size = (image_size // (2 ** (len(dim_mults) - 1)))**2
        # self.out = nn.Linear(last_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for convnext, convnext2, downsample in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = downsample(x)
        x = self.compress(x)
        x = torch.flatten(x, start_dim=1)
        return self.sigmoid(x)


if __name__ == "__main__":
    # d = discriminator_v2("cuda:0" , 3,64)
    d = discriminator_v3(image_size=128, dim=32,    # 16 ==> 128
        dim_mults=(1, 2, 8, 16),     # 16 ==> 128
        channels=3).to("cuda:1")

    input = torch.rand((8,3,128,128)).to("cuda:1")
    output = d(input)

    print(output.shape)
    print(output)     # 128:[1, 84] 256:[1,336]
