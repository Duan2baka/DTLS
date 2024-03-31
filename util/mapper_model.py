from torch import nn
import torch
from einops import rearrange

def exists(x):
    return x is not None
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

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

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            nn.BatchNorm2d(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.ReLU(True),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h #+ self.res_conv(x)


class ConvNextBlock_LN(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

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

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, input):
        return self.main(input)

class Generator_v2(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(8, 4, 2, 1),
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
        
        self.out = nn.Conv2d(dims[-1], 3, 1)
        self.tanh = nn.Tanh()
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_in, dim_in, 4, 1, 0) if ind ==0 else nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1),
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                ConvNextBlock(dim_out, dim_out),                
            ]))
            

    def forward(self, x):
        for upsample, convnext, attn, convnext2 in self.downs:
            x = upsample(x)
            x = convnext(x)
            x = attn(x)
            x = convnext2(x)
        return self.tanh(self.out(x))

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        
class Discriminator_v2(nn.Module):
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

        self.sigmoid = nn.Sigmoid()
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
                
                nn.Conv2d(dim_out, 1, 1),
                nn.Linear((32//(2**(ind+1)))**2, 100) if not is_last else nn.Linear((32//(2**(ind)))**2, 100)
            ]))


    def forward(self, x):
        feature = []
        for convnext, convnext2, attn, downsample, compress, linear in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = attn(x)
            x = downsample(x)
            h = compress(x)
            feature.append(linear(torch.flatten(h, start_dim=1)))
            
        feats = []
        for feat in feature:
            feats.append(self.sigmoid(feat))
        #print(feats.shape)
        return feats[0], feats[1], feats[2], feats[3]
        

if __name__ == "__main__":
    # d = discriminator_v2("cuda:0" , 3,64)
    d = Generator(nz=10, nc=3, ngf=10).to("cuda:1")
    import torch

    input = torch.randn((1, 10, 1, 1)).to("cuda:1")
    print(input)
    output = d(input)
    print(output)
    # print(output.shape)
    # print(output)     # 128:[1, 84] 256:[1,336]
