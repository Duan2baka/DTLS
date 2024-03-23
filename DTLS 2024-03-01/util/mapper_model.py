from torch import nn

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

            # Old end
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),

            # New end
            # nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. ``(ngf) x 32 x 32``
            # nn.Conv2d(ngf, nc, 3, 1, 1),
            # nn.ReLU(True),
            # state size. ``(nc) x 64 x 64``
        )
        ## Refinement section
        # self.conv1 = nn.Conv2d(3, ngf, 3, 1, 1)
        # self.batch1 = nn.BatchNorm2d(ngf)
        #
        # self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 1, 1)
        # self.batch2 = nn.BatchNorm2d(ngf * 2)
        # self.conv3 = nn.Conv2d(ngf * 2, ngf, 3, 1, 1)
        # self.batch3 = nn.BatchNorm2d(ngf)
        #
        # self.conv4 = nn.Conv2d(ngf, ngf * 2, 3, 1, 1)
        # self.batch4 = nn.BatchNorm2d(ngf * 2)
        # self.conv5 = nn.Conv2d(ngf * 2, ngf, 3, 1, 1)
        # self.batch5 = nn.BatchNorm2d(ngf)
        #
        # self.conv6 = nn.Conv2d(ngf, nc, 3, 1, 1)
        #
        # self.ReLU = nn.ReLU()
        # self.Tanh = nn.Tanh()

    def forward(self, input):
        x = self.main(input)

        # x = self.ReLU(self.batch1(self.conv1(x)))
        #
        # temp = x
        # res = self.ReLU(self.batch2(self.conv2(x)))
        # x = temp + self.ReLU(self.batch3(self.conv3(res)))
        #
        # temp = x
        # res = self.ReLU(self.batch4(self.conv4(x)))
        # x = temp + self.ReLU(self.batch5(self.conv5(res)))

        return x

class Generator_v2(nn.Module):
    def __init__(self, nz=3, ngf=64, nc=3):
        super(Generator_v2, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d( nz, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``

            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``

            nn.Conv2d( ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``

            nn.Conv2d( ngf * 2, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 32 x 32``
        )

    def forward(self, input):
        return self.main(input)

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
            # nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=True),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)