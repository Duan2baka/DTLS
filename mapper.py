import torch
import argparse
from torch import nn
from PIL import Image
from torchvision import transforms, utils
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
import copy
from util.mapper_model import Generator, Generator_v2, Discriminator

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--mode', default='eval', type=str, help="mode for either 'train' or 'eval'")
parser.add_argument('--lr_size', default=32, type=int, help="size of LR image")
parser.add_argument('--train_steps', default=100001, type=int)
parser.add_argument('--lr_rate', default=2e-4, help="learning rate")
parser.add_argument('--sample_every_iterations', default=1000, type=int, help="sample SR images for every number of iterations")
parser.add_argument('--save_folder', default="ablation_gen_Conv2d", type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', default=None, type=str, help="None or directory to pretrained model")
parser.add_argument('--data_path', default='/hdda/Datasets/Face_super_resolution/images1024x1024/', type=str, help="directory to your training dataset")
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_samples', default=30, type=int, help="number of samples to generate")
args = parser.parse_args()

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer(object):
    def __init__(
            self):
        super().__init__()
        self.device = args.device
        self.ema = EMA(0.995)

        self.generator = Generator(nz=1024).to(self.device)
        #model_state_dict = torch.load('./pre_trained_weight/DTLS_128.pt', map_location=self.device)
        model_state_dict = torch.load('./DTLS_mapper/model_last.pt', map_location=self.device)
        print(model_state_dict.keys())
        self.generator.load_state_dict(model_state_dict['model'])
        self.discriminator = Discriminator().to(self.device)

        #self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.ema_model = copy.deepcopy(self.generator)
        self.ema_model_d = copy.deepcopy(self.discriminator)

        self.batch_size = args.batch_size
        self.image_size = args.lr_size
        self.train_num_steps = args.train_steps

        self.ds = Dataset(args.data_path, self.image_size)

        #self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True, num_workers=1))
        self.opt = Adam(self.generator.parameters(), lr=args.lr_rate, betas=(0.9, 0.999), eps=1e-8)
        self.opt_d = Adam(self.discriminator.parameters(), lr=args.lr_rate, betas=(0.9, 0.999), eps=1e-8)
        self.step = 0

        self.step_start_ema = 2000
        self.update_ema_every = 10
        self.reset_parameters()

        self.results_folder = Path(args.save_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.BCE_loss = torch.nn.BCELoss()

        self.fp16 = False

        if args.load_path is not None:
            self.load(args.load_path)

    def save_last(self):
        data = {
            'step': self.step,
            'model': self.generator.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model_last.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.generator.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.generator.state_dict())
        self.ema_model_d.load_state_dict(self.discriminator.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.generator)
        self.ema.update_model_average(self.ema_model_d, self.discriminator)

    def train(self):
        self.generator.train()
        while self.step < self.train_num_steps:
            self.opt_d.zero_grad()

            data = next(self.dl).to(self.device)
            score_true = self.discriminator(data).view(-1)
            true_gt = torch.ones_like(score_true)
            d_loss_real = self.BCE_loss(score_true, true_gt)
            d_loss_real.backward()


            noise = torch.randn(self.batch_size, 1024, 1, 1, device=self.device)
            fake_lr = self.generator(noise)

            score_fake = self.discriminator(fake_lr.detach()).view(-1)
            false_gt = torch.zeros_like(score_fake)
            d_loss_fake = self.BCE_loss(score_fake, false_gt)
            d_loss_fake.backward()

            self.opt_d.step()

            self.opt.zero_grad()

            score_false = self.discriminator(fake_lr).view(-1)
            true_gt = torch.ones_like(score_false)

            g_loss = self.BCE_loss(score_false, true_gt)
            g_loss.backward()

            self.opt.step()

            print(f'{self.step}: Generate: {g_loss.item()} | Discriminator real: {d_loss_real.item()} '
                  f'| Discriminator fake: {d_loss_fake.item()}')

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % args.sample_every_iterations == 0:
                noise = torch.randn(self.batch_size, 1024, 1, 1, device=self.device)
                fake_lr = self.generator(noise)
                noise = noise.view(self.batch_size, 1, 32, 32)
                utils.save_image(data, f"{self.results_folder}/{self.step}_True_lr.png")
                utils.save_image((noise+1)/2, f"{self.results_folder}/{self.step}_noise.png")
                utils.save_image(fake_lr, f"{self.results_folder}/{self.step}_fake_lr.png")

                print("saving")
                self.save_last()

            self.step += 1

    def eval(self):
        self.generator.eval()
        for i in range(args.n_samples):
            noise = torch.randn(1, 1024, 1, 1, device=self.device)
            fake_lr = self.generator(noise)
            print(fake_lr.shape)
            utils.save_image(fake_lr, f"{self.results_folder}/sample_fake_lr_{i}.png")
            print("Generating", i)

if __name__ == "__main__":
    trainer = Trainer()

    if args.mode == "train":
        trainer.train()
    else:
        trainer.eval()
