import copy
import torch.nn.functional as F
import numpy as np
import glob
import shutil
import cv2
import os
import errno
import torch
import pyiqa
import shutil
import math
import lpips

from torch import nn
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

from util.fid_score import calculate_fid_given_paths
from util.sagan_models import Generator, Discriminator

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

####### helpers functions

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

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


class DTLS(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        size_list,
        stride,
        timesteps,
        device,
        stochastic=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.size_list = size_list
        self.stride = stride
        self.device = device
        self.MSE_loss = nn.MSELoss()

    def transform_func(self, img, target_size):
        dice = torch.rand(1)
        dice2 = torch.rand(1)

        n = target_size
        m = self.image_size

        if dice < 0.25:
            img_1 = F.interpolate(img, size=n, mode='bicubic', antialias=True)
        elif 0.25 <= dice < 0.5:
            img_1 = F.interpolate(img, size=n, mode='bilinear', antialias=True)
        elif 0.5 <= dice < 0.75:
            img_1 = F.interpolate(img, size=n, mode='area')
        else:
            img_1 = F.interpolate(img, size=n, mode='nearest-exact')

        if dice2 < 0.25:
            img_1 = F.interpolate(img_1, size=m, mode='bicubic', antialias=True)
        elif 0.25 <= dice2 < 0.5:
            img_1 = F.interpolate(img_1, size=m, mode='bilinear', antialias=True)
        elif 0.5 <= dice2 < 0.75:
            img_1 = F.interpolate(img_1, size=m, mode='area')
        else:
            img_1 = F.interpolate(img_1, size=m, mode='nearest-exact')

        # sigma = (1 - (n**2/m**2)) * 0.05
        # noise = torch.randn_like(img_1)
        # img_1 += noise * sigma

        return  img_1

    def transform_func_sample(self, img, target_size):
        n = target_size
        m = self.image_size

        # img_1 = F.interpolate(img, size=n, mode='nearest-exact')
        # img_1 = F.interpolate(img_1, size=m, mode='nearest-exact')

        img_1 = F.interpolate(img, size=n, mode='bicubic', antialias=True)
        img_1 = F.interpolate(img_1, size=m, mode='bicubic', antialias=True)

        # sigma = (1 - (n**2/m**2)) * 0.065
        # noise = torch.randn_like(img_1)
        # img_1 += noise * sigma

        return  img_1

    def transform_mapper(self, img, target_size):
        img = F.interpolate(img, size=target_size, mode='bicubic', antialias=True)
        return  img

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None, imgname=None):
        if t == None:
            t = self.num_timesteps

        blur_img = self.transform_func_sample(img.clone(), self.size_list[t])
        img_t = blur_img.clone()
        previous_x_s0 = None
        momentum = 0

        ####### Domain Transfer
        while (t):
            current_step = self.size_list[t]
            next_step = self.size_list[t-1]
            print(f"Current Step of img: from {current_step} to {next_step}")

            step = torch.full((batch_size,), t, dtype=torch.long).to(self.device)

            if previous_x_s0 is None:
                momentum_l = 0
            else:
                momentum_l = self.transform_func_sample(momentum, current_step)

            # weight = (1 - (current_step**2/self.image_size**2))
            weight = (1 - math.log(current_step + 1 - self.size_list[-1]) / math.log(self.image_size))
            # print(weight)

            if previous_x_s0 is None:
                R_x = self.denoise_fn(img_t, step)
                previous_x_s0 = R_x
            else:
                R_x = self.denoise_fn(img_t + momentum_l * 0, step)

            momentum += previous_x_s0 - R_x
            previous_x_s0 = R_x

            # R_x = self.denoise_fn(img_t, step)

            x4 = self.transform_func_sample(R_x, next_step)
            # utils.save_image((x4+1)/2, f"fake_lr_result_iii/{imgname}_{next_step}_SR.png")

            img_t = x4
            t -= 1
        return blur_img, img_t
    
    def gen_sr(self, fake_lr):
        fake_lr = self.transform_mapper(fake_lr, self.size_list[-1])
        t = torch.ones(fake_lr.shape[0], device=self.device).long() * self.num_timesteps
        return  self.denoise_fn(fake_lr, t)
        
    def p_losses(self, x_start, fake_x_start, t):
        ###################################### v1
        x_blur = x_start.clone()

        for i in range(t.shape[0]):
            current_step = self.size_list[t[i]]
            if current_step == self.size_list[-1]:
                x_blur[i] = self.transform_func(fake_x_start[i].unsqueeze(0), current_step)
            else:
                x_blur[i] = self.transform_func(x_blur[i].unsqueeze(0), current_step)
        x_recon = self.denoise_fn(x_blur, t)

        ### Loss function
        loss = self.MSE_loss(x_recon, x_start)
        return loss, x_recon
        ###################################### v2
        # x_blur = x_start.clone()

        # for i in range(t.shape[0]):
        #     current_step = self.size_list[t[i]]
        #     x_blur[i] = self.transform_func(x_blur[i].unsqueeze(0), current_step)
        # x_recon = self.denoise_fn(x_blur, t)
        
        # x_fake = self.transform_func(fake_x_start, self.size_list[-1])
        # last_step = torch.full((t.shape[0],), self.num_timesteps, dtype=torch.long).to(self.device)
        # x_recon_fake = self.denoise_fn(x_fake, last_step)

        # loss = self.MSE_loss(x_recon, x_start) + self.MSE_loss(x_recon_fake, x_start)

        # return loss, x_recon, x_recon_fake

    def forward(self, x, fake_x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(1, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, fake_x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, fake_folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.fake_folder = fake_folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.paths_fake = [p for ext in exts for p in Path(f'{fake_folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            # transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        fake_path = os.path.basename(path)
        path = f"{self.fake_folder}/{fake_path}"
        img_fake = Image.open(path)

        return self.transform(img), self.transform(img_fake)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        discriminator,
        folder,
        fake_folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        gan_image_size = 16,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()

        self.model = diffusion_model
        self.discriminator = discriminator

        self.mapper = Generator().to(device)

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.save_and_sample_every = save_and_sample_every

        self.image_size = diffusion_model.image_size
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.nrow = 4
        self.metrics_list = []

        self.folder_path = folder
        self.ds = Dataset(folder, fake_folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=2))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
        self.opt_d = Adam(self.discriminator.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)

        self.BCE_loss = torch.nn.BCELoss()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        self.best_quality = 0
        self.load_path = load_path


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_last(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'dis': self.discriminator.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'DTLS.pt'))

    
    def load_all(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        self.discriminator.load_state_dict(data['dis'], strict=False)

    def load_for_eval(self, load_path):
        print("Loading : ", "weight/gan.pt")
        data = torch.load(load_path, map_location=self.device)
        gan_data = torch.load("weight/gan.pt", map_location=self.device)

        self.ema_model.load_state_dict(data['ema'], strict=False)
        self.mapper.load_state_dict(gan_data['mapper'], strict=False)

    def train(self):
        if self.load_path != None:
            self.load_all(self.load_path)
        acc_loss = 0
        self.discriminator.train()

        while self.step < self.train_num_steps:
            
            data, fake_data = next(self.dl) #.to(self.device)
            data = data.to(self.device)
            fake_data = fake_data.to(self.device)


            self.opt_d.zero_grad()
            score_true = self.discriminator(data)
            GAN_true = torch.ones_like(score_true)
            loss_dis_true = self.BCE_loss(score_true, GAN_true)
            # loss_dis_true = torch.nn.ReLU()(1.0 - score_true).mean()
            loss_dis_true.backward()

            loss_mse, x_recon, x_recon_fake = self.model(data, fake_data)
            # loss_mse = loss_mse * 0.5

            score_false = self.discriminator(x_recon.detach())
            score_false_fake = self.discriminator(x_recon_fake.detach())

            GAN_false = torch.zeros_like(score_false)
            loss_dis_false = (self.BCE_loss(score_false, GAN_false) + self.BCE_loss(score_false_fake, GAN_false))
            # loss_dis_false = (torch.nn.ReLU()(1.0 + score_false).mean() + torch.nn.ReLU()(1.0 + score_false_fake).mean())
            loss_dis_false.backward()
            self.opt_d.step()

            self.opt.zero_grad()

            score_fake = self.discriminator(x_recon)
            score_fake2 = self.discriminator(x_recon_fake)

            GAN_fake = torch.ones_like(score_fake)
            loss_gen = (self.BCE_loss(score_fake, GAN_fake) + self.BCE_loss(score_fake2, GAN_fake)) * 1e-3

            # loss_gen = (score_fake + score_fake2).mean() * 1e-3

            # lpips_loss = self.loss_fn_alex(x_recon, data) + self.loss_fn_alex(x_recon_fake, data)
            # lpips_loss = lpips_loss.mean() * 2e-6
            
            (loss_mse + loss_gen).backward()
            self.opt.step()

            
            if self.step % 10 == 0:
                print(f'{self.step} DTLS: Total loss: {loss_mse.item() + loss_gen.item()} | MSE: {loss_mse.item()} | Generate: {loss_gen.item()}'
                      f'| Dis real: {loss_dis_true.item()} | Dis false: {loss_dis_false.item()}')
                print(' ')

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0 or self.step % self.save_and_sample_every == 0:

                
                _, sample_hr = self.ema_model.sample(batch_size=self.batch_size, img=fake_data)
                
                save_img = torch.cat((fake_data, sample_hr),dim=0)
                utils.save_image((save_img+1)/2, str(self.results_folder / f'{self.step}_gan_DTLS.png'), nrow=self.nrow)

                score_map = self.discriminator(sample_hr)
                utils.save_image(score_map, str(self.results_folder / f'{self.step}_gan_score.png'), nrow=self.nrow)

                # utils.save_image((fake_data+1)/2, str(self.results_folder / f'{self.step}_gan.png'), nrow=self.nrow)
                # utils.save_image((sample_hr+1)/2, str(self.results_folder / f'{self.step}_gan_DTLS.png'), nrow=self.nrow)

                _, sr_real = self.ema_model.sample(batch_size=self.batch_size, img=data)

                save_img = torch.cat((data, sr_real),dim=0)
                utils.save_image((save_img+1)/2, str(self.results_folder / f'{self.step}_DTLS.png'), nrow=self.nrow)
                self.save_last()

            self.step += 1

        print('training completed')

    def evaluation(self):
        if self.load_path != None:
            self.load_for_eval(self.load_path)

        #blur_img_set = torch.tensor([])
        #hq_img_set = torch.tensor([])
        self.mapper.eval()
        for i in range(10):
            # data, imgname = next(self.dl) #.to(self.device)
            # data = data.to(self.device)
            # utils.save_image((data + 1) /2, str(self.results_folder /  f'data_{i}.png'), nrow=1)

            # sample_lr, sample_hr = self.ema_model.sample(batch_size=1, img=data, imgname=imgname)
            # utils.save_image((sample_hr + 1) /2, str(self.results_folder /  f'data_hr_{i}.png'), nrow=1)
            # utils.save_image((sample_lr + 1) /2, str(self.results_folder /  f'data_lr_{i}.png'), nrow=1)


            noise = torch.randn(1, 100, 1, 1, device=self.device)
            sample_lr = self.mapper(noise)
            noise = noise.view(1, 1, 10, 10)
            noise = F.interpolate(noise, size=128, mode='nearest-exact')

            utils.save_image((noise + 1) /2, str(self.results_folder /  f'noise_{i}.png'), nrow=1)
            
            sample_lr, sample_hr = self.ema_model.sample(batch_size=1, img=sample_lr,imgname=f"gen_{i}")

            utils.save_image((sample_lr + 1) /2, str(self.results_folder /  f'gen_lr_{i}.png'), nrow=1)
            utils.save_image((sample_hr + 1) /2, str(self.results_folder /  f'gen_hr_{i}.png'), nrow=1)


            # blur_img_set = torch.cat((blur_img_set, sample_lr.to("cpu")), dim=0)
            # hq_img_set = torch.cat((hq_img_set, sample_hr.to("cpu")), dim=0)

            # utils.save_image((blur_img_set + 1) /2, str(self.results_folder /  f'sample_lr.png'), nrow=4)
            # utils.save_image((hq_img_set + 1) /2, str(self.results_folder /  f'sample_hr.png'), nrow=4)
