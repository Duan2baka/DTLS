import copy
import torch.nn.functional as F
import numpy as np
import glob
import shutil
import cv2
import os
import errno
import torch
import lpips
import pyiqa
import math

from torch import nn
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from util.discriminator import discriminator_v3 as d_3

from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr

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
            down_sample_method = 'bicubic'
        elif 0.25 <= dice < 0.5:
            down_sample_method = 'bilinear'
        elif 0.5 <= dice < 0.75:
            down_sample_method = 'area'
        else:
            down_sample_method = 'nearest-exact'

        if dice2 < 0.25:
            up_sample_method = 'bicubic'
        elif 0.25 <= dice < 0.5:
            up_sample_method = 'bilinear'
        elif 0.5 <= dice < 0.75:
            up_sample_method = 'area'
        else:
            up_sample_method = 'nearest-exact'

        if self.image_size // target_size > 16:
            if down_sample_method == "bicubic" or down_sample_method == "bilinear":
                img_1 = F.interpolate(img, size=m // 2, mode=down_sample_method, antialias=True)
                img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode=down_sample_method, antialias=True)
                img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode=down_sample_method, antialias=True)
                img_1 = F.interpolate(img_1, size=n, mode=down_sample_method, antialias=True)
            else:
                img_1 = F.interpolate(img, size=m // 2, mode=down_sample_method)
                img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode=down_sample_method)
                img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode=down_sample_method)
                img_1 = F.interpolate(img_1, size=n, mode=down_sample_method)
        else:
            if down_sample_method == "bicubic" or down_sample_method == "bilinear":
                img_1 = F.interpolate(img, size=n, mode=down_sample_method, antialias=True)
            else:
                img_1 = F.interpolate(img, size=n, mode=down_sample_method)

        if up_sample_method == "bicubic" or down_sample_method == "bilinear":
            img_1 = F.interpolate(img_1, size=m, mode=up_sample_method, antialias=True)
        else:
            img_1 = F.interpolate(img_1, size=m, mode=up_sample_method)


        return  img_1

    def transform_func_sample(self, img, target_size):
        n = target_size
        m = self.image_size
        
        if self.image_size // target_size > 16:
            img_1 = F.interpolate(img, size=m // 2, mode="bicubic", antialias=True)
            img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode="bicubic", antialias=True)
            img_1 = F.interpolate(img_1, size=img_1.shape[2] // 2, mode="bicubic", antialias=True)
            img_1 = F.interpolate(img_1, size=n, mode="bicubic", antialias=True)
        else:
            img_1 = F.interpolate(img, size=n, mode="bicubic", antialias=True)
            
        img_1 = F.interpolate(img_1, size=m, mode="bicubic", antialias=True)

        return  img_1
        
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

            #weight = (1 - (current_step**2/self.image_size**2))
            #weight = (1 - math.log(current_step + 1 - self.size_list[-1])/math.log(self.image_size))

            if previous_x_s0 is None:
                R_x = self.denoise_fn(img_t, step)
                # return blur_img, R_x
                previous_x_s0 = R_x
            else:
                R_x = self.denoise_fn(img_t + momentum_l, step)

            momentum += previous_x_s0 - R_x
            previous_x_s0 = R_x

            # R_x = self.denoise_fn(img_t, step)

            # utils.save_image((R_x+1)/2, f"20230103_eval/{current_step}_SR.png")
            x4 = self.transform_func_sample(R_x, next_step)
            img_t = x4
            t -= 1
        return blur_img, img_t

    def p_losses(self, x_start, t):
        x_blur = x_start.clone()
        for i in range(t.shape[0]):
            current_step = self.size_list[t[i]]
            x_blur[i] = self.transform_func(x_blur[i].unsqueeze(0), current_step)
        x_recon = self.denoise_fn(x_blur, t)

        ### Loss function
        loss = self.MSE_loss(x_recon, x_start)
        return loss, x_recon

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()

        return self.p_losses(x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        input_image,
        results_folder,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)

        self.image_size = diffusion_model.image_size

        self.discriminator = d_3(image_size=self.image_size, dim=24,
                                 dim_mults=(8, 4, 4, 2, 2, 1, 1),channels=3).to(device)
        # self.lpips = lpips.LPIPS(net='vgg').to(device)

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.nrow = 4
        self.metrics_list = []
        self.input_image = input_image

        self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=1))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
        self.opt_d = Adam(self.discriminator.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
        self.BCE_loss = torch.nn.BCELoss(size_average=True,reduction='none')

        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        self.niqe = pyiqa.create_metric('niqe', device=torch.device(self.device))
        self.MANIQA = pyiqa.create_metric('maniqa', device=torch.device(self.device))

        self.best_quality = 0
        #if load_path != None:
        #    self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_best(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'dis': self.discriminator.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model_best.pt'))

    def save_last(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'dis': self.discriminator.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model_last.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        # self.discriminator.load_state_dict(data['dis'], strict=False)

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        self.step = 0
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).to(self.device)
                print(data.shape)
                exit(0)
                score_true = self.discriminator(data)
                GAN_true = torch.ones_like(score_true)
                loss_dis_true = self.BCE_loss(score_true, GAN_true)
                backwards(loss_dis_true / self.gradient_accumulate_every, self.opt_d)


                loss, x_recon = self.model(data)

                score_false = self.discriminator(x_recon.detach())
                GAN_false = torch.zeros_like(score_false)
                loss_dis_false = self.BCE_loss(score_false, GAN_false)
                backwards(loss_dis_false / self.gradient_accumulate_every, self.opt_d)


                score_fake = self.discriminator(x_recon)
                GAN_fake = torch.ones_like(score_fake)
                loss_gen = self.BCE_loss(score_fake, GAN_fake) * 2e-3

                backwards((loss + loss_gen) / self.gradient_accumulate_every, self.opt)

                print(f'{self.step}: MSE: {loss.item()} | Generate: {loss_gen.item()} '
                    f'| Dis real: {loss_dis_true.item()} | Dis false: {loss_dis_false.item()}')
            
            self.opt_d.step()
            self.opt_d.zero_grad()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0 or self.step % self.save_and_sample_every == 0:
                data = next(self.dl).to(self.device)
                blur_img_set = torch.tensor([])
                hq_img_set = torch.tensor([])
                FFHQ_quality_MANIQA = 0
                FFHQ_quality_NIQE = 0
                for image in data:
                    blur_img, hq_img = self.ema_model.sample(batch_size=1, img=image.unsqueeze(0))
                    hq_img = (hq_img + 1) / 2
                    blur_img = (blur_img + 1) /2
                    utils.save_image(hq_img, str(self.results_folder / f'temp.png'), nrow=self.nrow)

                    blur_img_set = torch.cat((blur_img_set, blur_img.to("cpu")), dim=0)
                    hq_img_set = torch.cat((hq_img_set, hq_img.to("cpu")), dim=0)

                    NIQE_mark = self.niqe(str(self.results_folder / f'temp.png')).item()
                    MANIQA_mark = self.MANIQA(str(self.results_folder / f'temp.png')).item()

                    FFHQ_quality_MANIQA += MANIQA_mark
                    FFHQ_quality_NIQE += NIQE_mark

                    os.remove(str(self.results_folder / f'temp.png'))

                FFHQ_quality_MANIQA /= self.batch_size
                FFHQ_quality_NIQE /= self.batch_size

                img_set = torch.cat((blur_img_set, hq_img_set), dim=0)
                utils.save_image(img_set, str(self.results_folder / f'{self.step}_FFHQ.png'), nrow=blur_img_set.shape[0])

                
                self.metrics_list.append(f"FFHQ Images MANIQA: {FFHQ_quality_MANIQA} | NIQE: {FFHQ_quality_NIQE}")
                
                file = open(f"{self.results_folder}/quality.txt", 'w')
                for line in self.metrics_list:
                    file.write(line + "\n")
                file.close()

                print(f'Mean of last {self.step}: {acc_loss}')
                self.save_last()
                acc_loss = 0

            self.step += 1
        print('training completed')

    def evaluation(self):
        total_quality_MANIQA = 0
        total_quality_NIQE = 0
        total_img = 0
        blur_img_set = torch.tensor([])
        hq_img_set = torch.tensor([])

        # data = next(self.dl).to(self.device)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_hr.png")
        # data = F.interpolate(data, 32, mode="bilinear", antialias=True)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_lr.png")


        for idx, path in enumerate(sorted(glob.glob(os.path.join(self.input_image, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print(idx, imgname)
            # read image
            print(path)
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
            img = img.unsqueeze(0).to(self.device)
            img = img * 2 - 1

            blur_img, img_sr = self.ema_model.sample(batch_size=1, img=img)
            img_sr = (img_sr + 1) / 2
            blur_img = (blur_img + 1) / 2
            utils.save_image(img_sr, str(self.results_folder /  f'{imgname}.png'), nrow=1)
            utils.save_image(img_sr, str(self.results_folder /  f'temp.png'), nrow=self.nrow)

            NIQE_mark = self.niqe(str(self.results_folder / f'temp.png')).item()
            MANIQA_mark = self.MANIQA(str(self.results_folder / f'temp.png')).item()

            os.remove(str(self.results_folder / f'temp.png'))

            blur_img_set = torch.cat((blur_img_set, blur_img.to("cpu")), dim=0)
            hq_img_set = torch.cat((hq_img_set, img_sr.to("cpu")), dim=0)

            total_quality_MANIQA += MANIQA_mark
            total_quality_NIQE += NIQE_mark
            total_img +=1



        img_set = torch.cat((blur_img_set, hq_img_set), dim=0)
        utils.save_image(img_set, str(self.results_folder / f'{self.step}_overall.png'), nrow=blur_img_set.shape[0])
        utils.save_image(blur_img_set, str(self.results_folder / f'lq_overall.png'), nrow=6)
        utils.save_image(hq_img_set, str(self.results_folder / f'hq_overall.png'), nrow=6)

        print(f"Avg MANIQA: {total_quality_MANIQA / total_img}, NIQE: {total_quality_NIQE / total_img}")
        return total_quality_MANIQA / total_img, total_quality_NIQE / total_img
