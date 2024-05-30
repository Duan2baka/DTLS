import math
import torch.cuda
from util.unet_smiling import Unet
# from unet_old import Unet
# from dtls import DTLS, Trainer
from dtls_smiling import DTLS, Trainer
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default='myDTLS_smiling_1/model_last.pt', type=str, help="None or directory to pretrained model")
parser.add_argument('--data_path', default='./fake_dataset_128/', type=str, help="directory to your training dataset")

args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    residual=False
).to(device)

dtls = DTLS(
    model,
    image_size = args.size,
    device=device,
).to(device)


trainer = Trainer(
    dtls,
    args.data_path,
    image_size = args.size,
    train_batch_size = args.batch_size,
    train_lr = args.lr_rate,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    input_image = args.input_image,
    device = device,
    save_and_sample_every = args.sample_every_iterations
)


if args.mode == 'train':
    trainer.train()
elif args.mode == 'eval':
    trainer.evaluation()