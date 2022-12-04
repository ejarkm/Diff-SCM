import os

import numpy as np
import torch
from guided_diffusion.resample import create_named_schedule_sampler
from utils.script_util import create_gaussian_diffusion
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from models.unet import UNetModel
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion import dist_util
from train_utils.guided_diffusion_train_util import TrainLoop
from custom_datasets.MNIST import MNISTDataset


# UNET Config
image_size = 28  # the target image resolution # Not used in UNetModel
in_channels = 1  # the number of input channels, 3 for RGB images
model_channels = 32
out_channels = 1  # the number of output channels
num_res_blocks = 1
attention_resolutions = tuple([])
dropout = 0.1
channel_mult = (1, 2, 2)
num_classes = None
num_heads = 1
num_head_channels = -1
num_heads_upsample = -1
use_scale_shift_norm = True
resblock_updown = False



# Sampler config
sampler_name = "uniform"

# Data
dataset = MNISTDataset(root="mnist_data", train=True, download=True)
shuffle = True
num_workers = 1
drop_last = True
batch_size = 100
validation_split = 0.2
random_seed = 42


# Training
batch_size = 256
microbatch = -1
lr = 1e-4
ema_rate = "0.9999"
log_interval = 50
save_interval = 1000
resume_checkpoint = ""
use_fp16 = False
fp16_scale_growth = 1e-3
weight_decay = 0.01
lr_anneal_steps = 0
cond_dropout_rate = 0.0
conditioning_variable = "y"
iterations = 3e4



output_path = "refactored_model_output"

dist_util.setup_dist()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


model = UNetModel(
    in_channels=in_channels,
    model_channels=model_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    attention_resolutions=attention_resolutions,
    dropout=dropout,
    channel_mult=channel_mult,
    num_classes=num_classes,
    num_heads=num_heads,
    num_head_channels=num_head_channels,
    num_heads_upsample=num_heads_upsample,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
)
model.to(device)

diffusion = create_gaussian_diffusion()
schedule_sampler = create_named_schedule_sampler(name=sampler_name, diffusion=diffusion)


# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


def get_generator_from_loader(loader):
    while True:
        yield from loader

train_generator_loader = get_generator_from_loader(train_loader)
validation_generator_loader = get_generator_from_loader(validation_loader)



TrainLoop(
    model=model,
    diffusion=diffusion,
    data=train_generator_loader,
    data_val=validation_generator_loader,
    batch_size=batch_size,
    microbatch=microbatch,
    lr=lr,
    ema_rate=ema_rate,
    log_interval=log_interval,
    save_interval=save_interval,
    resume_checkpoint=resume_checkpoint,
    use_fp16=use_fp16,
    fp16_scale_growth=fp16_scale_growth,
    schedule_sampler=schedule_sampler,
    weight_decay=weight_decay,
    lr_anneal_steps=lr_anneal_steps,
    cond_dropout_rate=cond_dropout_rate,
    conditioning_variable=conditioning_variable,
    iterations=iterations,
).run_loop()

