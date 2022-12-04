import os

import numpy as np
import torch
from models.unet import EncoderUNetModel
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import create_named_schedule_sampler
from utils.script_util import create_gaussian_diffusion
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from custom_datasets.MNIST import MNISTDataset

# UNET Config
image_size = 128  # the target image resolution
in_channels = 1  # the number of input channels, 3 for RGB images
model_channels = 32  # Classifier width
out_channels = 10  # the number of output channels  # 10 for MNIST
num_res_blocks = 1  # Classifier depth
attention_resolutions = tuple([])
channel_mult = (1, 2, 4, 4)
num_head_channels = 64  # What is this? It is like this on guided-diffusion repo
use_scale_shift_norm = True
resblock_updown = True
pool = "attention"



# Sampler config
sampler_name = "uniform"

# Data
# dataset = MNIST(root="mnist_data", download=True)
dataset = MNISTDataset(root="mnist_data", train=True, download=True)
shuffle = True
num_workers = 1
drop_last = True
batch_size = 100
validation_split = 0.2
random_seed = 42

# Mixed precision training
initial_lg_loss_scale = 16.0

# Training
learning_rate = 1e-4
weight_decay = 0.0
iterations = 3000
# eval_interval = 1
output_path = "refactored_model_output"


model = EncoderUNetModel(
    image_size=image_size,
    in_channels=in_channels,
    model_channels=model_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    attention_resolutions=attention_resolutions,
    channel_mult=channel_mult,
    num_head_channels=num_head_channels,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    pool=pool,
)


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

mp_trainer = MixedPrecisionTrainer(
    model=model, initial_lg_loss_scale=initial_lg_loss_scale
)

# I am not adding The DistributedDataParallel wrapper here because I am not using multiple GPUs
# model = DistributedDataParallel()

optimizer = torch.optim.AdamW(
    mp_trainer.master_params, lr=learning_rate, weight_decay=weight_decay
)

mp_trainer.optimize(optimizer)

# I have deleted the log and the anneal learning rate functions


for step in tqdm(range(iterations)):

    # This next part I am not quite sure why we need it
    # if validation_loader is not None and not step % eval_interval:
    with torch.no_grad():
        model.eval()
        model.train()

torch.save(
    mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
    os.path.join(output_path, f"model_{iterations:06d}.pt"),
)
torch.save(
    optimizer.state_dict(), os.path.join(output_path, f"opt_{iterations:06d}.pt")
)
