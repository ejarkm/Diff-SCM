import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from utils.script_util import create_gaussian_diffusion
from mpi4py import MPI
import blobfile
import torch
import io
from models.unet import UNetModel
import os
from models.unet import EncoderUNetModel
from sampling.sampling_utils import get_models_functions, estimate_counterfactual
from torch.utils.data import DataLoader
from custom_datasets.MNIST import MNISTDataset


# Data
dataset = MNISTDataset(root="mnist_data", train=True, download=True)
shuffle = True
num_workers = 1
drop_last = True
batch_size = 100
validation_split = 0.2
random_seed = 42


# Scorer model path
scorer_output_path = "/tmp/openai-2022-12-04-15-14-29-426217"
scorer_model_name = "model000000.pt"
scorer_model_path = os.path.join(scorer_output_path, scorer_model_name)

# Classifier Model path
classifier_output_path = "refactored_model_output"
classifier_model_name = "model_003000.pt"
classifier_model_path = os.path.join(classifier_output_path, classifier_model_name)

# Sampler config
classifier_scale = 1.0


# UNET Config (scorer model)
image_size = 128  # the target image resolution # Not used in UNetModel
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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

scorer_model = UNetModel(
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
scorer_model.to(device)
scorer_model.eval()
with blobfile.BlobFile(scorer_model_path, "rb") as f:
    data = f.read()
data = MPI.COMM_WORLD.bcast(data)
scorer_model_state_dict = torch.load(io.BytesIO(data), map_location=device)
scorer_model.load_state_dict(scorer_model_state_dict)


# Reading the test dataset. I am just testing the code,
# so I have to read a proper test dataset.
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
_, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


def get_generator_from_loader(loader):
    while True:
        yield from loader


test_generator_loader = get_generator_from_loader(test_loader)


# Reading the model
diffusion = create_gaussian_diffusion()


# Classifier EncoderUNET Config
# image_size = 128  # the target image resolution
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



classifier_model = EncoderUNetModel(
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

classifier_model.to(device)
classifier_model.eval()
with blobfile.BlobFile(classifier_model_path, "rb") as f:
    data = f.read()
data = MPI.COMM_WORLD.bcast(data)
classifier_model_state_dict = torch.load(io.BytesIO(data), map_location=device)
classifier_model.load_state_dict(classifier_model_state_dict)


# Model functions config
## Sampling
sampling_label_of_intervention = "y"
sampling_classifier_scale = 1.0
sampling_batch_size = 100
sampling_norm_cond_scale = 3.0  # This one was not in MNIST config and I think it is important based on the README
sampling_target_class = (
    5  # I think that this is the one that we will use to generate the counterfactual
)
sampling_dynamic_sampling = True
sampling_progress = False
sampling_eta = 0.0
sampling_clip_denoised = True
sampling_progression_ratio = 0.75
sampling_reconstruction = True


## Scorer
scorer_num_classes = 10
scorer_num_input_channels = 1
scorer_classifier_free_cond = False


cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(
    model=scorer_model,
    anti_causal_predictor=classifier_model,
    device=device,
    sampling_label_of_intervention=sampling_label_of_intervention,
    sampling_classifier_scale=sampling_classifier_scale,
    sampling_batch_size=sampling_batch_size,
    sampling_norm_cond_scale=sampling_norm_cond_scale,
    scorer_num_classes=scorer_num_classes,
)

# data_dict = next(test_generator_loader)

# results_per_sample = {
#     "original": ((data_dict["image"] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
# }
# # send data points to GPU
# model_kwargs = {k: v.to(device) for k, v in data_dict.items()}
# init_image = data_dict["image"].to(device)

# # Idk if this is correct
# sampling_conterfactual_class = 1
# num_input_channels = 1

# model_kwargs["y"] = (
#     (sampling_conterfactual_class * torch.ones((sampling_batch_size,)))
#     .to(torch.long)
#     .to(device)
# )


# Ver el sample conterfactual y usar el estimate_counterfactual
# counterfactual = diffusion.diffscm_counterfactual_sample(
#     model_fn(sampling_batch_size, num_input_channels, image_size, image_size),
#     factual_image=init_image,
#     anticausal_classifier_fn=cond_fn,
#     model_kwargs=model_kwargs,
#     device=device,
# )


all_results = []
for i, data_dict in enumerate(test_generator_loader):

    counterfactual_image, sampling_progression = estimate_counterfactual(
        sampling_batch_size=sampling_batch_size,
        sampling_target_class=sampling_target_class,
        scorer_num_input_channels=scorer_num_input_channels,
        sampling_dynamic_sampling=sampling_dynamic_sampling,
        sampling_progress=sampling_progress,
        sampling_eta=sampling_eta,
        sampling_clip_denoised=sampling_clip_denoised,
        sampling_progression_ratio=sampling_progression_ratio,
        sampling_classifier_scale=sampling_classifier_scale,
        scorer_image_size=image_size,
        scorer_classifier_free_cond=scorer_classifier_free_cond,
        device=device,
        diffusion=diffusion,
        cond_fn=cond_fn,
        model_fn=model_fn,
        model_classifier_free_fn=model_classifier_free_fn,
        denoised_fn=denoised_fn,
        data_dict=data_dict,
        reconstruction=sampling_reconstruction,
    )
    results_per_sample = {
        "original": data_dict,
        "counterfactual_sample": counterfactual_image.cpu().numpy(),
    }

    # diffusion, cond_fn, model_fn,
    # model_classifier_free_fn, denoised_fn,
    # data_dict)

    pass
