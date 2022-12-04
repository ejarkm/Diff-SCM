import torch
from typing import Dict
import torch.nn.functional as F


def get_models_functions(
    model,
    anti_causal_predictor,
    device,
    sampling_label_of_intervention,
    sampling_classifier_scale,
    sampling_batch_size,
    sampling_norm_cond_scale,
    scorer_num_classes,
):
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = anti_causal_predictor(x_in, t)
            if isinstance(out, Dict):
                logits = out[sampling_label_of_intervention]
            else:
                logits = out

            ## deal with
            y_new = (
                torch.cat(2 * [y[: y.size()[0] // 2]])
                if y.max() >= logits.size()[-1]
                else y
            )

            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_new.view(-1)]
            grad_log_conditional = torch.autograd.grad(selected.sum(), x_in)[0]

            return (
                grad_log_conditional * sampling_classifier_scale
            )  # * scaling[:, None, None, None]

    def model_fn(x, t, y=None, conditioning_x=None, **kwargs):
        y = (
            (scorer_num_classes * torch.ones((sampling_batch_size,)))
            .to(torch.long)
            .to(device)
        )
        return model(x, t, y=y, conditioning_x=conditioning_x)

    # Create an classifier-free guidance sampling function from Glide code
    def model_classifier_free_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + sampling_norm_cond_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps

    # classifier-free guidance without increasing batch - trading off space for time
    def model_classifier_free_opt_fn(x_t, ts, **kwargs):
        ## conditional diffusion output
        cond_eps = model(x_t, ts, **kwargs)
        ## unconditional diffusion output
        uncond_kwargs = kwargs.copy()
        uncond_kwargs["y"] = scorer_num_classes * torch.ones_like(kwargs["y"])
        uncond_eps = model(x_t, ts, **uncond_kwargs)
        eps = uncond_eps + sampling_norm_cond_scale * (cond_eps - uncond_eps)
        return eps

    def inpainting_denoised_fn(x_start, **kwargs):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return x_start * kwargs["inpaint_mask"] + kwargs["image"] * (
            1 - kwargs["inpaint_mask"]
        )

    # dynamic normalisation
    def clamp_to_spatial_quantile(x: torch.Tensor, **kwargs):
        p = 0.99
        b, c, *spatial = x.shape
        quantile = torch.quantile(torch.abs(x).view(b, c, -1), p, dim=-1, keepdim=True)
        quantile = torch.max(quantile, torch.ones_like(quantile))
        quantile_broadcasted, _ = torch.broadcast_tensors(quantile.unsqueeze(-1), x)
        return (
            torch.min(torch.max(x, -quantile_broadcasted), quantile_broadcasted)
            / quantile_broadcasted
        )

    return cond_fn, model_fn, model_classifier_free_opt_fn, clamp_to_spatial_quantile


def estimate_counterfactual(
    sampling_batch_size,
    sampling_target_class,
    scorer_num_input_channels,
    sampling_dynamic_sampling,
    sampling_progress,
    sampling_eta,
    sampling_clip_denoised,
    sampling_progression_ratio,
    sampling_classifier_scale,
    scorer_image_size,
    scorer_classifier_free_cond,
    device,
    diffusion,
    cond_fn,
    model_fn,
    model_classifier_free_fn,
    denoised_fn,
    data_dict,
    reconstruction,
):
    model_kwargs, init_image = get_input_data(
        device, sampling_batch_size, sampling_target_class, data_dict
    )
    # DDIM loop in reverse time order for inferring exogenous noise (image latent space)
    exogenous_noise, abduction_progression = diffusion.ddim_sample_loop(
        model_fn,
        (
            sampling_batch_size,
            scorer_num_input_channels,
            scorer_image_size,
            scorer_image_size,
        ),
        clip_denoised=sampling_clip_denoised,
        model_kwargs=model_kwargs,
        denoised_fn=denoised_fn if sampling_dynamic_sampling else None,
        noise=init_image,
        cond_fn=None,
        device=device,
        progress=sampling_progress,
        eta=sampling_eta,
        reconstruction=reconstruction,
        sampling_progression_ratio=sampling_progression_ratio,
    )
    init_image = exogenous_noise
    # DDIM diffusion inference  with conditioning (intervention), starting from a latent image instead of random noise
    
    # TODO: For some reason, the reconstruction was hardcoded and it was not running!!! (on the fucntion before too)
    
    counterfactual_image, diffusion_progression = diffusion.ddim_sample_loop(
        model_classifier_free_fn if scorer_classifier_free_cond else model_fn,
        (
            sampling_batch_size,
            scorer_num_input_channels,
            scorer_image_size,
            scorer_image_size,
        ),
        clip_denoised=sampling_clip_denoised,
        model_kwargs=model_kwargs,
        denoised_fn=denoised_fn if sampling_dynamic_sampling else None,
        noise=init_image,
        cond_fn=cond_fn if sampling_classifier_scale != 0 else None,
        device=device,
        progress=sampling_progress,
        eta=sampling_eta,
        reconstruction=reconstruction,
        sampling_progression_ratio=sampling_progression_ratio,
    )
    sampling_progression = abduction_progression + diffusion_progression
    return counterfactual_image, sampling_progression


def get_input_data(device, sampling_batch_size, sampling_target_class, data_dict):

    model_kwargs = {k: v.to(device) for k, v in data_dict.items()}
    model_kwargs["y"] = (
        (sampling_batch_size * torch.ones((sampling_target_class,)))
        .to(torch.long)
        .to(device)
    )

    init_image = data_dict["image"].to(device)

    return model_kwargs, init_image
