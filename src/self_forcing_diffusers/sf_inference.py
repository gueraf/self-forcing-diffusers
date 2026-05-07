"""Shared Self-Forcing inference helpers used by both the autoregressive export script
and the upstream parity validator.

These reproduce the schedule and re-noising contract of the original Self-Forcing repo
(``guandeh17/Self-Forcing``) so diffusers and upstream stay bit-exact. Keep edits in
sync with the upstream causal-DMD inference path.
"""

from __future__ import annotations

import torch


def build_sf_denoising_steps(device) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, 1001, device=device, dtype=torch.float64)[:-1]
    sigmas = 5.0 * sigmas / (1.0 + 4.0 * sigmas)
    all_timesteps = torch.cat([sigmas * 1000.0, torch.tensor([0.0], device=device, dtype=torch.float64)])
    original_schedule = torch.tensor([1000, 750, 500, 250], device=device, dtype=torch.long)
    return all_timesteps[1000 - original_schedule].to(dtype=torch.float32)


def build_sf_scheduler_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.linspace(1.0, 0.0, 1001, device=device, dtype=torch.float32)[:-1]
    sigmas = 5.0 * sigmas / (1.0 + 4.0 * sigmas)
    timesteps = torch.cat([sigmas * 1000.0, torch.tensor([0.0], device=device, dtype=torch.float32)])
    return timesteps, sigmas


def lookup_sf_sigma(timestep, scheduler_timesteps, scheduler_sigmas):
    timestep = timestep.to(device=scheduler_timesteps.device, dtype=scheduler_timesteps.dtype)
    flat_timestep = timestep.reshape(-1)
    timestep_id = torch.argmin((scheduler_timesteps.unsqueeze(0) - flat_timestep.unsqueeze(1)).abs(), dim=1)
    return scheduler_sigmas[timestep_id].reshape(timestep.shape)


def convert_sf_flow_to_x0(flow_pred, xt, timestep, scheduler_timesteps, scheduler_sigmas):
    sigma = lookup_sf_sigma(timestep, scheduler_timesteps, scheduler_sigmas).to(torch.float64)
    sigma = sigma.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    x0_pred = xt.to(torch.float64) - sigma * flow_pred.to(torch.float64)
    return x0_pred.to(flow_pred.dtype)


def add_sf_noise(x0_pred, noise, timestep, scheduler_timesteps, scheduler_sigmas):
    sigma = lookup_sf_sigma(timestep, scheduler_timesteps, scheduler_sigmas)
    sigma = sigma.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    noisy = (1.0 - sigma) * x0_pred + sigma * noise
    return noisy.to(noise.dtype)


def sample_sf_renoise(latents, *, generator=None):
    batch_size, channels, num_frames, height, width = latents.shape
    noise = torch.randn(
        (batch_size, num_frames, channels, height, width),
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return noise.permute(0, 2, 1, 3, 4).contiguous()
